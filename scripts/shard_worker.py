#!/usr/bin/env python3
"""
shard_worker.py — один шард тяжёлой работы на бесплатном раннере GitHub.

Режимы:
  whisper --shard i --of n   транскрипт своей части аудио (faster-whisper large-v3)
  caption --shard i --of n   VLM-подписи своих кадров (Ollama, модель из inputs)

Вход: ассеты релиза, подготовленные песочницей (audio-part-*.opus, frames-shard-*.zip, meta.json).
Выход: shard-<mode>-<i>.json (артефакт + коммит в results/<run>/shards/).
"""
import argparse, base64, json, os, re, shutil, subprocess, sys, time, urllib.request

WORK = "/tmp/w"
CAPTION_PROMPT_RU = ("Опиши, что происходит в этом кадре видео: кто в кадре, что делает, "
                     "объекты, надписи на экране, место действия. Одно-два предложения, "
                     "только то, что реально видно.")
CAPTION_PROMPT_EN = ("Describe what happens in this video frame: who is visible, what they do, "
                     "objects, on-screen text, setting. One or two sentences, only what is visible.")


def log(*a):
    print("[shard]", *a, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def download_asset(repo, tag, name, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name)
    url = f"https://github.com/{repo}/releases/download/{tag}/{name}"
    with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    return dst


def load_meta(path):
    return json.load(open(path, encoding="utf-8"))


# ---------------------------------------------------------------- whisper shard
def do_whisper(a):
    repo = os.environ["GITHUB_REPOSITORY"]
    meta = load_meta(download_asset(repo, a.tag, "meta.json", WORK))
    ap_info = (meta.get("audio_parts") or [{}])[a.shard] if meta.get("audio_parts") else {}
    name = ap_info.get("file") or meta.get("audio") or "audio.opus"
    off = float(ap_info.get("start", 0.0))
    end = float(ap_info.get("end", 1e9))
    src = download_asset(repo, a.tag, name, WORK)

    wav = os.path.join(WORK, f"p{a.shard}.wav")
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav])
    dur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", wav]).stdout or 0)
    log(f"часть {a.shard}: {dur:.0f}s аудио, смещение {off:.0f}s, модель {a.model}")

    from faster_whisper import WhisperModel
    t0 = time.time()
    model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
    segs, info = model.transcribe(wav, language=a.lang or None, beam_size=5,
                                  vad_filter=False, word_timestamps=True,
                                  condition_on_previous_text=False)
    out = []
    for s in segs:
        t = s.start + off
        if t > end:
            break
        words = [{"t": round(w.start + off, 2), "w": w.word} for w in (s.words or [])]
        out.append({"start": round(t, 2), "end": round(s.end + off, 2), "text": s.text.strip(),
                    "words": words})
    speed = dur / max(1e-6, time.time() - t0)
    log(f"готово: {len(out)} сегментов за {time.time()-t0:.0f}s ({speed:.2f}x реалтайма)")
    return {"mode": "whisper", "shard": a.shard, "of": a.of, "model": a.model,
            "lang": info.language, "segments": out, "speed_x_realtime": round(speed, 2)}


# ---------------------------------------------------------------- caption shard
def do_caption(a):
    repo = os.environ["GITHUB_REPOSITORY"]
    zpath = download_asset(repo, a.tag, f"frames-shard-{a.shard}.zip", WORK)
    meta = load_meta(download_asset(repo, a.tag, "meta.json", WORK))
    import zipfile
    fdir = os.path.join(WORK, f"frames{a.shard}")
    os.makedirs(fdir, exist_ok=True)
    zipfile.ZipFile(zpath).extractall(fdir)

    if shutil.which("ollama") is None:
        log("ставим ollama…")
        run(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"])
    env = dict(os.environ, OLLAMA_MODELS=os.path.join(WORK, "ollama"), OLLAMA_HOST="127.0.0.1:11434")
    os.makedirs(env["OLLAMA_MODELS"], exist_ok=True)
    subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(45):
        time.sleep(2)
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
            break
        except Exception:
            continue
    log("тянем", a.model)
    r = run(["ollama", "pull", a.model], env=env)
    if r.returncode != 0:
        return {"mode": "caption", "shard": a.shard, "of": a.of, "model": a.model,
                "error": "ollama pull failed: " + (r.stderr or "")[-300:], "captions": []}

    declared = {f["file"]: f["t"] for f in meta.get("frames", [])}
    files = sorted(f for f in os.listdir(fdir) if f.endswith(".jpg"))
    prompt = CAPTION_PROMPT_EN if a.model.startswith("moondream") else CAPTION_PROMPT_RU
    caps, t0 = [], time.time()
    for i, name in enumerate(files, 1):
        b64 = base64.b64encode(open(os.path.join(fdir, name), "rb").read()).decode()
        body = json.dumps({"model": a.model, "prompt": prompt, "images": [b64], "stream": False,
                           "keep_alive": "30m",
                           "options": {"temperature": 0.1, "num_predict": 48,
                                       "repeat_penalty": 1.3, "stop": ["\n\n"]}}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        t0f = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                txt = json.loads(resp.read()).get("response", "").strip()
            if not txt:
                txt = "(пустой ответ модели)"
        except Exception as e:
            txt = f"(ошибка: {type(e).__name__})"
        caps.append({"t": declared.get(name), "file": name, "text": txt,
                     "sec": round(time.time() - t0f, 1)})
        if i % 3 == 0 or i == len(files):
            el = time.time() - t0
            log(f"  {i}/{len(files)} ({el/i:.1f} с/кадр) → {txt[:70]}")
    log(f"готово: {len(caps)} подписей за {time.time()-t0:.0f}s")
    return {"mode": "caption", "shard": a.shard, "of": a.of, "model": a.model, "captions": caps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["whisper", "caption"])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--tag", required=True, help="release tag с ассетами")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", default=None)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    res = do_whisper(a) if a.mode == "whisper" else do_caption(a)

    out = os.environ.get("OUT_DIR", "results")
    run_dir = os.environ.get("RUN_DIR", "current")
    ddir = os.path.join(out, run_dir, "shards")
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, f"{a.mode}-{a.shard:03d}.json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    log("записано:", path)


if __name__ == "__main__":
    main()
