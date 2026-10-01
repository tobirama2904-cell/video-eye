#!/usr/bin/env python3
"""
watch_job.py — runs on a GitHub Actions runner (free, unlimited minutes for public repos).

Two modes
---------
1) asset mode (default, works even though datacenter IPs are blocked by YouTube):
   the agent sandbox downloads the video, then uploads `audio.opus`, `frames.zip` and `meta.json`
   to a GitHub release. This job pulls those assets and does the heavy work:
     - Whisper transcript   (weights from OpenAI's CDN, not Hugging Face)
     - per-frame VLM captions via Ollama (registry.ollama.ai, not Hugging Face)
     - report.md, transcript.md, contact sheet
2) direct mode: VID_URL points at a media file the runner can fetch itself (mp4/mkv/m3u8/...).

Env: ASSET_TAG | VID_URL, WHISPER_MODEL, CAPTION_MODE, VID_FRAMES, RUN_LABEL, GITHUB_REPOSITORY
"""
import base64, io, json, os, re, shutil, subprocess, sys, time, urllib.request, zipfile
from datetime import datetime, timezone

WORK = "/tmp/work"
OUT_ROOT = "results"


def log(*a):
    print("[watch]", *a, flush=True)

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def die(msg, code=2):
    log("ERROR:", msg)
    sys.exit(code)

def mmss(sec):
    sec = int(sec or 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def slug(s, n=50):
    s = re.sub(r"[^\w\s-]", "", s or "", flags=re.UNICODE).strip().replace(" ", "-")
    return s[:n] or "video"

def duration_of(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path])
    try:
        return float(r.stdout.strip())
    except Exception:
        r = run(["ffmpeg", "-hide_banner", "-i", path])
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
        return (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else None


# ------------------------------------------------------------------ assets
def gh_api(path, token=None):
    req = urllib.request.Request(f"https://api.github.com{path}",
                                 headers={"Accept": "application/vnd.github+json"})
    if token:
        req.add_header("Authorization", f"token {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_assets(tag, work):
    """pull audio + frames.zip + meta.json from the release created by the sandbox"""
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    rel = gh_api(f"/repos/{repo}/releases/tags/{tag}")
    got = {}
    for a in rel["assets"]:
        dst = os.path.join(work, a["name"])
        log("asset:", a["name"], f"({a['size']} B)")
        urllib.request.urlretrieve(a["browser_download_url"], dst)
        got[a["name"]] = dst
    return got


# ------------------------------------------------------------------ transcript
def transcribe(media, model_name, start=0.0):
    if model_name in ("off", "none", ""):
        return None, "disabled"
    try:
        import whisper
    except ImportError:
        return None, "openai-whisper not installed"
    wav = os.path.join(WORK, "audio16k.wav")
    run(["ffmpeg", "-v", "error", "-y", "-i", media, "-vn", "-ac", "1", "-ar", "16000", wav])
    if not os.path.exists(wav):
        return None, "ffmpeg could not decode audio"
    t0 = time.time()
    try:
        model = whisper.load_model(model_name, download_root=os.path.join(WORK, "whisper"))
        res = model.transcribe(wav, verbose=False, fp16=False, condition_on_previous_text=False)
        segs = [(s["start"] + start, s["text"].strip()) for s in res.get("segments", []) if s.get("text", "").strip()]
        counts = {}
        for _, t in segs:
            k = re.sub(r"[^0-9a-zа-яё]+", "", t.lower())
            counts[k] = counts.get(k, 0) + 1
        segs = [(t, x) for t, x in segs if counts[re.sub(r"[^0-9a-zа-яё]+", "", x.lower())] <= 3]
        log(f"whisper {model_name}: {len(segs)} segments in {time.time()-t0:.0f}s")
        return segs, res.get("language", "?")
    except Exception as e:
        return None, f"whisper failed: {type(e).__name__}: {e}"
    finally:
        if os.path.exists(wav):
            os.remove(wav)


# ------------------------------------------------------------------ captions
def ollama_setup(model, work):
    if shutil.which("ollama") is None:
        log("installing ollama…")
        run(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"])
        if shutil.which("ollama") is None:
            return None, "ollama install failed"
    env = dict(os.environ, OLLAMA_MODELS=os.path.join(work, "ollama"), OLLAMA_HOST="127.0.0.1:11434")
    os.makedirs(env["OLLAMA_MODELS"], exist_ok=True)
    subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(45):
        time.sleep(2)
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
            break
        except Exception:
            continue
    log("pulling", model)
    r = run(["ollama", "pull", model], env=env)
    if r.returncode != 0:
        return None, "ollama pull failed: " + (r.stderr or "")[-300:]
    return env, None


def caption_frames(frames, fdir, model):
    prompt = ("Опиши, что происходит в этом кадре видео: кто и что в кадре, объекты, надписи, "
              "место действия. 1-2 предложения, только то, что видно.")
    out = []
    for i, (t, name) in enumerate(frames, 1):
        with open(os.path.join(fdir, name), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        body = json.dumps({"model": model, "prompt": prompt, "images": [b64], "stream": False,
                           "options": {"temperature": 0.1, "num_predict": 80}}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                txt = json.loads(resp.read()).get("response", "").strip()
        except Exception as e:
            txt = f"(caption failed: {e})"
        out.append((t, txt))
        log(f"caption {i}/{len(frames)} @{mmss(t)}: {txt[:70]}")
    return out


def contact_sheet(fdir, out, cols=5):
    jpgs = sorted(x for x in os.listdir(fdir) if x.endswith(".jpg"))
    if not jpgs:
        return None
    dst = os.path.join(out, "sheet.jpg")
    run(["ffmpeg", "-v", "error", "-y", "-pattern_type", "glob", "-i", os.path.join(fdir, "*.jpg"),
         "-filter_complex", f"scale=320:-2,tile={cols}x{max(1, (len(jpgs) + cols - 1) // cols)}",
         "-frames:v", "1", dst])
    return dst if os.path.exists(dst) else None


# ------------------------------------------------------------------ report
def write_report(out, meta, frames, lines, lang, captions, args):
    title = meta.get("title") or "video"
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"- channel: {meta.get('uploader') or '—'}\n")
        f.write(f"- duration: {mmss(meta.get('duration'))}   |   frames: {len(frames)}\n")
        if meta.get("published"):
            f.write(f"- published: {meta['published']}\n")
        if meta.get("url"):
            f.write(f"- url: {meta['url']}\n")
        f.write(f"- pipeline: GitHub Actions runner (free) · whisper `{args['whisper_model']}`"
                f" · captions `{args['caption']}`\n")
        if meta.get("chapters"):
            f.write("\n## Chapters\n")
            for c in meta["chapters"]:
                f.write(f"- [{mmss(c.get('start_time', 0))}] {c.get('title')}\n")
        if captions:
            f.write(f"\n## Frame captions ({args['caption']})\n\n")
            for t, txt in captions:
                f.write(f"**[{mmss(t)}]** {txt}\n\n")
        f.write(f"\n## Transcript (whisper `{args['whisper_model']}`, language={lang})\n\n")
        if lines:
            for t, txt in lines:
                f.write(f"[{mmss(t)}] {txt}\n")
        else:
            f.write("_no transcript_\n")
        f.write("\n## Frame index\n\n")
        for t, name in frames:
            f.write(f"- [{mmss(t)}] frames/{name}\n")
        if meta.get("description"):
            f.write("\n## Description\n\n" + str(meta["description"])[:2000] + "\n")

    with open(os.path.join(out, "transcript.md"), "w", encoding="utf-8") as f:
        f.write(f"# Transcript — {title}\n\n")
        for t, txt in (lines or []):
            f.write(f"[{mmss(t)}] {txt}\n")

    json.dump({"meta": meta, "frames": [{"t": t, "file": n} for t, n in frames],
               "captions": [{"t": t, "text": x} for t, x in (captions or [])],
               "whisper_lang": lang, "args": args},
              open(os.path.join(out, "info.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)


# ------------------------------------------------------------------ main
def main():
    tag = os.environ.get("ASSET_TAG", "").strip()
    direct = os.environ.get("VID_URL", "").strip()
    args = {
        "asset_tag": tag or None,
        "url": direct or None,
        "whisper_model": (os.environ.get("WHISPER_MODEL") or "small").strip(),
        "caption": (os.environ.get("CAPTION_MODE") or "off").strip(),
        "run_label": os.environ.get("RUN_LABEL", ""),
    }
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)

    if tag:
        log("asset mode, tag:", tag)
        assets = fetch_assets(tag, WORK)
        audio = next((p for n, p in assets.items() if n.startswith("audio")), None)
        frames_zip = next((p for n, p in assets.items() if n.endswith(".zip")), None)
        meta = json.load(open(assets["meta.json"], encoding="utf-8")) if "meta.json" in assets else {}
        if not audio:
            die("release has no audio asset")
    elif direct:
        log("direct mode:", direct)
        out_media = os.path.join(WORK, "source.%(ext)s")
        r = run(["yt-dlp", "--no-warnings", "--no-playlist", "--ffmpeg-location", "/usr/bin",
                 "-f", "bv*[height<=720]+ba/b[height<=720]/b", "--merge-output-format", "mp4",
                 "-o", out_media, direct])
        files = [os.path.join(WORK, f) for f in os.listdir(WORK)
                 if os.path.getsize(os.path.join(WORK, f)) > 1000 and not f.endswith(".part")]
        with_video = [p for p in files if "Video:" in run(["ffmpeg", "-hide_banner", "-i", p]).stderr]
        if not with_video:
            die("could not download media:\n" + (r.stderr or "")[-800:])
        media = max(with_video, key=os.path.getsize)
        audio, frames_zip, meta = media, None, {"title": os.path.basename(media), "url": direct,
                                                "duration": duration_of(media)}
    else:
        die("set ASSET_TAG or VID_URL")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = os.path.join(OUT_ROOT, f"{stamp}-{slug(str(meta.get('title') or 'video'))}")
    fdir = os.path.join(out, "frames")
    os.makedirs(fdir, exist_ok=True)

    frames = []
    if frames_zip and os.path.exists(frames_zip):
        zf = zipfile.ZipFile(frames_zip)
        for n in zf.namelist():
            if n.endswith(".jpg"):
                open(os.path.join(fdir, os.path.basename(n)), "wb").write(zf.read(n))
        declared = {fr["file"]: fr["t"] for fr in meta.get("frames", []) if "file" in fr}
        for f in sorted(os.listdir(fdir)):
            if f in declared:                      # timestamps come from the sandbox, authoritative
                frames.append((float(declared[f]), f))
            else:                                  # fallback: parse HH-MM-SS / MM-SS out of the name
                mm = re.search(r"t=(\d+)(?:-(\d+))?(?:-(\d+))?", f)
                if mm:
                    parts = [int(x) for x in mm.groups() if x is not None]
                    t = parts[0] * 60 + parts[1] if len(parts) == 2 else (parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0])
                    frames.append((t, f))
        log(f"frames from asset: {len(frames)}")

    if not frames and audio and audio.endswith((".mp4", ".mkv", ".webm")):
        n = int(os.environ.get("VID_FRAMES") or 60)
        dur = meta.get("duration") or duration_of(audio) or 60
        fps = min(2.0, max(0.05, n / max(1.0, dur)))
        run(["ffmpeg", "-v", "error", "-y", "-i", audio, "-vf", f"fps={fps:.6f},scale=768:-2",
             "-q:v", "4", "-frames:v", str(n), os.path.join(fdir, "f_%04d.jpg")])
        for f in sorted(os.listdir(fdir)):
            mm = re.match(r"f_(\d+)\.jpg$", f)
            if mm:
                idx = int(mm.group(1)) - 1
                t = idx / fps
                nice = f"{idx:04d}_t={mmss(t).replace(':', '-')}.jpg"
                os.rename(os.path.join(fdir, f), os.path.join(fdir, nice))
                frames.append((t, nice))
        frames.sort(key=lambda x: x[1])

    sheet = contact_sheet(fdir, out) if frames else None

    log(f"transcribing with whisper {args['whisper_model']}…")
    lines, lang = transcribe(audio, args["whisper_model"], float(meta.get("start") or 0))

    captions = None
    if frames and args["caption"] not in ("off", "none", ""):
        env, err = ollama_setup(args["caption"], WORK)
        if env:
            pick = frames if len(frames) <= 60 else frames[:: max(1, len(frames) // 60)]
            captions = caption_frames(pick, fdir, args["caption"])
        else:
            log("caption skipped:", err)

    write_report(out, meta, frames, lines, lang, captions, args)
    open(os.path.join(OUT_ROOT, "latest.txt"), "w").write(os.path.basename(out) + "\n")
    json.dump({"run_id": args["run_label"], "dir": os.path.basename(out), "title": meta.get("title"),
               "duration": meta.get("duration"), "frames": len(frames),
               "transcript_segments": len(lines or []), "lang": lang, "captions": bool(captions),
               "url": meta.get("url"), "sheet": bool(sheet), "asset_tag": tag},
              open(os.path.join(OUT_ROOT, "latest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log("done ->", out)


if __name__ == "__main__":
    main()
