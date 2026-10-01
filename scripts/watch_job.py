#!/usr/bin/env python3
"""
watch_job.py — runs on a GitHub Actions runner (free, unlimited for public repos).

Pipeline (no Hugging Face anywhere):
  1. yt-dlp  -> download the video (ffmpeg from apt does the merging)
  2. ffmpeg  -> duration-aware frame sampling (<=2 fps, <=200 frames) + contact sheet
  3. whisper -> timestamped transcript. Weights come from OpenAI's CDN (openaipublic.azureedge.net),
                not from HF. Model can be switched off.
  4. ollama  -> optional per-frame captions with a local VLM (registry.ollama.ai, not HF)
  5. writes results/<timestamp>-<slug>/{report.md,transcript.md,frames/*.jpg,sheet.jpg,info.json}
"""
import base64, json, os, re, shutil, subprocess, sys, time, urllib.request
from datetime import datetime, timezone

OUT_ROOT = "results"
WORK = "/tmp/work"


# ------------------------------------------------------------------ helpers
def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def log(*a):
    print("[watch]", *a, flush=True)

def die(msg, code=2):
    log("ERROR:", msg)
    sys.exit(code)

def parse_ts(s):
    if not s:
        return None
    s = str(s).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", s)
    if not m:
        raise ValueError(f"bad timestamp: {s}")
    h, mi, se = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + float(se)

def mmss(sec):
    sec = int(sec or 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def slug(s, n=50):
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE).strip().replace(" ", "-")
    return s[:n] or "video"

def duration_of(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path])
    try:
        return float(r.stdout.strip())
    except Exception:
        r = run(["ffmpeg", "-hide_banner", "-i", path])
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
        if m:
            h, mi, se = m.groups()
            return int(h) * 3600 + int(mi) * 60 + float(se)
    return None


# ------------------------------------------------------------------ 1. download
def download(url, work, start, end):
    os.makedirs(work, exist_ok=True)
    j = run(["yt-dlp", "--no-warnings", "--no-playlist", "-J", url])
    if j.returncode != 0:
        die("yt-dlp metadata failed:\n" + j.stderr[-1500:])
    info = json.loads(j.stdout)

    sect = []
    if start is not None:
        sect = ["--download-sections", f"*{start}-{end if end is not None else 'inf'}"]
    r = run(["yt-dlp", "--no-warnings", "--no-playlist", "--ffmpeg-location", "/usr/bin"] + sect +
            ["-f", "bv*[height<=720]+ba/b[height<=720]/b", "--merge-output-format", "mp4",
             "-o", os.path.join(work, "source.%(ext)s"), url])
    files = [os.path.join(work, f) for f in os.listdir(work)
             if os.path.getsize(os.path.join(work, f)) > 1000]
    vids = [p for p in files if "Video:" in run(["ffmpeg", "-hide_banner", "-i", p]).stderr]
    if not vids:
        die("no video stream downloaded:\n" + (r.stderr or "")[-1200:])
    return max(vids, key=os.path.getsize), info


# ------------------------------------------------------------------ 2. frames
def extract_frames(media, out, n, width, start, end, duration):
    fdir = os.path.join(out, "frames")
    os.makedirs(fdir, exist_ok=True)
    off = start or 0
    seg = (end - start) if (start is not None and end is not None) else max(1.0, (duration or 60) - off)
    fps = min(2.0, max(0.05, n / max(seg, 1)))
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if start is not None and end is not None:
        cmd += ["-t", str(end - start)]
    cmd += ["-i", media, "-vf", f"fps={fps:.6f},scale={width}:-2", "-q:v", "4",
            "-frames:v", str(n), os.path.join(fdir, "f_%04d.jpg")]
    r = run(cmd)
    if r.returncode != 0 and not os.listdir(fdir):
        die("ffmpeg failed:\n" + (r.stderr or "")[-1000:])
    frames = []
    for f in sorted(os.listdir(fdir)):
        m = re.match(r"f_(\d+)\.jpg$", f)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        t = off + idx / fps
        nice = f"{idx:04d}_t={mmss(t).replace(':', '-')}.jpg"
        os.rename(os.path.join(fdir, f), os.path.join(fdir, nice))
        frames.append((t, nice))
    frames.sort(key=lambda x: x[1])
    return frames, fps

def contact_sheet(fdir, out, cols=5):
    jpgs = sorted(x for x in os.listdir(fdir) if x.endswith(".jpg"))
    if not jpgs:
        return None
    dst = os.path.join(out, "sheet.jpg")
    run(["ffmpeg", "-v", "error", "-y", "-pattern_type", "glob", "-i", os.path.join(fdir, "*.jpg"),
         "-filter_complex", f"scale=320:-2,tile={cols}x{max(1, (len(jpgs) + cols - 1) // cols)}",
         "-frames:v", "1", dst])
    return dst if os.path.exists(dst) else None


# ------------------------------------------------------------------ 3. transcript
WHISPER_URL = "https://openaipublic.azureedge.net/main/whisper/models"   # NOT Hugging Face

def transcribe(media, model_name, start):
    if model_name in ("off", "none", ""):
        return None, "disabled"
    try:
        import whisper
    except ImportError:
        return None, "openai-whisper not installed"
    wav = os.path.join(WORK, "audio16k.wav")
    run(["ffmpeg", "-v", "error", "-y", "-i", media, "-vn", "-ac", "1", "-ar", "16000", wav])
    if not os.path.exists(wav):
        return None, "ffmpeg could not extract audio"
    try:
        model = whisper.load_model(model_name, download_root=os.path.join(WORK, "whisper"))
        res = model.transcribe(wav, language=None, verbose=False, fp16=False)
        out = [(seg["start"] + (start or 0), seg["text"].strip())
               for seg in res.get("segments", []) if seg.get("text", "").strip()]
        # drop boilerplate that whisper invents on silence (repeats > 3 times)
        counts = {}
        for _, t in out:
            k = re.sub(r"[^0-9a-zа-яё]+", "", t.lower())
            counts[k] = counts.get(k, 0) + 1
        out = [(t, x) for t, x in out
               if counts[re.sub(r"[^0-9a-zа-яё]+", "", x.lower())] <= 3]
        return out, res.get("language", "?")
    except Exception as e:
        return None, f"whisper failed: {type(e).__name__}: {e}"
    finally:
        if os.path.exists(wav):
            os.remove(wav)


# ------------------------------------------------------------------ 4. captions (Ollama)
def ollama_setup(model):
    if shutil.which("ollama") is None:
        log("installing ollama…")
        r = run(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"])
        if shutil.which("ollama") is None:
            return None, "ollama install failed: " + (r.stderr or "")[-400:]
    env = dict(os.environ, OLLAMA_MODELS=os.path.join(WORK, "ollama"),
               OLLAMA_HOST="127.0.0.1:11434")
    if not os.path.exists(env["OLLAMA_MODELS"]):
        os.makedirs(env["OLLAMA_MODELS"], exist_ok=True)
    subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    for _ in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
            break
        except Exception:
            continue
    log("pulling", model, "…")
    r = run(["ollama", "pull", model], env=env)
    if r.returncode != 0:
        return None, "ollama pull failed: " + (r.stderr or "")[-400:]
    return env, None

def caption_frames(frames, fdir, model, env):
    out = []
    prompt = ("Опиши коротко, что происходит в этом кадре видео: кто и что в кадре, "
              "какие объекты, надписи, место действия. 1-2 предложения, без домыслов.")
    for i, (t, name) in enumerate(frames, 1):
        with open(os.path.join(fdir, name), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        body = json.dumps({"model": model, "prompt": prompt, "images": [b64],
                           "stream": False, "options": {"temperature": 0.2, "num_predict": 90}}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                txt = json.loads(resp.read()).get("response", "").strip()
        except Exception as e:
            txt = f"(caption failed: {e})"
        out.append((t, txt))
        log(f"caption {i}/{len(frames)} t={mmss(t)}: {txt[:70]}")
    return out


# ------------------------------------------------------------------ 5. report
def write_report(out, info, frames, fps, lines, lang, captions, duration, args):
    title = info.get("title") or "video"
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"- channel: {info.get('uploader') or info.get('channel') or '—'}\n")
        f.write(f"- duration: {mmss(duration)}   |   frames: {len(frames)} @ {fps:.3f} fps\n")
        if info.get("upload_date"):
            f.write(f"- published: {info['upload_date']}\n")
        if info.get("webpage_url"):
            f.write(f"- url: {info['webpage_url']}\n")
        if info.get("chapters"):
            f.write("\n## Chapters\n")
            for c in info["chapters"]:
                f.write(f"- [{mmss(c.get('start_time', 0))}] {c.get('title')}\n")
        if captions:
            f.write(f"\n## Frame captions (VLM: {args['caption']})\n\n")
            for t, txt in captions:
                f.write(f"**[{mmss(t)}]** {txt}\n\n")
        f.write(f"\n## Transcript (whisper {args['whisper_model']}, lang={lang})\n\n")
        if lines:
            for t, txt in lines:
                f.write(f"[{mmss(t)}] {txt}\n")
        else:
            f.write("_no transcript_\n")
        f.write("\n## Frame index\n\n")
        for t, name in frames:
            f.write(f"- [{mmss(t)}] frames/{name}\n")
        if info.get("description"):
            f.write("\n## Description\n\n" + info["description"][:2000] + "\n")

    with open(os.path.join(out, "transcript.md"), "w", encoding="utf-8") as f:
        f.write(f"# Transcript — {title}\n\n")
        for t, txt in (lines or []):
            f.write(f"[{mmss(t)}] {txt}\n")

    json.dump({"info": info, "frames": [{"t": t, "file": n} for t, n in frames],
               "captions": [{"t": t, "text": x} for t, x in (captions or [])],
               "args": args},
              open(os.path.join(out, "info.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)


# ------------------------------------------------------------------ main
def main():
    url = os.environ.get("VID_URL", "").strip()
    if not url:
        die("VID_URL is empty")
    args = {
        "url": url,
        "start": os.environ.get("VID_START", "").strip(),
        "end": os.environ.get("VID_END", "").strip(),
        "frames": int(os.environ.get("VID_FRAMES") or 60),
        "width": int(os.environ.get("VID_WIDTH") or 768),
        "whisper_model": (os.environ.get("WHISPER_MODEL") or "small").strip(),
        "caption": (os.environ.get("CAPTION_MODE") or "off").strip(),
        "run_label": os.environ.get("RUN_LABEL", ""),
    }
    n = min(max(args["frames"], 1), 200)
    start, end = parse_ts(args["start"]), parse_ts(args["end"])

    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)

    log("downloading", url)
    media, info = download(url, WORK, start, end)
    duration = duration_of(media)
    log(f"got {os.path.basename(media)} ({duration and mmss(duration)})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = os.path.join(OUT_ROOT, f"{stamp}-{slug(str(info.get('title') or 'video'))}")
    os.makedirs(out, exist_ok=True)

    log(f"sampling {n} frames @ width {args['width']}")
    frames, fps = extract_frames(media, out, n, args["width"], start, end, duration)
    sheet = contact_sheet(os.path.join(out, "frames"), out)
    log(f"frames: {len(frames)}, sheet: {bool(sheet)}")

    log(f"transcribing (whisper {args['whisper_model']})…")
    lines, lang = transcribe(media, args["whisper_model"], start)
    log(f"transcript: {len(lines) if lines else 0} segments, lang={lang}")

    captions = None
    if args["caption"] not in ("off", "none", ""):
        env, err = ollama_setup(args["caption"])
        if env:
            pick = frames if len(frames) <= 60 else frames[:: max(1, len(frames) // 60)]
            captions = caption_frames(pick, os.path.join(out, "frames"), args["caption"], env)
        else:
            log("caption skipped:", err)

    write_report(out, info, frames, fps, lines, lang, captions, duration, args)
    log("report ->", os.path.join(out, "report.md"))

    # make the sandbox-side fetch trivial
    open(os.path.join(OUT_ROOT, "latest.txt"), "w").write(os.path.basename(out) + "\n")
    json.dump({"run_id": args["run_label"], "dir": os.path.basename(out), "title": info.get("title"),
               "duration": duration, "frames": len(frames), "transcript_segments": len(lines or []),
               "lang": lang, "captions": bool(captions), "url": url},
              open(os.path.join(OUT_ROOT, "latest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log("done")


if __name__ == "__main__":
    main()
