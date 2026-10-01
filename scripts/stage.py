#!/usr/bin/env python3
"""
stage.py — подготовка в песочнице (2 ядра / 1.9 ГБ, но зато YouTube тут открыт).

Что делает:
  1. качает видео (yt-dlp) — в песочнице, потому что IP дата-центров GitHub YouTube блокирует
  2. кадры: равномерная сетка + кадры на смене сцен (ffmpeg scene detect), дедуп, ★ для сцен
  3. контактные листы с таймкодами (PIL) — чтобы агент мог «посмотреть» всё видео пачками
  4. аудио: один opus 24 кбит/с + нарезка на N частей с перекрытием 2 с (для шардов Whisper)
  5. frames-shard-*.zip — по шарду на job
  6. meta.json — полное описание прогона

Запуск:
  python3 stage.py --url <ссылка> --out-dir stage/ [--every 4] [--max-frames 700]
                   [--whisper-shards 4] [--caption-shards 8] [--start 0:00] [--end 0:00]
                   [--sheet-cols 5] [--sheet-rows 5] [--tile 320]
"""
import argparse, json, math, os, re, shutil, subprocess, sys, zipfile

try:
    from static_ffmpeg import run as _sf
    FFMPEG = _sf.get_or_fetch_platform_executables_else_raise()[0]
except Exception:
    FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def log(*a):
    print("[stage]", *a, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def mmss(sec):
    sec = max(0, int(sec or 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def duration_of(path):
    r = run([FFMPEG, "-hide_banner", "-i", path])
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
    return (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else None


# ---------------------------------------------------------------- download
def download(url, work, height=480):
    j = run([sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-playlist", "-J", url])
    if j.returncode != 0:
        sys.exit("yt-dlp metadata failed:\n" + j.stderr[-800:])
    info = json.loads(j.stdout)
    os.makedirs(work, exist_ok=True)
    # YouTube отдаёт 403 разным клиентам по-разному — перебираем, пока не получим видеопоток
    clients = ["android", "default", "ios,mweb", "tv_simply,android_vr"]
    last = ""
    for client in clients:
        for old in os.listdir(work):
            if old.startswith("source"):
                os.remove(os.path.join(work, old))
        r = run([sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-playlist",
                 "--extractor-args", f"youtube:player_client={client}",
                 "--ffmpeg-location", os.path.dirname(FFMPEG),
                 "-f", f"bv*[height<={height}]+ba/b[height<={height}]/b",
                 "--merge-output-format", "mp4", "-o", os.path.join(work, "source.%(ext)s"), url])
        files = [os.path.join(work, f) for f in os.listdir(work)
                 if os.path.getsize(os.path.join(work, f)) > 1000 and "source" in f]
        with_video = [p for p in files if "Video:" in run([FFMPEG, "-hide_banner", "-i", p]).stderr]
        if with_video:
            log(f"клиент {client}: ок ({os.path.getsize(with_video[0])//1024} КБ)")
            return max(with_video, key=os.path.getsize), info
        last = (r.stderr or r.stdout or "")[-400:]
        log(f"клиент {client}: не вышло")
    sys.exit("не удалось скачать видео:\n" + last)


# ---------------------------------------------------------------- frames
def scene_times(media, thr=0.3):
    r = run([FFMPEG, "-v", "error", "-i", media, "-vf",
             f"select='gt(scene,{thr})',metadata=print:file=-", "-an", "-f", "null", "-"])
    ts = []
    for m in re.finditer(r"pts_time:([0-9.]+)", r.stdout or ""):
        ts.append(float(m.group(1)))
    return ts


def extract_grid(media, fdir, every, max_frames):
    """равномерная сетка кадров одним проходом ffmpeg"""
    os.makedirs(fdir, exist_ok=True)
    fps = 1.0 / max(0.5, every)
    r = run([FFMPEG, "-v", "error", "-y", "-i", media, "-vf", f"fps={fps:.6f}",
             "-q:v", "4", os.path.join(fdir, "f_%05d.jpg")])
    files = sorted(f for f in os.listdir(fdir) if f.startswith("f_"))
    if not files:
        sys.exit("кадры не извлеклись: " + (r.stderr or "")[-500:])
    if len(files) > max_frames:                      # прореживаем равномерно
        step = math.ceil(len(files) / max_frames)
        keep = set(files[::step])
        for f in files:
            if f not in keep:
                os.remove(os.path.join(fdir, f))
        files = sorted(keep)
    return files, fps


# ---------------------------------------------------------------- sheets
FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/dejavu/DejaVuSans.ttf"]


def make_sheets(fdir, frames, out_dir, cols, rows, tile, scene_flags):
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            font = ImageFont.truetype(p, max(14, tile // 14))
            break
    if font is None:
        font = ImageFont.load_default()
    os.makedirs(out_dir, exist_ok=True)
    per = cols * rows
    made = []
    for si in range(math.ceil(len(frames) / per)):
        chunk = frames[si * per:(si + 1) * per]
        th = tile * 9 // 16
        sheet = Image.new("RGB", (cols * tile, rows * (th + 26)), (18, 18, 22))
        d = ImageDraw.Draw(sheet)
        for i, (t, name) in enumerate(chunk):
            r, c = divmod(i, cols)
            p = os.path.join(fdir, name)
            try:
                im = Image.open(p).convert("RGB").resize((tile, th))
            except Exception:
                continue
            x, y = c * tile, r * (th + 26)
            sheet.paste(im, (x, y + 26))
            mark = "★" if scene_flags.get(name) else " "
            d.text((x + 4, y + 4), f"{mark}[{mmss(t)}] {si*per+i:04d}", fill=(240, 240, 240), font=font)
        dst = os.path.join(out_dir, f"sheet-{si:02d}.jpg")
        sheet.save(dst, quality=86, optimize=True)
        made.append(dst)
    return made


# ---------------------------------------------------------------- audio
def prepare_audio(media, out_dir, shards, bitrate="24k"):
    audio = os.path.join(out_dir, "audio.opus")
    r = run([FFMPEG, "-v", "error", "-y", "-i", media, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libopus", "-b:a", bitrate, audio])
    if not os.path.exists(audio):
        audio = os.path.join(out_dir, "audio.m4a")
        run([FFMPEG, "-v", "error", "-y", "-i", media, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "aac", "-b:a", "32k", audio])
    dur = duration_of(audio) or duration_of(media) or 0
    parts, overlap = [], 2.0
    step = dur / shards if shards else dur
    for i in range(shards):
        s = i * step
        e = min(dur, s + step)
        if e - s < 1:
            break
        part = os.path.join(out_dir, f"audio-part-{i}.opus")
        run([FFMPEG, "-v", "error", "-y", "-ss", f"{s:.2f}", "-to", f"{min(dur, e + overlap):.2f}",
             "-i", audio, "-c", "copy", part])
        if not os.path.exists(part) or os.path.getsize(part) < 1000:
            run([FFMPEG, "-v", "error", "-y", "-ss", f"{s:.2f}", "-to",
                 f"{min(dur, e + overlap):.2f}", "-i", audio, "-c:a", "libopus", "-b:a", bitrate, part])
        parts.append({"file": os.path.basename(part), "start": round(s, 2), "end": round(e, 2)})
    return audio, parts, dur


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out-dir", default="stage")
    ap.add_argument("--every", type=float, default=4.0, help="секунд между кадрами сетки")
    ap.add_argument("--max-frames", type=int, default=700)
    ap.add_argument("--scene-threshold", type=float, default=0.3)
    ap.add_argument("--whisper-shards", type=int, default=4)
    ap.add_argument("--caption-shards", type=int, default=8)
    ap.add_argument("--sheet-cols", type=int, default=5)
    ap.add_argument("--sheet-rows", type=int, default=5)
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    out = os.path.abspath(a.out_dir)
    shutil.rmtree(out, ignore_errors=True)
    for d in ("frames", "sheets", "assets"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    log("качаю видео…")
    media, info = download(a.url, os.path.join(out, "assets"), a.height)
    dur = duration_of(media) or info.get("duration") or 0
    log(f"видео: {mmss(dur)}, {os.path.getsize(media)//1024} КБ")

    log(f"сетка кадров каждые {a.every} с (максимум {a.max_frames})…")
    fdir = os.path.join(out, "frames")
    files, fps = extract_grid(media, fdir, a.every, a.max_frames)
    log(f"сетка: {len(files)} кадров")

    log("поиск смен сцен…")
    sc = scene_times(media, a.scene_threshold)
    log(f"смен сцен: {len(sc)}")

    # приводим сетку к (t, file) и помечаем кадры, ближайшие к смене сцены
    frames = []
    for i, f in enumerate(files):
        t = i / fps
        frames.append((round(t, 2), f))
    scene_flags = {}
    for t in sc:
        if not frames:
            break
        near = min(frames, key=lambda x: abs(x[0] - t))
        if abs(near[0] - t) <= max(1.5, a.every / 2):
            scene_flags[near[1]] = True
    log(f"кадров, помеченных сменой сцены: {len(scene_flags)}")

    log("контактные листы…")
    sheets = make_sheets(fdir, frames, os.path.join(out, "sheets"),
                         a.sheet_cols, a.sheet_rows, a.tile, scene_flags)
    log(f"листов: {len(sheets)} (по {a.sheet_cols*a.sheet_rows} кадров)")

    log("аудио и нарезка на части…")
    audio, parts, adur = prepare_audio(media, os.path.join(out, "assets"), a.whisper_shards)
    log(f"частей аудио: {len(parts)}, всего {mmss(adur)}")

    # шарды кадров для VLM-подписей
    log("упаковка шардов кадров…")
    sc_shards = []
    step = math.ceil(len(frames) / a.caption_shards) if a.caption_shards else len(frames)
    for i in range(a.caption_shards):
        chunk = frames[i * step:(i + 1) * step]
        if not chunk:
            break
        z = os.path.join(out, "assets", f"frames-shard-{i}.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, name in chunk:
                zf.write(os.path.join(fdir, name), name)
        sc_shards.append({"shard": i, "count": len(chunk),
                          "file": os.path.basename(z), "size": os.path.getsize(z)})

    meta = {
        "title": info.get("title"), "uploader": info.get("uploader") or info.get("channel"),
        "duration": dur, "published": info.get("upload_date"), "url": info.get("webpage_url"),
        "chapters": info.get("chapters") or [], "description": (info.get("description") or "")[:6000],
        "frames": [{"t": t, "file": f, "scene": bool(scene_flags.get(f))} for t, f in frames],
        "frame_count": len(frames), "scene_changes": len(sc),
        "sheets": [os.path.basename(s) for s in sheets],
        "sample_every_s": a.every, "audio": os.path.basename(audio),
        "audio_parts": parts, "caption_shards": sc_shards,
        "detected_scene_times": [round(x, 2) for x in sc[:400]],
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    total_mb = sum(os.path.getsize(os.path.join(out, "assets", f))
                   for f in os.listdir(os.path.join(out, "assets"))) / 1e6
    log(f"готово: {out}")
    log(f"  кадров {len(frames)}, листов {len(sheets)}, ассетов {total_mb:.1f} МБ")
    print(json.dumps({"out": out, "frames": len(frames), "sheets": len(sheets),
                      "assets_mb": round(total_mb, 1), "duration": dur}, ensure_ascii=False))


if __name__ == "__main__":
    main()
