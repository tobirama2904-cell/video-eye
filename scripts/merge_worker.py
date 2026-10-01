#!/usr/bin/env python3
"""
merge_worker.py — финальная сборка на раннере: транскрипт + подписи кадров + события звука
в одну хронологическую ленту timeline.md, плюс audio-events.json и report.md.

Звук размечается по-настоящему: RMS/спектральная плоскостность/центроид по секундам (librosa,
никакого Hugging Face) → речь / музыка / шум-аплодисменты / тишина.
"""
import json, glob, os, subprocess, sys, math

AUDIO = os.environ.get("AUDIO_ASSET", "audio.opus")


def log(*a):
    print("[merge]", *a, flush=True)


def mmss(sec):
    sec = max(0, int(sec or 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ------------------------------------------------------------------ audio events
def audio_events(path, speech_intervals):
    try:
        import librosa, numpy as np
    except ImportError:
        log("librosa нет — пропускаю разметку звука")
        return []
    wav = "/tmp/merge-audio.wav"                    # декодируем ffmpeg'ом: opus читается не везде
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-ac", "1",
                    "-ar", "16000", wav], check=False)
    src = wav if os.path.exists(wav) else path
    y, sr = librosa.load(src, sr=16000, mono=True)
    hop = sr
    rms = librosa.feature.rms(y=y, frame_length=2 * sr, hop_length=hop)[0]
    flat = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(y=y, n_fft=2048, hop_length=hop)[0]
    n = min(len(rms), len(flat), len(cent))
    db = 20 * np.log10(np.maximum(rms[:n], 1e-6))
    events = []
    for i in range(n):
        t = i
        speaking = any(a <= t <= b for a, b in speech_intervals)
        dbf, fl, ce = db[i], flat[i], cent[i]
        if dbf < -45:
            label = "тишина"
        elif speaking:
            label = "речь"
        elif fl > 0.25 and dbf > -30:
            label = "шум/аплодисменты"
        elif fl < 0.12 and dbf > -35:
            label = "музыка"
        else:
            label = "фон"
        events.append((t, label, float(dbf), float(fl), float(ce)))
    # слить соседние одинаковые
    merged = []
    for t, label, dbf, fl, ce in events:
        if merged and merged[-1]["label"] == label:
            merged[-1]["end"] = t + 1
            merged[-1]["peak_db"] = max(merged[-1]["peak_db"], dbf)
        else:
            merged.append({"start": t, "end": t + 1, "label": label, "peak_db": dbf,
                           "flatness": round(fl, 3), "centroid": round(ce)})
    return [e for e in merged if e["end"] - e["start"] >= 2]


# ------------------------------------------------------------------ timeline
def bucket(items, step=10):
    """собирает единую ленту с шагом step секунд"""
    lines = []
    for it in items:
        lines.append(it)
    lines.sort(key=lambda x: x["t"])
    return lines


def main():
    run_dir = os.environ.get("RUN_DIR", "current")
    base = os.path.join("results", run_dir)
    shards = sorted(glob.glob(os.path.join(base, "shards", "*.json")))
    log(f"шардов: {len(shards)}")

    segs, caps, langs, speeds = [], [], set(), {}
    for p in shards:
        d = json.load(open(p, encoding="utf-8"))
        if d.get("mode") == "whisper":
            segs += d.get("segments", [])
            if d.get("lang"):
                langs.add(d["lang"])
            speeds[f"whisper:{d.get('model')}"] = d.get("speed_x_realtime")
        elif d.get("mode") == "caption":
            caps += d.get("captions", [])
            speeds[f"vlm:{d.get('model')}"] = None
    segs.sort(key=lambda s: s["start"])
    caps = [c for c in caps if c.get("t") is not None]
    caps.sort(key=lambda c: c["t"])
    log(f"сегментов речи: {len(segs)}, подписей кадров: {len(caps)}")

    meta = json.load(open(os.path.join(base, "meta.json"), encoding="utf-8")) if os.path.exists(
        os.path.join(base, "meta.json")) else {}
    speech_iv = [(s["start"], s["end"]) for s in segs]

    audio_path = os.environ.get("AUDIO_PATH", "")
    ev = audio_events(audio_path, speech_iv) if audio_path and os.path.exists(audio_path) else []
    log(f"событий звука: {len(ev)}")

    # --- единая лента: каждые 10 секунд — что слышно и что видно
    tl = []
    if ev or segs or caps:
        end = max([c["t"] for c in caps] + [s["end"] for s in segs] + [e["end"] for e in ev] + [0])
        t = 0
        while t < end + 10:
            bucket_caps = [c for c in caps if t <= c["t"] < t + 10]
            bucket_segs = [s for s in segs if t <= s["start"] < t + 10]
            bucket_ev = [e for e in ev if not (e["end"] <= t or e["start"] >= t + 10)]
            labels = []
            for e in bucket_ev:
                if e["label"] not in labels:
                    labels.append(e["label"])
            tl.append({"t": t, "speech": " ".join(s["text"] for s in bucket_segs)[:400],
                       "audio": ", ".join(labels),
                       "visual": [c["text"] for c in bucket_caps][:2]})
            t += 10

    # --- файлы
    with open(os.path.join(base, "timeline.md"), "w", encoding="utf-8") as f:
        f.write(f"# Хронология: {meta.get('title','video')}\n\n")
        f.write(f"длительность {mmss(meta.get('duration'))} · кадров с подписями {len(caps)} · "
                f"реплик {len(segs)} · интервал {len(ev)} секций звука\n\n")
        for row in tl:
            f.write(f"**[{mmss(row['t'])}]**")
            if row["audio"]:
                f.write(f" 🔊 {row['audio']}")
            if row["speech"]:
                f.write(f"\n  🗣 {row['speech']}")
            for v in row["visual"]:
                f.write(f"\n  👁 {v}")
            f.write("\n\n")

    with open(os.path.join(base, "transcript.md"), "w", encoding="utf-8") as f:
        f.write(f"# Транскрипт: {meta.get('title','video')}\n\n")
        for s in segs:
            f.write(f"[{mmss(s['start'])}] {s['text']}\n")

    json.dump({"segments": segs, "captions": caps, "audio_events": ev, "timeline": tl,
               "speed": speeds, "languages": sorted(langs)},
              open(os.path.join(base, "analysis.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    with open(os.path.join(base, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# {meta.get('title','video')} — отчёт просмотра\n\n")
        f.write(f"- канал: {meta.get('uploader') or '—'}\n")
        f.write(f"- длительность: {mmss(meta.get('duration'))}\n")
        f.write(f"- url: {meta.get('url') or '—'}\n")
        f.write(f"- покрытие: **{len(caps)} кадров с подписями**, {len(segs)} реплик речи, "
                f"{len(ev)} секций аудио-разметки\n")
        f.write(f"- модели: whisper {os.environ.get('WHISPER_MODEL','?')}, "
                f"VLM {os.environ.get('CAPTION_MODEL','?')}; шардов {len(shards)}, "
                f"скорость {speeds}\n")
        if meta.get("chapters"):
            f.write("\n## Главы\n")
            for c in meta["chapters"]:
                f.write(f"- [{mmss(c.get('start_time',0))}] {c.get('title')}\n")
        f.write("\n## Что где звучит (первые 40 секций)\n\n")
        for e in ev[:40]:
            f.write(f"- [{mmss(e['start'])}–{mmss(e['end'])}] {e['label']} ({e['peak_db']:.0f} dB)\n")
        f.write("\n## Кадры с подписью (каждый 3-й)\n\n")
        for c in caps[::3]:
            f.write(f"**[{mmss(c['t'])}]** {c['text']}\n\n")
        f.write("\n## Полный транскрипт\n\n")
        for s in segs:
            f.write(f"[{mmss(s['start'])}] {s['text']}\n")
        if meta.get("description"):
            f.write("\n## Описание\n\n" + str(meta["description"])[:1500] + "\n")
    log("готово: report.md, timeline.md, transcript.md, analysis.json")


if __name__ == "__main__":
    main()
