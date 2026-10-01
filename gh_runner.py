#!/usr/bin/env python3
"""
gh_runner.py — orchestrator that lives in the agent sandbox and drives the free GitHub Actions
backend for the heavy work (frames, Whisper, local VLM captions).

  export GH_TOKEN=...            # repo + workflow scopes
  python3 gh_runner.py push                       # sync this folder to the GitHub repo
  python3 gh_runner.py run --url "<youtube url>" [--start 1:30] [--end 2:00] \
        [--frames 80] [--width 1024] [--whisper small] [--caption off|moondream|qwen2.5vl:3b] [--wait]
  python3 gh_runner.py fetch [latest|<run_id>]    # pull report.md / transcript / frames into ./runs/

Public repo => unlimited free runner minutes. No API keys, no Hugging Face.
"""
import argparse, io, json, os, re, shutil, subprocess, sys, time, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER_REPO = os.environ.get("VIDEO_EYE_REPO", "tobirama2904-cell/video-eye")
REPO_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "video-eye-repo")
RUNS_DIR = os.path.join(os.path.expanduser("~"), "runs")
def sync_list():
    """everything under scripts/ and .github/workflows/, plus top-level docs"""
    out = []
    for d in ("scripts", ".github/workflows"):
        base = os.path.join(HERE, d)
        for root, _, files in os.walk(base):
            for f in files:
                out.append(os.path.relpath(os.path.join(root, f), HERE))
    for f in ("README.md", "gh_runner.py"):
        if os.path.exists(os.path.join(HERE, f)):
            out.append(f)
    return out

FILES = sync_list()
WORKFLOW = "watch.yml"


def token():
    t = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not t:
        sys.exit("ERROR: set GH_TOKEN (GitHub personal access token with repo+workflow scopes)")
    return t.strip()


def api(path, method="GET", data=None, accept="application/vnd.github+json", raw=False):
    req = urllib.request.Request(f"https://api.github.com{path}", method=method,
                                 headers={"Authorization": f"token {token()}",
                                          "Accept": accept,
                                          "X-GitHub-Api-Version": "2022-11-28"})
    body = json.dumps(data).encode() if data is not None else None
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=body, timeout=60) as r:
            payload = r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"GitHub API {method} {path} -> {e.code}: {e.read()[:400].decode(errors='replace')}")
    return payload if raw else json.loads(payload or b"{}")


def git(args, cwd):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    return r


# ------------------------------------------------------------------ push
def cmd_push(a):
    url = f"https://x-access-token:{token()}@github.com/{OWNER_REPO}.git"
    os.makedirs(os.path.dirname(REPO_CACHE), exist_ok=True)
    if not os.path.isdir(os.path.join(REPO_CACHE, ".git")):
        shutil.rmtree(REPO_CACHE, ignore_errors=True)
        r = git(["clone", url, REPO_CACHE], os.path.expanduser("~"))
        if r.returncode:
            sys.exit("clone failed: " + r.stderr)
    git(["remote", "set-url", "origin", url], REPO_CACHE)
    git(["fetch", "origin"], REPO_CACHE)
    git(["checkout", "main"], REPO_CACHE)
    git(["reset", "--hard", "origin/main"], REPO_CACHE)
    for rel in FILES:
        src = os.path.join(HERE, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(REPO_CACHE, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    git(["add", "-A"], REPO_CACHE)
    r = git(["-c", "user.name=agent", "-c", "user.email=agent@local", "commit",
             "-m", "sync: " + time.strftime("%Y-%m-%d %H:%M:%S")], REPO_CACHE)
    if "nothing to commit" in (r.stdout + r.stderr):
        print("repo already up to date")
        return
    r = git(["push", "origin", "main"], REPO_CACHE)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode:
        sys.exit("push failed: " + r.stderr)
    print("pushed ->", f"https://github.com/{OWNER_REPO}")


# ------------------------------------------------------------------ run
def latest_run():
    runs = api(f"/repos/{OWNER_REPO}/actions/runs?per_page=10")["workflow_runs"]
    return runs[0] if runs else None


def cmd_run(a):
    inputs = {"url": a.url, "frames": str(a.frames), "width": str(a.width),
              "whisper_model": a.whisper, "caption": a.caption}
    if a.start: inputs["start"] = a.start
    if a.end:   inputs["end"] = a.end
    before = {r["id"] for r in api(f"/repos/{OWNER_REPO}/actions/runs?per_page=20")["workflow_runs"]}
    api(f"/repos/{OWNER_REPO}/actions/workflows/{WORKFLOW}/dispatches", "POST",
        {"ref": "main", "inputs": inputs})
    print("dispatched:", inputs)

    run = None
    for _ in range(30):
        time.sleep(4)
        for r in api(f"/repos/{OWNER_REPO}/actions/runs?per_page=20")["workflow_runs"]:
            if r["id"] not in before:
                run = r
                break
        if run:
            break
    if not run:
        sys.exit("could not find the dispatched run")
    print("run:", run["html_url"])

    if not a.wait:
        return run["id"]
    t0 = time.time()
    while True:
        time.sleep(15)
        r = api(f"/repos/{OWNER_REPO}/actions/runs/{run['id']}")
        el = int(time.time() - t0)
        print(f"  [{el:4d}s] {r['status']} / {r.get('conclusion')}", flush=True)
        if r["status"] == "completed":
            print("conclusion:", r["conclusion"])
            if r["conclusion"] != "success":
                jobs = api(f"/repos/{OWNER_REPO}/actions/runs/{run['id']}/jobs")["jobs"]
                for j in jobs:
                    for s in j.get("steps", []):
                        print(f"   {s['name']}: {s['conclusion']}")
            return run["id"]


# ------------------------------------------------------------------ fetch
def raw(path):
    import urllib.parse
    url = f"https://raw.githubusercontent.com/{OWNER_REPO}/main/{urllib.parse.quote(path, safe='/')}"
    req = urllib.request.Request(url, headers={"Authorization": f"token {token()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def cmd_fetch(a):
    meta = json.loads(raw("results/latest.json"))
    if a.run_id and a.run_id not in ("latest",):
        rid = a.run_id
    else:
        rid = str(meta.get("run_id") or "")
    d = meta["dir"]
    dst = os.path.join(RUNS_DIR, d)
    os.makedirs(dst, exist_ok=True)
    for name in ("report.md", "transcript.md", "timeline.md", "analysis.json", "info.json", "sheet.jpg"):
        try:
            data = raw(f"results/{d}/{name}")
            open(os.path.join(dst, name), "wb").write(data)
            print(f"  {name}: {len(data)}B")
        except Exception as e:
            print(f"  {name}: missing ({e})")

    # frames come from the run artifact (keeps the repo small)
    r = latest_run()
    for run in [r] + api(f"/repos/{OWNER_REPO}/actions/runs?per_page=10")["workflow_runs"]:
        arts = api(f"/repos/{OWNER_REPO}/actions/runs/{run['id']}/artifacts")["artifacts"]
        art = next((x for x in arts if x["name"].startswith("frames-")), None)
        if not art:
            continue
        blob = api(f"/repos/{OWNER_REPO}/actions/artifacts/{art['id']}/zip", raw=True,
                   accept="application/vnd.github+json")
        zf = zipfile.ZipFile(io.BytesIO(blob))
        fdir = os.path.join(dst, "frames")
        os.makedirs(fdir, exist_ok=True)
        n = 0
        for m in zf.namelist():
            if m.endswith(".jpg"):
                open(os.path.join(fdir, os.path.basename(m)), "wb").write(zf.read(m))
                n += 1
        print(f"  frames: {n}")
        break

    print("\nfetched ->", dst)
    print(json.dumps(meta, ensure_ascii=False, indent=2))



# ------------------------------------------------------------------ local staging
SANDBOX_WATCH = os.path.expanduser("~/skills/watch-video/watch.py")

def frame_ts(name):
    """0007_t=01-23.jpg / 0007_t=1-02-03.jpg -> seconds"""
    m = re.search(r"t=(\d+)(?:-(\d+))?(?:-(\d+))?", name)
    if not m:
        return None
    p = [int(x) for x in m.groups() if x is not None]
    return p[0] * 60 + p[1] if len(p) == 2 else (p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0])

def ffmpeg_path():
    from static_ffmpeg import run as sf
    return sf.get_or_fetch_platform_executables_else_raise()[0]

def stage_locally(url, stage, frames, width, start, end):
    """download + sample frames in the sandbox (YouTube is reachable here), keep it cheap"""
    os.makedirs(stage, exist_ok=True)
    cmd = [sys.executable, SANDBOX_WATCH, url, "--out", stage, "--max-frames", str(frames),
           "--width", str(width), "--whisper", "never", "--keep"]
    if start: cmd += ["--start", start]
    if end:   cmd += ["--end", end]
    print("staging locally:", " ".join(cmd[-8:]))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("local stage failed:\n" + (r.stdout or "")[-800:] + (r.stderr or "")[-800:])
    info = json.load(open(os.path.join(stage, "info.json"), encoding="utf-8"))
    fdir = os.path.join(stage, "frames")
    frames_list = [{"t": frame_ts(f), "file": f} for f in sorted(os.listdir(fdir)) if f.endswith(".jpg")]
    media = None
    mdir = os.path.join(stage, "media")
    if os.path.isdir(mdir):
        cands = [os.path.join(mdir, f) for f in os.listdir(mdir) if os.path.getsize(os.path.join(mdir, f)) > 1000]
        media = max(cands, key=os.path.getsize) if cands else None

    meta = {"title": info.get("title"), "uploader": info.get("uploader") or info.get("channel"),
            "duration": info.get("duration"), "published": info.get("upload_date"),
            "url": info.get("webpage_url") or url, "chapters": info.get("chapters") or [],
            "description": (info.get("description") or "")[:4000], "frames": frames_list,
            "start": start or "", "end": end or ""}
    json.dump(meta, open(os.path.join(stage, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)

    audio = os.path.join(stage, "audio.opus")
    if media:
        ff = ffmpeg_path()
        r = subprocess.run([ff, "-v", "error", "-y", "-i", media, "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "libopus", "-b:a", "24k", audio], capture_output=True, text=True)
        if not os.path.exists(audio):
            audio = os.path.join(stage, "audio.m4a")
            subprocess.run([ff, "-v", "error", "-y", "-i", media, "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "aac", "-b:a", "32k", audio], check=True)
        shutil.rmtree(mdir, ignore_errors=True)          # media stays in the sandbox, only audio travels
    zpath = os.path.join(stage, "frames.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(fdir)):
            z.write(os.path.join(fdir, f), f)
    return meta, audio if os.path.exists(audio) else None, zpath


def create_release(tag, title):
    rel = api(f"/repos/{OWNER_REPO}/releases", "POST",
              {"tag_name": tag, "name": title, "draft": False, "prerelease": True})
    return rel["id"]

def upload_asset(rel_id, path):
    name = os.path.basename(path)
    data = open(path, "rb").read()
    req = urllib.request.Request(
        f"https://uploads.github.com/repos/{OWNER_REPO}/releases/{rel_id}/assets?name={urlquote(name)}",
        data=data, method="POST",
        headers={"Authorization": f"token {token()}", "Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    print(f"  uploaded {name}: {out.get('size')}B")

def urlquote(s):
    import urllib.parse
    return urllib.parse.quote(s)

def wait_for(run_id, timeout=10800):
    t0 = time.time()
    while True:
        time.sleep(15)
        r = api(f"/repos/{OWNER_REPO}/actions/runs/{run_id}")
        print(f"  [{int(time.time()-t0):5d}s] {r['status']} / {r.get('conclusion')}", flush=True)
        if r["status"] == "completed":
            if r["conclusion"] != "success":
                for j in api(f"/repos/{OWNER_REPO}/actions/runs/{run_id}/jobs")["jobs"]:
                    for st in j.get("steps", []):
                        print(f"    step {st['name']}: {st['conclusion']}")
            return r["conclusion"]
        if time.time() - t0 > timeout:
            return "timeout"

def dispatch_watch(inputs, workflow=None):
    wf = workflow or WORKFLOW
    before = {r["id"] for r in api(f"/repos/{OWNER_REPO}/actions/runs?per_page=20")["workflow_runs"]}
    api(f"/repos/{OWNER_REPO}/actions/workflows/{wf}/dispatches", "POST",
        {"ref": "main", "inputs": inputs})
    for _ in range(30):
        time.sleep(4)
        for r in api(f"/repos/{OWNER_REPO}/actions/runs?per_page=20")["workflow_runs"]:
            if r["id"] not in before:
                return r["id"], r["html_url"]
    return None, None

def cmd_analyze(a):
    """full pipeline: local download -> release assets -> free runner (whisper+vlm) -> fetch report"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stage = os.path.join(os.path.expanduser("~"), ".cache", "video-eye-stage", stamp)
    meta, audio, zpath = stage_locally(a.url, stage, a.frames, a.width, a.start, a.end)
    print(f"staged: '{meta['title']}'  {meta['duration'] and round(meta['duration'])}s  "
          f"{len(meta['frames'])} frames  audio={os.path.basename(audio) if audio else '—'}")

    tag = f"eye-{stamp}"
    rel_id = create_release(tag, f"{meta.get('title')} [{stamp}]")
    for p in (audio, zpath, os.path.join(stage, "meta.json")):
        if p and os.path.exists(p):
            upload_asset(rel_id, p)
    print("release:", f"https://github.com/{OWNER_REPO}/releases/tag/{tag}")

    rid, url = dispatch_watch({"asset_tag": tag, "whisper_model": a.whisper, "caption": a.caption})
    if not rid:
        sys.exit("dispatch failed")
    print("run:", url)
    if not a.wait:
        return
    conc = wait_for(rid)
    print("conclusion:", conc)

    # pull the report
    class NS: run_id = "latest"
    cmd_fetch(NS())
    # frames already local: link them into ~/runs/<dir> for the agent to read
    got = json.loads(raw("results/latest.json"))
    dst = os.path.join(RUNS_DIR, got["dir"])
    os.makedirs(dst, exist_ok=True)
    link = os.path.join(dst, "frames")
    if not os.path.exists(link):
        try:
            os.symlink(os.path.join(stage, "frames"), link)
        except OSError:
            shutil.copytree(os.path.join(stage, "frames"), link)
    print("frames ready:", link)

def cmd_prune(a):
    rels = api(f"/repos/{OWNER_REPO}/releases?per_page=100")
    rels.sort(key=lambda r: r["created_at"], reverse=True)
    for r in rels[a.keep:]:
        api(f"/repos/{OWNER_REPO}/releases/{r['id']}", "DELETE")
        api(f"/repos/{OWNER_REPO}/git/refs/tags/{r['tag_name']}", "DELETE")
        print("pruned", r["tag_name"])
    print(f"kept {min(len(rels), a.keep)} releases")


# ------------------------------------------------------------------ deep (параллельный просмотр)
def cmd_deep(a):
    """Плотное покрытие: сценозависимые кадры + шарды Whisper/VLM на бесплатных раннерах."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"deep-{stamp}"
    stage_dir = a.stage_dir or os.path.join(os.path.expanduser("~"), ".cache", "video-eye-stage", run_dir)
    if a.stage_dir:
        print(f"шаг 1/4: стейджинг беру готовый: {stage_dir}")
        run_dir = f"big-{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        cmd = [sys.executable, os.path.join(HERE, "scripts", "stage.py"), "--url", a.url,
               "--out-dir", stage_dir, "--every", str(a.every), "--max-frames", str(a.max_frames),
               "--whisper-shards", str(a.whisper_shards), "--caption-shards", str(a.caption_shards),
               "--tile", str(a.tile),
               "--caption-max-frames", str(a.caption_max_frames)]
        print("шаг 1/4: подготовка в песочнице (скачивание, кадры, листы, аудио)")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit("stage.py упал")
    meta = json.load(open(os.path.join(stage_dir, "meta.json"), encoding="utf-8"))
    print(f"  {meta['title']} | {meta['duration'] and round(meta['duration'])} с | "
          f"кадров {meta['frame_count']} (сцен {meta['scene_changes']}) | "
          f"листов {len(meta['sheets'])} | частей аудио {len(meta['audio_parts'])}")

    tag = f"eye-{run_dir}"   # тег релиза = eye-<run_dir>, без расхождений
    print("шаг 2/4: загрузка ассетов в релиз", tag)
    rel_id = create_release(tag, f"{meta.get('title')} [{stamp}]")
    adir = os.path.join(stage_dir, "assets")
    uploaded = 0
    for f in sorted(os.listdir(adir)):
        if f.endswith((".zip", ".opus", ".m4a", ".wav")):
            upload_asset(rel_id, os.path.join(adir, f))
            uploaded += 1
    upload_asset(rel_id, os.path.join(stage_dir, "meta.json"))
    print(f"  загружено файлов: {uploaded + 1}")

    print("шаг 3/4: запуск шардов на раннерах GitHub")
    jobs = [{"mode": "whisper", "shard": i, "of": len(meta["audio_parts"]), "model": a.whisper}
            for i in range(len(meta["audio_parts"]))]
    jobs += [{"mode": "caption", "shard": j["shard"], "of": len(meta["caption_shards"]),
              "model": a.model} for j in meta["caption_shards"]]
    rid, url = dispatch_watch({"asset_tag": tag, "run_dir": run_dir,
                               "jobs_json": json.dumps(jobs), "whisper_model": a.whisper,
                               "caption_model": a.model}, workflow="watch2.yml")
    if not rid:
        sys.exit("не удалось запустить workflow")
    print(f"  job'ов: {len(jobs)} (whisper {len(meta['audio_parts'])}, VLM {len(meta['caption_shards'])})")
    print("  ", url)

    if not a.wait:
        print("запущено. Следить: python3 gh_runner.py status")
        return
    print("шаг 4/4: жду завершения (это время шарда, а не сумма)")
    conc = wait_for(rid)
    print("итог:", conc)
    class NS: run_id = "latest"
    cmd_fetch(NS())
    dst = os.path.join(RUNS_DIR, run_dir)
    os.makedirs(dst, exist_ok=True)
    link = os.path.join(dst, "sheets")
    if not os.path.exists(link):
        try:
            os.symlink(os.path.join(stage_dir, "sheets"), link)
        except OSError:
            shutil.copytree(os.path.join(stage_dir, "sheets"), link)
    print("листы кадров (смотреть как изображения):", link)

# ------------------------------------------------------------------ status
def cmd_status(a):
    r = api(f"/repos/{OWNER_REPO}")
    print(f"repo: {r['full_name']}  private={r['private']}  size={r['size']}KB")
    lim = api(f"/repos/{OWNER_REPO}/actions/permissions")
    print("actions enabled:", lim.get("enabled"))
    for run in api(f"/repos/{OWNER_REPO}/actions/runs?per_page=5")["workflow_runs"]:
        print(f"  run {run['id']}: {run['status']}/{run.get('conclusion')}  {run['html_url']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push")
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--start", default="")
    r.add_argument("--end", default="")
    r.add_argument("--frames", default=60)
    r.add_argument("--width", default=768)
    r.add_argument("--whisper", default="small")
    r.add_argument("--caption", default="off")
    r.add_argument("--wait", action="store_true", help="block until the run finishes")
    f = sub.add_parser("fetch")
    f.add_argument("run_id", nargs="?", default="latest")
    sub.add_parser("status")
    an = sub.add_parser("analyze", help="local download + free GitHub runner for the heavy models")
    an.add_argument("--url", required=True)
    an.add_argument("--start", default="")
    an.add_argument("--end", default="")
    an.add_argument("--frames", default=60)
    an.add_argument("--width", default=768)
    an.add_argument("--whisper", default="small")
    an.add_argument("--caption", default="off")
    an.add_argument("--wait", action="store_true")
    dp = sub.add_parser("deep", help="плотный параллельный просмотр: кадры сцен + шарды Whisper/VLM")
    dp.add_argument("--url", required=True)
    dp.add_argument("--every", type=float, default=4.0, help="секунд между кадрами сетки")
    dp.add_argument("--max-frames", type=int, default=700)
    dp.add_argument("--whisper-shards", type=int, default=4)
    dp.add_argument("--caption-shards", type=int, default=8)
    dp.add_argument("--whisper", default="large-v3")
    dp.add_argument("--model", default="qwen3-vl:2b")
    dp.add_argument("--tile", type=int, default=320)
    dp.add_argument("--caption-max-frames", type=int, default=240)
    dp.add_argument("--stage-dir", help="готовый стейджинг (пропустить скачивание)")
    dp.add_argument("--wait", action="store_true")
    pr = sub.add_parser("prune")
    pr.add_argument("--keep", type=int, default=5)
    a = ap.parse_args()
    {"push": cmd_push, "run": cmd_run, "fetch": cmd_fetch, "status": cmd_status,
     "analyze": cmd_analyze, "deep": cmd_deep, "prune": cmd_prune}[a.cmd](a)


if __name__ == "__main__":
    main()
