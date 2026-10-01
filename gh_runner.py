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
import argparse, io, json, os, shutil, subprocess, sys, time, urllib.request, zipfile

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
    url = f"https://raw.githubusercontent.com/{OWNER_REPO}/main/{path}"
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
    for name in ("report.md", "transcript.md", "info.json", "sheet.jpg"):
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
    a = ap.parse_args()
    {"push": cmd_push, "run": cmd_run, "fetch": cmd_fetch, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
