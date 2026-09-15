"""
Pre-deploy checks. Run this before pushing a release.

    py scripts/preflight_deploy.py              # check everything
    py scripts/preflight_deploy.py --target render
    py scripts/preflight_deploy.py --target vercel

It answers the questions that only bite after a deploy:

  * will the build context still contain the model files, or did an ignore rule
    quietly drop them?
  * does the serverless function have every module it imports?
  * are the environment variables set, and consistent between hosts?
  * do both entrypoints import and register the routes they should?

Exit code is 0 when everything a target needs is in place, 1 otherwise.
Warnings (optional things, like a TURN relay) never fail the run.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
if os.name == "nt" and not os.getenv("WT_SESSION"):
    # Old consoles render the escapes literally; plain text is better than noise.
    GREEN = YELLOW = RED = DIM = RESET = ""

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}    {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  {YELLOW}WARN{RESET}  {msg}")


def bad(msg: str) -> None:
    failures.append(msg)
    print(f"  {RED}FAIL{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n{title}")


# --------------------------------------------------------------------------- #
# Ignore-file handling
# --------------------------------------------------------------------------- #
def load_ignore(name: str) -> list[str]:
    path = PROJECT_ROOT / name
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def is_ignored(rel_path: str, patterns: list[str]) -> str | None:
    """Return the pattern that excludes ``rel_path``, or None.

    Implements the subset of the syntax these files actually use: directory
    prefixes ("docs/"), globs ("*.pt") and path globs ("outputs/reports/").
    """
    rel = rel_path.replace("\\", "/")
    parts = rel.split("/")
    for pattern in patterns:
        pat = pattern.rstrip("/")
        if pattern.endswith("/"):
            # Directory rule: matches the dir itself or anything beneath it.
            if rel == pat or rel.startswith(pat + "/"):
                return pattern
            # A bare dir name like "tests/" also matches at any depth.
            if "/" not in pat and pat in parts[:-1]:
                return pattern
            continue
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(parts[-1], pat):
            return pattern
        if rel.startswith(pat + "/"):
            return pattern
        if pat.startswith("**/"):
            tail = pat[3:]
            if any(fnmatch.fnmatch(p, tail) for p in parts):
                return pattern
    return None


# Files the container image cannot start without.
CONTAINER_REQUIRED = [
    "Dockerfile",
    "requirements.txt",
    "requirements-recognition.txt",
    "src/web_demo/backend.py",
    "src/web_demo/webapp.py",
    "src/web_demo/deps.py",
    "src/web_demo/social.py",
    "src/web_demo/db.py",
    "src/web_demo/frontend/index.html",
    "src/web_demo/frontend/friends.html",
    "src/web_demo/frontend/landing.html",
    "src/web_demo/frontend/register.html",
    "src/web_demo/frontend/models/azsl_hierarchical_model.json",
    "src/models/hand_landmarker.task",
    "outputs/vocabulary_24_cap50/checkpoints/gru_24_cap50_best.pt",
    "outputs/vocabulary_24_cap50/metadata/feature_normalization_stats_24.json",
]

# Files the Vercel function cannot start without (no ML stack).
VERCEL_REQUIRED = [
    "api/index.py",
    "requirements.txt",
    "src/web_demo/__init__.py",
    "src/web_demo/webapp.py",
    "src/web_demo/deps.py",
    "src/web_demo/social.py",
    "src/web_demo/db.py",
    "src/web_demo/frontend/index.html",
    "src/web_demo/frontend/friends.html",
    "src/web_demo/frontend/landing.html",
    "src/web_demo/frontend/register.html",
]


def check_build_context(label: str, ignore_file: str, required: list[str]) -> None:
    section(f"{label} build context ({ignore_file})")
    patterns = load_ignore(ignore_file)
    if not patterns:
        warn(f"{ignore_file} not found or empty — the upload may be much larger than needed")
        return

    missing_on_disk, excluded = [], []
    for rel in required:
        if not (PROJECT_ROOT / rel).exists():
            missing_on_disk.append(rel)
            continue
        hit = is_ignored(rel, patterns)
        if hit:
            excluded.append((rel, hit))

    for rel in missing_on_disk:
        bad(f"missing from the repository: {rel}")
    for rel, pattern in excluded:
        bad(f"{ignore_file} excludes a required file: {rel}  (rule: {pattern!r})")

    if not missing_on_disk and not excluded:
        ok(f"all {len(required)} required files present and not excluded")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def check_env(target: str) -> None:
    section(f"environment ({target})")
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass

    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        warn("DATABASE_URL not set here — must be set on the host")
    elif db_url.startswith("postgresql://"):
        bad("DATABASE_URL must use the psycopg v3 driver: postgresql+psycopg://…")
    elif "localhost" in db_url or "127.0.0.1" in db_url:
        warn("DATABASE_URL points at localhost — unreachable from a cloud host")
    else:
        ok("DATABASE_URL looks like a remote psycopg URL")

    secret = os.getenv("SESSION_SECRET", "").strip()
    if not secret or secret == "change-me":
        warn("SESSION_SECRET not set here — it MUST be set on the host, and match across hosts")
    elif len(secret) < 32:
        warn(f"SESSION_SECRET is short ({len(secret)} chars); 64 hex chars recommended")
    else:
        ok("SESSION_SECRET present and long enough")

    if os.getenv("SESSION_COOKIE_SECURE", "0") != "1":
        warn("SESSION_COOKIE_SECURE is not 1 — set it to 1 on any HTTPS deployment")
    else:
        ok("SESSION_COOKIE_SECURE=1")

    turn = [os.getenv(k, "").strip() for k in ("TURN_URL", "TURN_USERNAME", "TURN_CREDENTIAL")]
    if all(turn):
        ok("TURN relay configured — calls work on restrictive networks too")
    elif any(turn):
        bad("TURN is half-configured; all three of TURN_URL / TURN_USERNAME / "
            "TURN_CREDENTIAL are required, or calls fail at ICE time")
    else:
        warn("no TURN relay — calls use STUN only and will fail for users behind "
             "symmetric NAT (mobile data, corporate networks). This is optional.")

    if target == "vercel" and not os.getenv("RECOGNITION_WS_URL", "").strip():
        warn("RECOGNITION_WS_URL not set — the Vercel deploy will have no camera "
             "recognition, no live chat and no calling (friends/messaging still work over REST)")


# --------------------------------------------------------------------------- #
# Imports and routes
# --------------------------------------------------------------------------- #
def check_web_layer() -> None:
    section("web layer")
    os.environ.setdefault("SESSION_SECRET", "preflight-placeholder-secret-value-0123456789")
    prior_secure = os.environ.pop("SESSION_COOKIE_SECURE", None)
    try:
        from fastapi import FastAPI

        from src.web_demo.webapp import build_web_layer

        app = FastAPI()
        build_web_layer(app, serves_ws=True)
        paths = {getattr(r, "path", "") for r in app.routes}

        expected = [
            "/", "/login", "/register", "/app", "/friends", "/health",
            "/api/register", "/api/login", "/api/logout", "/api/me",
            "/api/ws-token", "/api/tts",
            "/api/friends", "/api/friends/requests", "/api/friends/request",
            "/api/friends/respond", "/api/friends/remove",
            "/api/users/search", "/api/messages", "/api/rtc-config",
            "/ws/social",
        ]
        missing = [p for p in expected if p not in paths]
        if missing:
            bad(f"routes not registered: {', '.join(missing)}")
        else:
            ok(f"all {len(expected)} expected routes registered (serves_ws=True)")

        if "torch" in sys.modules:
            bad("importing the web layer pulled in torch — the Vercel function would exceed its size limit")
        else:
            ok("web layer imports without the ML stack (Vercel-safe)")
    except Exception as exc:
        bad(f"web layer failed to import: {type(exc).__name__}: {exc}")
    finally:
        if prior_secure is not None:
            os.environ["SESSION_COOKIE_SECURE"] = prior_secure


def _declared_packages(path: Path) -> str:
    """The requirement lines only, lower-cased.

    Comments matter here: api/requirements.txt explains that it *excludes*
    torch and mediapipe, so a naive substring search finds them and reports the
    opposite of the truth.
    """
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            lines.append(line.lower())
    return "\n".join(lines)


def check_requirements() -> None:
    section("dependencies")
    web = _declared_packages(PROJECT_ROOT / "requirements.txt")
    api_req = PROJECT_ROOT / "api" / "requirements.txt"

    # Everything webapp.py / social.py / db.py / deps.py import at module level.
    needed = {
        "fastapi": "fastapi",
        "sqlalchemy": "sqlalchemy",
        "psycopg": "psycopg",
        "argon2-cffi": "argon2",
        "itsdangerous": "itsdangerous",
        "python-dotenv": "dotenv",
        "email-validator": "email_validator",
        "edge-tts": "edge_tts",
    }
    missing = [pkg for pkg in needed if pkg not in web]
    if missing:
        bad(f"requirements.txt is missing: {', '.join(missing)}")
    else:
        ok(f"requirements.txt covers all {len(needed)} web-layer imports")

    if api_req.is_file():
        api_text = _declared_packages(api_req)
        api_missing = [pkg for pkg in needed if pkg not in api_text]
        if api_missing:
            bad(f"api/requirements.txt is missing: {', '.join(api_missing)}")
        else:
            ok("api/requirements.txt covers the serverless function's imports")
        for heavy in ("torch", "mediapipe", "opencv"):
            if heavy in api_text:
                bad(f"api/requirements.txt contains {heavy} — it will blow Vercel's 250 MB limit")


def check_single_instance_config() -> None:
    section("single-instance constraint (in-memory presence and calls)")
    script = PROJECT_ROOT / "scripts" / "deploy-cloudrun.ps1"
    if script.is_file():
        text = script.read_text(encoding="utf-8")
        if '"--max-instances", "1"' in text:
            ok("Cloud Run script pins --max-instances 1")
        else:
            bad("Cloud Run script does not pin --max-instances 1 — users on "
                "different instances could not see or call each other")

    fly = PROJECT_ROOT / "fly.toml"
    if fly.is_file() and "min_machines_running = 0" in fly.read_text(encoding="utf-8"):
        warn("fly.toml scales to zero — fine, but keep the app at ONE machine "
             "(`fly scale count 1`) while presence is in-memory")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["all", "render", "fly", "cloudrun", "vercel"],
        default="all",
        help="which deployment to check (default: all)",
    )
    args = parser.parse_args()
    target = args.target

    print(f"{DIM}Preflight checks for: {target}{RESET}")

    if target in ("all", "render", "fly", "cloudrun"):
        check_build_context("Container (Render / Fly)", ".dockerignore", CONTAINER_REQUIRED)
    if target in ("all", "cloudrun"):
        check_build_context("Cloud Run", ".gcloudignore", CONTAINER_REQUIRED)
    if target in ("all", "vercel"):
        check_build_context("Vercel", ".vercelignore", VERCEL_REQUIRED)

    check_requirements()
    check_web_layer()
    check_single_instance_config()
    check_env("vercel" if target == "vercel" else "host")

    print("\n" + "=" * 62)
    if failures:
        print(f"{RED}{len(failures)} blocking issue(s){RESET}, {len(warnings)} warning(s)")
        for f in failures:
            print(f"  - {f}")
        print("=" * 62)
        return 1
    print(f"{GREEN}No blocking issues{RESET}, {len(warnings)} warning(s).")
    if warnings:
        print(f"{DIM}Warnings are things to confirm on the host, not build failures.{RESET}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
