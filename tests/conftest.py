"""
Test-wide configuration.

``src/web_demo/db.py`` builds its engine from DATABASE_URL at import time, so
the override has to happen before any test module imports it. conftest.py is
imported first by pytest, which makes this the only reliable place for it.

A file-backed SQLite database is used rather than ``:memory:`` because the API
tests need several connections to see the same data.
"""

import os
import tempfile
from pathlib import Path

_TEST_DB = Path(tempfile.gettempdir()) / "azsl_test_auth.sqlite3"

# Start from a clean schema on every run.
try:
    _TEST_DB.unlink(missing_ok=True)
except OSError:  # pragma: no cover - a stale handle on Windows
    pass

os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{_TEST_DB.as_posix()}"
os.environ.setdefault("SESSION_SECRET", "test-secret-not-used-in-production")
