import os
from pathlib import Path
import subprocess
import sys


def test_package_metadata_with_non_utf8_default_encoding():
    """Source metadata must build even when Python's locale is not UTF-8."""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, LC_ALL="C", PYTHONCOERCECLOCALE="0", PYTHONUTF8="0")
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "setup.py", "--name"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "phayes"
