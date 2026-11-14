import os
from pathlib import Path
import subprocess
import sys


def test_package_metadata_with_non_utf8_default_encoding():
    """Metadata must preserve Unicode even when the locale is not UTF-8."""
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        LC_ALL="C",
        PYTHONCOERCECLOCALE="0",
        PYTHONUTF8="0",
        PYTHONIOENCODING="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "setup.py", "--long-description"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip("\r\n") == root.joinpath("README.md").read_text(
        encoding="utf-8"
    ).rstrip("\r\n")
