import subprocess
import sys
import tempfile


def test_cli_runs_script():
    script = """
import pandas as pd
df = pd.DataFrame({"x": [1, 2, 3]})
print(df["x"].sum())
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        result = subprocess.run(
            [sys.executable, "-m", "metaldf", f.name],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "6" in result.stdout


def test_cli_passes_args_to_script():
    script = """
import sys
print(f"args: {sys.argv}")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        result = subprocess.run(
            [sys.executable, "-m", "metaldf", f.name, "--flag", "value"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--flag" in result.stdout
        assert "value" in result.stdout


def test_cli_no_script_shows_help():
    result = subprocess.run(
        [sys.executable, "-m", "metaldf"],
        capture_output=True,
        text=True,
    )
    # argparse exits with code 2 when required positional argument is missing
    assert result.returncode == 2
    assert "usage:" in result.stderr.lower()
