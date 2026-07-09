import subprocess
import sys
import tempfile


def test_full_script_produces_same_output():
    """Run a pandas script through metaldf and verify output matches plain python."""
    script = """
import pandas as pd

df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
df["z"] = df["x"] + df["y"]
print(df["z"].sum())
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()

        # Run with metaldf
        result_metaldf = subprocess.run(
            [sys.executable, "-m", "metaldf", f.name],
            capture_output=True,
            text=True,
        )

        # Run with plain python
        result_plain = subprocess.run(
            [sys.executable, f.name],
            capture_output=True,
            text=True,
        )

        assert result_metaldf.returncode == 0, result_metaldf.stderr
        assert result_plain.returncode == 0, result_plain.stderr
        assert result_metaldf.stdout == result_plain.stdout


def test_isinstance_check_works():
    """After install, isinstance(proxy, pd.DataFrame) must be True."""
    import sys

    from metaldf._accelerator import install, uninstall

    install()
    try:
        sys.modules.pop("pandas", None)
        import pandas as pd2

        df = pd2.DataFrame({"x": [1, 2, 3]})
        assert type(df).__name__ == "ProxyDataFrame"
        assert isinstance(df, pd2.DataFrame)
    finally:
        uninstall()
        sys.modules.pop("pandas", None)


def test_read_csv_through_cli():
    """Read CSV through metaldf CLI and verify results."""
    script = """
import tempfile
import pandas as pd

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as cf:
    cf.write("x,y\\n1,4\\n2,5\\n3,6\\n")
    cf.flush()
    df = pd.read_csv(cf.name)
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


def test_groupby_through_cli():
    """GroupBy aggregation through metaldf CLI."""
    script = """
import pandas as pd

df = pd.DataFrame({"key": ["a", "a", "b"], "val": [10, 20, 30]})
result = df.groupby("key")["val"].sum()
print(result["a"])
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
        assert "30" in result.stdout


def test_series_ops_through_cli():
    """Series operations through metaldf CLI."""
    script = """
import pandas as pd

s = pd.Series([1, 2, 3, 4, 5])
print(s.mean())
print(s.min())
print(s.max())
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
        assert "3.0" in result.stdout
        assert "1" in result.stdout
        assert "5" in result.stdout


def test_metaldf_script_no_crash():
    """Verify metaldf CLI doesn't crash on empty DataFrame."""
    script = """
import pandas as pd

df = pd.DataFrame()
print("empty_OK")
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
        assert "empty_OK" in result.stdout
