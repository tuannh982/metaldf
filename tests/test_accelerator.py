import sys

import pandas as pd

from metaldf._accelerator import install, uninstall


def _clear_pandas_cache() -> None:
    """Remove pandas top-level module from sys.modules cache.

    We only remove the top-level module so that re-importing pandas
    reuses cached submodules. This avoids class-mismatch issues where
    pd.Index._data_cls references an old ExtensionArray class while
    StringArray inherits from a new one.
    """
    sys.modules.pop("pandas", None)


def test_install_adds_meta_path_finder():
    install()
    try:
        assert any(
            hasattr(finder, "_metaldf_marker") for finder in sys.meta_path
        )
    finally:
        uninstall()


def test_uninstall_removes_meta_path_finder():
    install()
    uninstall()
    assert not any(
        hasattr(finder, "_metaldf_marker") for finder in sys.meta_path
    )


def test_import_pandas_returns_proxy_after_install():
    install()
    try:
        _clear_pandas_cache()
        import pandas as pd2

        # After install, pd.DataFrame should be the proxy type
        from metaldf._wrappers import ProxyDataFrame
        assert pd2.DataFrame is ProxyDataFrame
    finally:
        uninstall()
        _clear_pandas_cache()


def test_dataframe_created_via_proxy_is_proxy():
    install()
    try:
        _clear_pandas_cache()
        import pandas as pd2
        df = pd2.DataFrame({"x": [1, 2, 3]})
        from metaldf._wrappers import ProxyDataFrame
        assert type(df) is ProxyDataFrame
        assert isinstance(df, pd.DataFrame)
    finally:
        uninstall()
        _clear_pandas_cache()


def test_series_created_via_proxy_is_proxy():
    install()
    try:
        _clear_pandas_cache()
        import pandas as pd2
        s = pd2.Series([1, 2, 3])
        from metaldf._wrappers import ProxySeries
        assert type(s) is ProxySeries
        assert isinstance(s, pd.Series)
    finally:
        uninstall()
        _clear_pandas_cache()


def test_proxy_df_arithmetic_after_install():
    install()
    try:
        _clear_pandas_cache()
        import pandas as pd2
        df = pd2.DataFrame({"x": [1, 2, 3]})
        result = df + 1
        assert list(result["x"]) == [2, 3, 4]
    finally:
        uninstall()
        _clear_pandas_cache()
