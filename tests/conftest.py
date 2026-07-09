"""pytest configuration and fixtures for metaldf."""

import sys

import pandas as pd
import pytest

from metaldf._accelerator import install, uninstall
from metaldf._wrappers import ProxyDataFrame


@pytest.fixture
def real_pandas_df():
    """Returns a real pandas DataFrame for comparison."""
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


@pytest.fixture
def proxy_df(real_pandas_df):
    """Returns a ProxyDataFrame wrapping the real DataFrame."""
    return ProxyDataFrame(_pandas_obj=real_pandas_df)


@pytest.fixture
def installed_metaldf():
    """Installs metaldf import interception for the duration of the test."""
    install()
    # Remove cached pandas module so next import goes through our finder
    sys.modules.pop("pandas", None)
    yield
    uninstall()
    sys.modules.pop("pandas", None)
