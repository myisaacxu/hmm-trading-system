"""
测试配置和公共fixtures
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_stock_data():
    """生成模拟股票数据"""
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")

    # 生成模拟价格数据（随机游走）
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 10 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "date": dates,
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, len(dates)),
            "amount": prices * np.random.randint(1000000, 5000000, len(dates)),
        }
    )
    data.set_index("date", inplace=True)
    return data


@pytest.fixture
def sample_macro_data():
    """生成模拟宏观数据"""
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")

    data = pd.DataFrame(
        {
            "date": dates,
            "ebs_indicator": np.random.normal(3.0, 0.5, len(dates)),  # 股债利差
            "buffett_index": np.random.normal(80, 10, len(dates)),  # 巴菲特指数
        }
    )
    data.set_index("date", inplace=True)
    return data


@pytest.fixture
def sample_features():
    """生成模拟特征数据"""
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")

    data = pd.DataFrame(
        {
            "date": dates,
            "log_ret": np.random.normal(0, 0.02, len(dates)),
            "volatility": np.random.uniform(0.01, 0.05, len(dates)),
            "spread": np.random.uniform(-0.1, 0.1, len(dates)),
            "ebs": np.random.normal(3.0, 0.5, len(dates)),
            "buffett": np.random.normal(80, 10, len(dates)),
        }
    )
    data.set_index("date", inplace=True)
    return data


@pytest.fixture
def sample_regimes():
    """生成模拟市场状态"""
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")

    # 生成模拟状态序列（有粘性）
    np.random.seed(42)
    states = [0]
    for _ in range(1, len(dates)):
        if np.random.random() < 0.95:  # 95%概率保持当前状态
            states.append(states[-1])
        else:
            states.append(np.random.choice([0, 1, 2]))

    return pd.Series(states, index=dates)


@pytest.fixture
def hmm_config():
    """HMM模型配置"""
    return {
        "n_states": 3,
        "covariance_type": "diag",
        "n_iter": 100,
        "tol": 1e-4,
        "random_state": 42,
    }
