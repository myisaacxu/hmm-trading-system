import pytest
import pandas as pd
import numpy as np
from src.features.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """技术指标计算类的测试用例"""

    @pytest.fixture
    def technical_indicator(self):
        """创建TechnicalIndicators实例"""
        return TechnicalIndicators()

    @pytest.fixture
    def sample_prices(self):
        """创建示例价格数据"""
        dates = pd.date_range("2023-01-01", periods=100)
        prices = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates)
        return prices

    @pytest.fixture
    def sample_ohlcv(self):
        """创建示例OHLCV数据"""
        dates = pd.date_range("2023-01-01", periods=100)
        close = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates)
        high = close * 1.02
        low = close * 0.98
        open = close.shift(1).fillna(close.iloc[0])
        volume = pd.Series(np.random.randint(1000, 10000, size=100), index=dates)

        return pd.DataFrame(
            {"open": open, "high": high, "low": low, "close": close, "volume": volume}
        )

    def test_initialization(self, technical_indicator):
        """测试初始化参数设置"""
        assert technical_indicator.volatility_window == 30
        assert technical_indicator.ma_short_window == 20
        assert technical_indicator.ma_long_window == 100
        assert technical_indicator.rsi_window == 14
        assert technical_indicator.macd_fast == 12
        assert technical_indicator.macd_slow == 26
        assert technical_indicator.macd_signal == 9

    def test_calculate_log_returns(self, technical_indicator, sample_prices):
        """测试对数收益率计算"""
        log_returns = technical_indicator.calculate_log_returns(sample_prices)

        # 验证返回类型和长度
        assert isinstance(log_returns, pd.Series)
        assert len(log_returns) == len(sample_prices)

        # 验证第一个值为0（因为是差分后填充）
        assert log_returns.iloc[0] == 0.0

        # 验证计算结果的正确性
        expected_returns = np.log(sample_prices).diff().fillna(0.0)
        pd.testing.assert_series_equal(log_returns, expected_returns)

    def test_calculate_volatility(self, technical_indicator, sample_prices):
        """测试波动率计算"""
        returns = technical_indicator.calculate_log_returns(sample_prices)
        volatility = technical_indicator.calculate_volatility(returns)

        # 验证返回类型和长度
        assert isinstance(volatility, pd.Series)
        assert len(volatility) == len(returns)

        # 验证波动率为非负值
        assert (volatility >= 0).all()

        # 测试使用自定义窗口
        custom_volatility = technical_indicator.calculate_volatility(returns, window=10)
        assert isinstance(custom_volatility, pd.Series)
        assert len(custom_volatility) == len(returns)

    def test_calculate_moving_averages(self, technical_indicator, sample_prices):
        """测试移动平均线计算"""
        ma_features = technical_indicator.calculate_moving_averages(sample_prices)

        # 验证返回类型和包含的键
        assert isinstance(ma_features, dict)
        assert set(ma_features.keys()) == {"ma_short", "ma_long", "spread"}

        # 验证每个值都是Series类型
        for key, value in ma_features.items():
            assert isinstance(value, pd.Series)
            assert len(value) == len(sample_prices)

    def test_calculate_rsi(self, technical_indicator, sample_prices):
        """测试RSI计算"""
        rsi = technical_indicator.calculate_rsi(sample_prices)

        # 验证返回类型和长度
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_prices)

        # 验证RSI值在0-100之间
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

        # 测试使用自定义窗口
        custom_rsi = technical_indicator.calculate_rsi(sample_prices, window=10)
        assert isinstance(custom_rsi, pd.Series)
        assert len(custom_rsi) == len(sample_prices)

    def test_calculate_macd(self, technical_indicator, sample_prices):
        """测试MACD计算"""
        macd_features = technical_indicator.calculate_macd(sample_prices)

        # 验证返回类型和包含的键
        assert isinstance(macd_features, dict)
        assert set(macd_features.keys()) == {"macd", "signal", "histogram"}

        # 验证每个值都是Series类型
        for key, value in macd_features.items():
            assert isinstance(value, pd.Series)
            assert len(value) == len(sample_prices)

    def test_calculate_bollinger_bands(self, technical_indicator, sample_prices):
        """测试布林带计算"""
        bb_features = technical_indicator.calculate_bollinger_bands(sample_prices)

        # 验证返回类型和包含的键
        assert isinstance(bb_features, dict)
        expected_keys = {"upper", "middle", "lower", "bandwidth", "position"}
        assert set(bb_features.keys()) == expected_keys

        # 验证每个值都是Series类型
        for key, value in bb_features.items():
            assert isinstance(value, pd.Series)
            assert len(value) == len(sample_prices)

        # 验证上轨大于中轨，中轨大于下轨（跳过NaN值）
        valid_data = ~bb_features["upper"].isnull()
        assert (
            bb_features["upper"][valid_data] >= bb_features["middle"][valid_data]
        ).all()
        assert (
            bb_features["middle"][valid_data] >= bb_features["lower"][valid_data]
        ).all()

    def test_calculate_atr(self, technical_indicator, sample_ohlcv):
        """测试ATR计算"""
        atr = technical_indicator.calculate_atr(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )

        # 验证返回类型和长度
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_ohlcv)

        # 验证ATR为非负值
        assert (atr >= 0).all()

    def test_calculate_volume_indicators(self, technical_indicator, sample_ohlcv):
        """测试成交量指标计算"""
        volume_features = technical_indicator.calculate_volume_indicators(
            sample_ohlcv["volume"], sample_ohlcv["close"]
        )

        # 验证返回类型和包含的键
        assert isinstance(volume_features, dict)
        expected_keys = {"volume_ma", "volume_ratio", "money_flow", "money_flow_ma"}
        assert set(volume_features.keys()) == expected_keys

        # 验证每个值都是Series类型
        for key, value in volume_features.items():
            assert isinstance(value, pd.Series)
            assert len(value) == len(sample_ohlcv)

    def test_create_comprehensive_features(self, technical_indicator, sample_ohlcv):
        """测试综合特征创建"""
        features = technical_indicator.create_comprehensive_features(sample_ohlcv)

        # 验证返回类型和索引
        assert isinstance(features, pd.DataFrame)
        assert features.index.equals(sample_ohlcv.index)

        # 验证包含预期的特征列
        expected_columns = [
            "log_return",
            "volatility",
            "ma_ma_short",
            "ma_ma_long",
            "ma_spread",
            "rsi",
            "macd_macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_bandwidth",
            "bb_position",
            "volume_volume_ma",
            "volume_volume_ratio",
            "volume_money_flow",
            "volume_money_flow_ma",
        ]

        for col in expected_columns:
            assert col in features.columns

        # 验证没有缺失值
        assert features.isnull().sum().sum() == 0

    def test_create_comprehensive_features_minimal_data(self, technical_indicator):
        """测试使用最少数据创建特征"""
        # 创建只有收盘价的DataFrame
        dates = pd.date_range("2023-01-01", periods=50)
        minimal_data = pd.DataFrame(
            {"close": pd.Series(np.cumsum(np.random.randn(50)) + 100, index=dates)}
        )

        features = technical_indicator.create_comprehensive_features(minimal_data)

        # 验证返回类型和索引
        assert isinstance(features, pd.DataFrame)
        assert features.index.equals(minimal_data.index)

        # 验证没有缺失值
        assert features.isnull().sum().sum() == 0

    def test_initialization_with_custom_params(self):
        """测试使用自定义参数初始化"""
        custom_indicator = TechnicalIndicators(
            volatility_window=15,
            ma_short_window=10,
            ma_long_window=50,
            rsi_window=7,
            macd_fast=6,
            macd_slow=13,
            macd_signal=4,
        )

        # 验证自定义参数设置成功
        assert custom_indicator.volatility_window == 15
        assert custom_indicator.ma_short_window == 10
        assert custom_indicator.ma_long_window == 50
        assert custom_indicator.rsi_window == 7
        assert custom_indicator.macd_fast == 6
        assert custom_indicator.macd_slow == 13
        assert custom_indicator.macd_signal == 4

    def test_calculate_rsi_boundary_cases(self, technical_indicator):
        """测试RSI计算的边界情况"""
        # 创建持续上涨的价格序列
        rising_prices = pd.Series(
            [100, 101, 102, 103, 104, 105], index=pd.date_range("2023-01-01", periods=6)
        )
        rising_rsi = technical_indicator.calculate_rsi(rising_prices)

        # 验证RSI趋近于100但不超过
        assert rising_rsi.iloc[-1] <= 100
        assert rising_rsi.iloc[-1] > 90

        # 创建持续下跌的价格序列
        falling_prices = pd.Series(
            [100, 99, 98, 97, 96, 95], index=pd.date_range("2023-01-01", periods=6)
        )
        falling_rsi = technical_indicator.calculate_rsi(falling_prices)

        # 验证RSI趋近于0但不低于
        assert falling_rsi.iloc[-1] >= 0
        assert falling_rsi.iloc[-1] < 10

    def test_calculate_bollinger_bands_custom_params(
        self, technical_indicator, sample_prices
    ):
        """测试使用自定义参数计算布林带"""
        bb_features = technical_indicator.calculate_bollinger_bands(
            sample_prices, window=10, num_std=1.5
        )

        # 验证返回类型和包含的键
        assert isinstance(bb_features, dict)
        expected_keys = {"upper", "middle", "lower", "bandwidth", "position"}
        assert set(bb_features.keys()) == expected_keys

        # 验证每个值都是Series类型
        for key, value in bb_features.items():
            assert isinstance(value, pd.Series)
            assert len(value) == len(sample_prices)
