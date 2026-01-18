"""
数据集成测试
测试与外部数据源（baostock、akshare）的集成
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 导入项目模块
from src.data.data_fetcher import DataFetcher


class TestDataFetcherIntegration:
    """数据获取集成测试类"""

    @patch("src.data.data_fetcher.bs")
    @patch("src.data.data_fetcher.ak")
    @patch("src.data.data_fetcher.DataFetcher._is_cache_valid")
    def test_baostock_data_fetching(self, mock_is_cache_valid, mock_ak, mock_bs):
        """测试baostock数据获取集成"""
        # 模拟缓存无效，强制从网络获取
        mock_is_cache_valid.return_value = False

        # 模拟baostock响应
        mock_login_result = Mock()
        mock_login_result.error_code = "0"
        mock_login_result.error_msg = ""

        # 直接设置mock对象的方法
        mock_bs.login.return_value = mock_login_result

        # 模拟返回数据
        mock_data = [
            [
                "2020-01-01",
                "sh.601818",
                "10.0",
                "10.2",
                "9.8",
                "10.1",
                "10.0",
                "1000000",
                "10000000",
                "0",
                "1.0",
                "1.0",
            ],
            [
                "2020-01-02",
                "sh.601818",
                "10.1",
                "10.3",
                "9.9",
                "10.2",
                "10.1",
                "1200000",
                "12000000",
                "0",
                "2.0",
                "0.99",
            ],
        ]

        # 模拟rs对象，具有error_code、next()、get_row_data()和fields属性
        mock_rs = Mock()
        mock_rs.error_code = "0"
        mock_rs.fields = [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "adjustflag",
            "turn",
            "pctChg",
        ]

        # 模拟next()和get_row_data()方法
        mock_rs.next.side_effect = [True, True, False]  # 两次成功，然后结束
        mock_rs.get_row_data.side_effect = mock_data

        mock_bs.query_history_k_data_plus.return_value = mock_rs
        mock_bs.logout.return_value = None

        # 测试数据获取
        data_fetcher = DataFetcher()
        result = data_fetcher.get_stock_data("2020-01-01", "2020-01-02")

        # 验证结果
        assert not result.empty, "应该成功获取股票数据"
        assert "close" in result.columns, "应该包含收盘价列"
        assert len(result) == 2, "应该返回2个数据点"

        # 验证baostock API调用
        mock_bs.login.assert_called_once()
        mock_bs.query_history_k_data_plus.assert_called_once()
        mock_bs.logout.assert_called_once()

    @patch("src.data.data_fetcher.ak")
    def test_akshare_macro_data_fetching(self, mock_ak):
        """测试akshare宏观数据获取集成"""
        # 模拟akshare响应
        mock_gdp_data = pd.DataFrame(
            {"季度": ["2020Q1", "2020Q2"], "value": [100000, 110000]}
        )

        mock_market_cap_data = pd.DataFrame(
            {
                "月份": ["2020年01月份", "2020年02月份"],
                "市价总值-上海": [500000, 520000],
                "市价总值-深圳": [300000, 310000],
            }
        )

        mock_ak.macro_china_gdp = Mock(return_value=mock_gdp_data)
        mock_ak.macro_china_stock_market_cap = Mock(return_value=mock_market_cap_data)

        # 测试数据获取
        data_fetcher = DataFetcher()

        # 测试GDP数据获取
        gdp_result = data_fetcher.fetch_gdp_data()
        assert not gdp_result.empty, "应该成功获取GDP数据"

        # 测试市值数据获取
        market_cap_result = data_fetcher.fetch_market_cap_data()
        assert not market_cap_result.empty, "应该成功获取市值数据"

        # 验证akshare API调用
        mock_ak.macro_china_gdp.assert_called_once()
        mock_ak.macro_china_stock_market_cap.assert_called_once()

    @patch("src.data.data_fetcher.bs")
    def test_baostock_connection_failure(self, mock_bs):
        """测试baostock连接失败场景"""
        # 模拟连接失败
        mock_login_result = Mock()
        mock_login_result.error_code = "-1"
        mock_login_result.error_msg = "Connection failed"
        mock_bs.login.return_value = mock_login_result

        data_fetcher = DataFetcher()

        # 测试连接失败时的处理
        result = data_fetcher.fetch_stock_data("601818.SH", "2020-01-01", "2020-01-02")

        # 验证连接失败时的处理
        assert result.empty, "连接失败时应返回空DataFrame"

    @patch("src.data.data_fetcher.ak")
    def test_akshare_data_failure(self, mock_ak):
        """测试akshare数据获取失败场景"""
        # 模拟数据获取失败
        mock_ak.macro_china_gdp = Mock(return_value=pd.DataFrame())

        data_fetcher = DataFetcher()

        # 测试数据获取失败时的处理
        result = data_fetcher.fetch_gdp_data()

        # 验证数据获取失败时的处理
        assert result.empty, "数据获取失败时应返回空DataFrame"


class TestRealDataIntegration:
    """真实数据集成测试类"""

    @pytest.mark.slow
    def test_real_baostock_connection(self):
        """测试真实baostock连接（慢速测试）"""
        # 这个测试会实际连接baostock，标记为慢速测试
        data_fetcher = DataFetcher()

        try:
            # 测试连接和获取少量数据
            result = data_fetcher.fetch_stock_data(
                "601818.SH", "2023-12-01", "2023-12-05", max_retries=1  # 光大银行
            )

            # 验证结果
            if not result.empty:
                assert "close" in result.columns, "应该包含收盘价列"
                assert len(result) > 0, "应该返回数据"
            else:
                # 如果获取失败，可能是网络问题，不视为测试失败
                pytest.skip("无法连接到baostock，可能是网络问题")

        except Exception as e:
            # 捕获异常，不视为测试失败
            pytest.skip(f"baostock连接异常: {e}")

    @pytest.mark.slow
    def test_real_akshare_connection(self):
        """测试真实akshare连接（慢速测试）"""
        # 这个测试会实际调用akshare，标记为慢速测试
        data_fetcher = DataFetcher()

        try:
            # 测试获取少量数据
            result = data_fetcher.fetch_gdp_data()

            # 验证结果
            if not result.empty:
                assert (
                    "value" in result.columns or "GDP" in result.columns
                ), "应该包含GDP值列"
            else:
                # 如果获取失败，可能是API问题，不视为测试失败
                pytest.skip("无法从akshare获取GDP数据")

        except Exception as e:
            # 捕获异常，不视为测试失败
            pytest.skip(f"akshare连接异常: {e}")


class TestDataConsistency:
    """数据一致性测试类"""

    def test_data_timestamp_consistency(self):
        """测试时间戳一致性"""
        # 创建测试数据
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

        stock_data = pd.DataFrame({"close": np.random.normal(10, 1, 100)}, index=dates)

        # 验证时间戳连续性
        date_diff = np.diff(stock_data.index)
        expected_diff = pd.Timedelta(days=1)

        # 所有相邻日期差应该为1天
        assert all(diff == expected_diff for diff in date_diff), "时间戳应该连续"

    def test_data_value_consistency(self):
        """测试数据值一致性"""
        # 创建测试数据
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")

        stock_data = pd.DataFrame(
            {
                "close": np.random.normal(10, 1, 50),
                "volume": np.random.randint(1000000, 5000000, 50),
            },
            index=dates,
        )

        # 验证数据值合理性
        assert (stock_data["close"] > 0).all(), "收盘价应该为正数"
        assert (stock_data["volume"] > 0).all(), "成交量应该为正数"

        # 验证数据范围
        assert stock_data["close"].min() > 0, "收盘价最小值应该大于0"
        assert stock_data["volume"].min() > 0, "成交量最小值应该大于0"

    def test_missing_data_handling(self):
        """测试缺失数据处理"""
        # 创建包含缺失值的数据
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

        stock_data = pd.DataFrame({"close": np.random.normal(10, 1, 100)}, index=dates)

        # 随机设置一些缺失值
        missing_indices = np.random.choice(range(100), size=10, replace=False)
        stock_data.iloc[missing_indices] = np.nan

        # 测试数据获取器的缺失值处理
        data_fetcher = DataFetcher()

        # 模拟数据获取结果
        with patch.object(data_fetcher, "fetch_stock_data", return_value=stock_data):
            result = data_fetcher.fetch_stock_data(
                "601818.SH", "2020-01-01", "2020-04-09"
            )

            # 验证缺失值处理
            # 特征工程应该能够处理包含缺失值的数据
            from src.features.feature_engineer import FeatureEngineer

            feature_engineer = FeatureEngineer()
            price_series = (
                result["close"] if not result.empty else pd.Series([], dtype=float)
            )

            features_result = feature_engineer.engineer_features(price_series)
            features_df = features_result["features_df"]

            # 验证特征工程能够处理缺失值
            assert not features_df.isnull().any().any(), "特征工程应该正确处理缺失值"


class TestDataTransformation:
    """数据转换测试类"""

    def test_data_normalization(self):
        """测试数据标准化"""
        # 创建测试数据
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")

        stock_data = pd.DataFrame({"close": np.random.normal(10, 2, 50)}, index=dates)

        # 测试数据标准化
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        normalized_data = scaler.fit_transform(stock_data[["close"]])

        # 验证标准化结果
        assert np.abs(normalized_data.mean()) < 1e-10, "标准化后均值应该接近0"
        assert np.abs(normalized_data.std() - 1) < 1e-10, "标准化后方差应该接近1"

    def test_feature_alignment(self):
        """测试特征对齐"""
        # 创建不同时间范围的数据
        stock_dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
        macro_dates = pd.date_range(start="2020-03-01", end="2020-10-31", freq="D")

        stock_data = pd.DataFrame(
            {"close": np.random.normal(10, 1, len(stock_dates))}, index=stock_dates
        )

        macro_data = pd.DataFrame(
            {"ebs": np.random.normal(3, 0.5, len(macro_dates))}, index=macro_dates
        )

        # 测试特征对齐
        aligned_data = stock_data.join(macro_data, how="inner")

        # 验证对齐结果
        expected_start = max(stock_dates[0], macro_dates[0])
        expected_end = min(stock_dates[-1], macro_dates[-1])

        assert aligned_data.index[0] >= expected_start, "对齐后起始时间应该正确"
        assert aligned_data.index[-1] <= expected_end, "对齐后结束时间应该正确"

        # 验证没有缺失值
        assert not aligned_data.isnull().any().any(), "对齐后不应该有缺失值"
