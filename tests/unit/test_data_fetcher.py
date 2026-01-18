"""
数据获取模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path


class TestDataFetcher:
    """数据获取器测试类"""

    def test_cebbank_data_initialization(self):
        """测试光大银行数据获取器初始化"""
        # 这个测试会失败，因为我们还没有实现DataFetcher类
        # 这是测试驱动开发的第一步：编写失败的测试
        from src.data.data_fetcher import DataFetcher

        fetcher = DataFetcher(symbol="sh.601818")
        assert fetcher.symbol == "sh.601818"
        assert fetcher.data_source == "baostock"

    def test_get_stock_data_valid_params(self):
        """测试获取股票数据的有效参数"""
        from src.data.data_fetcher import DataFetcher

        fetcher = DataFetcher(symbol="sh.601818")

        start_date = "2020-01-01"
        end_date = "2023-12-31"

        with patch("baostock.login") as mock_login, patch(
            "baostock.query_history_k_data_plus"
        ) as mock_query:

            # 模拟登录成功
            mock_login.return_value.error_code = "0"

            # 模拟查询结果 - 修复Mock对象的行为
            mock_result = Mock()
            mock_result.error_code = "0"
            mock_result.fields = ["date", "open", "high", "low", "close", "volume"]

            # 模拟next()方法的调用
            # 第一次调用返回True，第二次返回False
            mock_result.next = Mock(side_effect=[True, False])

            mock_result.get_row_data.return_value = [
                "2023-01-01",
                "10.0",
                "10.5",
                "9.5",
                "10.2",
                "1000000",
            ]
            mock_query.return_value = mock_result

            data = fetcher.get_stock_data(start_date, end_date)

            assert isinstance(data, pd.DataFrame)
            assert "close" in data.columns
            assert len(data) > 0

    def test_get_stock_data_invalid_dates(self):
        """测试无效日期参数"""
        from src.data.data_fetcher import DataFetcher

        fetcher = DataFetcher(symbol="sh.601818")

        with pytest.raises(ValueError):
            fetcher.get_stock_data("2023-12-31", "2020-01-01")  # 结束日期早于开始日期

    def test_get_macro_data_ebs(self):
        """测试获取股债利差数据"""
        from src.data.data_fetcher import DataFetcher

        fetcher = DataFetcher()

        with patch("akshare.stock_ebs_lg") as mock_ebs:
            # 模拟股债利差数据
            mock_ebs.return_value = pd.DataFrame(
                {"日期": ["2023-01-01", "2023-01-02"], "股债利差": [3.5, 3.6]}
            )

            ebs_data = fetcher.get_macro_data("ebs")

            assert isinstance(ebs_data, pd.Series)
            assert "ebs_indicator" == ebs_data.name
            assert len(ebs_data) == 2

    def test_get_macro_data_buffett(self):
        """测试获取巴菲特指数数据"""
        from src.data.data_fetcher import DataFetcher

        fetcher = DataFetcher()

        with patch("akshare.macro_china_gdp") as mock_gdp, patch(
            "akshare.macro_china_stock_market_cap"
        ) as mock_market_cap:

            # 模拟GDP数据
            mock_gdp.return_value = pd.DataFrame(
                {"季度": ["2023Q1", "2023Q2"], "value": [1250000, 1251000]}
            )

            # 模拟市值数据
            mock_market_cap.return_value = pd.DataFrame(
                {
                    "月份": ["2023年1月份", "2023年2月份"],
                    "市价总值-上海": [500000, 510000],
                    "市价总值-深圳": [500000, 520000],
                }
            )

            buffett_data = fetcher.get_macro_data("buffett")

            assert isinstance(buffett_data, pd.Series)
            assert "buffett_index" == buffett_data.name
            assert len(buffett_data) > 0

    def test_data_alignment(self):
        """测试数据对齐功能"""
        from src.data.data_fetcher import DataFetcher

        fetcher = DataFetcher(symbol="sh.601818")

        # 创建不同时间索引的数据
        stock_dates = pd.date_range("2023-01-01", "2023-01-10", freq="D")
        macro_dates = pd.date_range("2023-01-05", "2023-01-15", freq="D")

        stock_data = pd.DataFrame({"close": range(10)}, index=stock_dates)

        macro_data = pd.DataFrame({"ebs_indicator": range(11)}, index=macro_dates)

        aligned_data = fetcher.align_data([stock_data, macro_data])

        assert len(aligned_data) == 6  # 交集日期数
        assert not aligned_data.isnull().any().any()

    def test_cache_directory_creation(self):
        """测试缓存目录创建功能"""
        from src.data.data_fetcher import DataFetcher

        with tempfile.TemporaryDirectory() as temp_dir:
            # 修改配置，使用临时目录作为缓存目录
            from src.config.config import HMMConfig

            config = HMMConfig()
            config.cache_dir = temp_dir

            fetcher = DataFetcher(config=config, symbol="sh.601818")

            # 检查缓存目录是否创建
            cache_dir = Path(temp_dir)
            assert cache_dir.exists()

    def test_save_data_to_csv(self):
        """测试数据保存为CSV功能"""
        from src.data.data_fetcher import DataFetcher

        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试数据
            test_data = pd.DataFrame(
                {
                    "close": [10, 20, 30, 40, 50],
                    "volume": [1000, 2000, 3000, 4000, 5000],
                },
                index=pd.date_range("2023-01-01", periods=5),
            )

            # 创建DataFetcher实例
            from src.config.config import HMMConfig

            config = HMMConfig()
            config.cache_dir = temp_dir
            fetcher = DataFetcher(config=config, symbol="sh.601818")

            # 调用保存方法
            file_path = Path(temp_dir) / "test_data.csv"
            fetcher._save_data_to_csv(test_data, file_path)

            # 检查文件是否存在
            assert file_path.exists()

            # 读取文件并验证内容
            saved_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            pd.testing.assert_frame_equal(test_data, saved_data, check_freq=False)

    def test_load_data_from_cache(self):
        """测试从缓存加载数据功能"""
        from src.data.data_fetcher import DataFetcher

        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试数据并保存到CSV
            test_data = pd.DataFrame(
                {
                    "close": [10, 20, 30, 40, 50],
                    "volume": [1000, 2000, 3000, 4000, 5000],
                },
                index=pd.date_range("2023-01-01", periods=5),
            )

            file_path = Path(temp_dir) / "test_data.csv"
            test_data.to_csv(file_path)

            # 创建DataFetcher实例
            from src.config.config import HMMConfig

            config = HMMConfig()
            config.cache_dir = temp_dir
            fetcher = DataFetcher(config=config, symbol="sh.601818")

            # 调用加载方法
            loaded_data = fetcher._load_data_from_cache(file_path)

            # 验证加载的数据与原始数据一致
            pd.testing.assert_frame_equal(test_data, loaded_data, check_freq=False)

    def test_is_cache_valid(self):
        """测试缓存有效性检查"""
        from src.data.data_fetcher import DataFetcher

        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试文件
            file_path = Path(temp_dir) / "test_data.csv"
            file_path.touch()

            # 创建DataFetcher实例
            from src.config.config import HMMConfig

            config = HMMConfig()
            config.cache_dir = temp_dir
            fetcher = DataFetcher(config=config, symbol="sh.601818")

            # 测试有效缓存（刚创建的文件）
            assert fetcher._is_cache_valid(file_path, max_age_days=7) == True

            # 测试无效缓存（设置非常小的过期时间）
            assert fetcher._is_cache_valid(file_path, max_age_days=0) == False

    def test_stock_data_cache_flow(self):
        """测试股票数据完整缓存流程"""
        from src.data.data_fetcher import DataFetcher

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "baostock.login"
        ) as mock_login, patch("baostock.query_history_k_data_plus") as mock_query:

            # 配置临时缓存目录
            from src.config.config import HMMConfig

            config = HMMConfig()
            config.cache_dir = temp_dir

            fetcher = DataFetcher(config=config, symbol="sh.601818")

            start_date = "2023-01-01"
            end_date = "2023-01-10"

            # 模拟登录成功
            mock_login.return_value.error_code = "0"

            # 模拟查询结果
            mock_result = Mock()
            mock_result.error_code = "0"
            mock_result.fields = ["date", "open", "high", "low", "close", "volume"]
            mock_result.next = Mock(side_effect=[True, True, False])
            mock_result.get_row_data.side_effect = [
                ["2023-01-01", "10.0", "10.5", "9.5", "10.2", "1000000"],
                ["2023-01-02", "10.2", "10.8", "9.8", "10.5", "1500000"],
            ]
            mock_query.return_value = mock_result

            # 第一次调用：从网络获取数据
            data1 = fetcher.get_stock_data(start_date, end_date)

            # 检查缓存文件是否创建
            cache_file = (
                Path(temp_dir)
                / f"stock/sh.601818/{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
            )
            assert cache_file.exists()

            # 第二次调用：从缓存获取数据
            data2 = fetcher.get_stock_data(start_date, end_date)

            # 验证两次获取的数据一致
            pd.testing.assert_frame_equal(data1, data2)

            # 验证mock_query只被调用一次
            assert mock_query.call_count == 1

    def test_macro_data_cache_flow(self):
        """测试宏观数据完整缓存流程"""
        from src.data.data_fetcher import DataFetcher

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "akshare.stock_ebs_lg"
        ) as mock_ebs:

            # 配置临时缓存目录
            from src.config.config import HMMConfig

            config = HMMConfig()
            config.cache_dir = temp_dir

            fetcher = DataFetcher(config=config)

            # 模拟股债利差数据
            mock_ebs.return_value = pd.DataFrame(
                {"日期": ["2023-01-01", "2023-01-02"], "股债利差": [3.5, 3.6]}
            )

            # 第一次调用：从网络获取数据
            data1 = fetcher.get_macro_data("ebs")

            # 检查缓存文件是否创建
            today = pd.Timestamp.now().strftime("%Y%m%d")
            cache_file = Path(temp_dir) / f"macro/ebs/{today}.csv"
            assert cache_file.exists()

            # 第二次调用：从缓存获取数据
            data2 = fetcher.get_macro_data("ebs")

            # 验证两次获取的数据一致
            pd.testing.assert_series_equal(data1, data2)

            # 验证mock_ebs只被调用一次
            assert mock_ebs.call_count == 1

    def test_raw_ebs_data_cache(self):
        """测试原始EBS数据缓存功能"""
        from src.data.data_fetcher import DataFetcher

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "akshare.stock_ebs_lg"
        ) as mock_ebs:

            # 配置临时缓存目录
            from src.config.config import HMMConfig

            config = HMMConfig()
            config.cache_dir = temp_dir

            fetcher = DataFetcher(config=config)

            # 模拟股债利差数据
            test_data = pd.DataFrame(
                {"日期": ["2023-01-01", "2023-01-02"], "股债利差": [3.5, 3.6]}
            )
            mock_ebs.return_value = test_data

            # 第一次调用：从网络获取原始数据
            data1 = fetcher._get_ebs_data()

            # 检查原始数据缓存文件是否创建
            today = pd.Timestamp.now().strftime("%Y%m%d")
            raw_cache_file = Path(temp_dir) / f"raw/ebs/{today}.csv"
            assert raw_cache_file.exists()

            # 读取原始缓存数据并验证
            raw_cache_data = pd.read_csv(raw_cache_file)
            pd.testing.assert_frame_equal(test_data, raw_cache_data)

            # 第二次调用：从原始缓存获取数据
            data2 = fetcher._get_ebs_data()

            # 验证两次获取的数据一致
            pd.testing.assert_series_equal(data1, data2)

            # 验证mock_ebs只被调用一次
            assert mock_ebs.call_count == 1

    def test_raw_buffett_data_cache(self):
        """测试原始巴菲特指数数据缓存功能"""
        from src.data.data_fetcher import DataFetcher

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "akshare.stock_buffett_index_lg"
        ) as mock_buffett:

            # 配置临时缓存目录
            from src.config.config import HMMConfig

            config = HMMConfig()
            config.cache_dir = temp_dir

            fetcher = DataFetcher(config=config)

            # 模拟巴菲特指数数据
            test_data = pd.DataFrame(
                {
                    "日期": ["2023-01-01", "2023-01-02"],
                    "总市值": [1000000, 1100000],
                    "GDP": [500000, 550000],
                }
            )
            mock_buffett.return_value = test_data

            # 第一次调用：从网络获取原始数据
            data1 = fetcher._get_buffett_index()

            # 检查原始数据缓存文件是否创建
            today = pd.Timestamp.now().strftime("%Y%m%d")
            raw_cache_file = Path(temp_dir) / f"raw/buffett/{today}.csv"
            assert raw_cache_file.exists()

            # 读取原始缓存数据并验证
            raw_cache_data = pd.read_csv(raw_cache_file)
            pd.testing.assert_frame_equal(test_data, raw_cache_data)

            # 第二次调用：从原始缓存获取数据
            data2 = fetcher._get_buffett_index()

            # 验证两次获取的数据一致
            pd.testing.assert_series_equal(data1, data2)

            # 验证mock_buffett只被调用一次
            assert mock_buffett.call_count == 1


class TestDataProcessor:
    """数据处理器测试类"""

    def test_data_cleaning(self):
        """测试数据清洗功能"""
        from src.data.data_processor import DataProcessor

        processor = DataProcessor()

        # 创建包含缺失值和异常值的数据
        data = pd.DataFrame(
            {
                "close": [10, 20, None, 40, 1000],  # 包含缺失值和异常值
                "volume": [1000, 2000, 3000, 4000, 5000],
            }
        )

        cleaned_data = processor.clean_data(data)

        assert cleaned_data.isnull().sum().sum() == 0
        assert (cleaned_data["close"] < 100).all()  # 异常值被处理

    def test_outlier_detection(self):
        """测试异常值检测"""
        from src.data.data_processor import DataProcessor

        processor = DataProcessor()

        # 创建包含异常值的数据
        data = pd.Series([1, 2, 3, 100, 4, 5])  # 100是异常值

        outliers = processor.detect_outliers(data)

        assert len(outliers) == 1
        assert outliers.iloc[0] == 100

    def test_data_normalization(self):
        """测试数据标准化"""
        from src.data.data_processor import DataProcessor

        processor = DataProcessor()

        data = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [100, 200, 300, 400, 500]}
        )

        normalized_data = processor.normalize_data(data)

        # 检查每列的均值为0，标准差为1
        for col in normalized_data.columns:
            assert abs(normalized_data[col].mean()) < 1e-10
            assert abs(normalized_data[col].std() - 1) < 1e-10

    def test_feature_engineering(self):
        """测试特征工程"""
        from src.data.data_processor import DataProcessor

        processor = DataProcessor()

        # 创建价格数据
        prices = pd.Series([100, 101, 102, 101, 100])

        features = processor.create_features(prices)

        expected_features = [
            "log_return",
            "volatility",
            "ma_short",
            "ma_long",
            "spread",
        ]

        for feature in expected_features:
            assert feature in features.columns
