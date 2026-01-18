"""
数据获取器配置集成测试
测试重构后的DataFetcher如何使用配置模块
"""

from src.data.data_fetcher import DataFetcher
from src.config.config import HMMConfig
import pytest
import sys
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestDataFetcherWithConfig:
    """测试使用配置模块的DataFetcher"""

    def test_data_fetcher_with_config_object(self):
        """测试使用配置对象初始化DataFetcher"""
        # 当使用配置对象创建DataFetcher时
        config = HMMConfig(stock_symbol="sh.601398", start_date="2015-01-01")
        fetcher = DataFetcher(config=config)

        # 应该使用配置对象中的参数
        assert fetcher.config == config
        assert fetcher.symbol == "sh.601398"
        assert fetcher.data_source == "baostock"

    def test_data_fetcher_without_config(self):
        """测试不使用配置对象初始化DataFetcher（向后兼容）"""
        # 当不使用配置对象创建DataFetcher时
        fetcher = DataFetcher(symbol="sh.000001", data_source="test_source")

        # 应该使用直接参数并创建默认配置
        assert isinstance(fetcher.config, HMMConfig)
        assert fetcher.symbol == "sh.000001"
        assert fetcher.data_source == "test_source"

    def test_data_fetcher_default_values(self):
        """测试DataFetcher的默认值"""
        # 当使用默认配置创建DataFetcher时
        fetcher = DataFetcher()

        # 应该使用配置的默认值
        assert fetcher.symbol == "sh.601818"
        assert fetcher.data_source == "baostock"

    @patch("src.data.data_fetcher.bs")
    @patch("src.data.data_fetcher.DataFetcher._is_cache_valid")
    def test_get_stock_data_with_config_dates(self, mock_is_cache_valid, mock_bs):
        """测试使用配置中的日期获取股票数据"""
        # 当使用配置对象且不指定日期时
        config = HMMConfig(start_date="2018-01-01")
        fetcher = DataFetcher(config=config)

        # 模拟缓存无效，强制从网络获取
        mock_is_cache_valid.return_value = False

        # 模拟登录成功
        mock_login_result = Mock()
        mock_login_result.error_code = "0"
        mock_bs.login.return_value = mock_login_result

        # 模拟查询结果 - 创建更复杂的模拟对象
        mock_result = Mock()
        mock_result.error_code = "0"
        mock_result.fields = [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
            "pctChg",
        ]

        # 模拟next()方法的行为
        # 第一次调用返回True，第二次返回False
        mock_result.next = Mock(side_effect=[True, False])
        mock_result.get_row_data.return_value = [
            "2023-01-01",
            "sh.601818",
            "10.0",
            "10.5",
            "9.5",
            "10.2",
            "10.0",
            "1000000",
            "10000000",
            "1.0",
            "1.0",
        ]

        mock_bs.query_history_k_data_plus.return_value = mock_result

        # 调用方法，不指定日期
        data = fetcher.get_stock_data()

        # 应该使用配置中的开始日期
        mock_bs.query_history_k_data_plus.assert_called_once()
        call_args = mock_bs.query_history_k_data_plus.call_args[1]
        assert call_args["start_date"] == "2018-01-01"

        # 验证其他参数
        assert call_args["code"] == "sh.601818"
        assert "end_date" in call_args
        assert call_args["frequency"] == "d"
        assert call_args["adjustflag"] == "3"

    @patch("src.data.data_fetcher.bs")
    @patch("src.data.data_fetcher.DataFetcher._is_cache_valid")
    def test_get_stock_data_with_custom_dates(self, mock_is_cache_valid, mock_bs):
        """测试使用自定义日期获取股票数据"""
        # 当指定自定义日期时
        config = HMMConfig(start_date="2018-01-01")
        fetcher = DataFetcher(config=config)

        # 模拟缓存无效，强制从网络获取
        mock_is_cache_valid.return_value = False

        # 模拟登录成功
        mock_login_result = Mock()
        mock_login_result.error_code = "0"
        mock_bs.login.return_value = mock_login_result

        # 模拟查询结果 - 创建更复杂的模拟对象
        mock_result = Mock()
        mock_result.error_code = "0"
        mock_result.fields = [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
            "pctChg",
        ]

        # 模拟next()方法的行为
        # 第一次调用返回True，第二次返回False
        mock_result.next = Mock(side_effect=[True, False])
        mock_result.get_row_data.return_value = [
            "2023-01-01",
            "sh.601818",
            "10.0",
            "10.5",
            "9.5",
            "10.2",
            "10.0",
            "1000000",
            "10000000",
            "1.0",
            "1.0",
        ]

        mock_bs.query_history_k_data_plus.return_value = mock_result

        # 调用方法，指定自定义日期
        data = fetcher.get_stock_data(start_date="2020-01-01", end_date="2023-12-31")

        # 应该使用自定义日期而不是配置日期
        mock_bs.query_history_k_data_plus.assert_called_once()
        call_args = mock_bs.query_history_k_data_plus.call_args[1]
        assert call_args["start_date"] == "2020-01-01"
        assert call_args["end_date"] == "2023-12-31"

        # 验证其他参数
        assert call_args["code"] == "sh.601818"
        assert call_args["frequency"] == "d"
        assert call_args["adjustflag"] == "3"

    @patch("src.data.data_fetcher.bs")
    @patch("src.data.data_fetcher.DataFetcher._is_cache_valid")
    def test_get_combined_data_with_config(self, mock_is_cache_valid, mock_bs):
        """测试使用配置获取组合数据"""
        # 当使用配置获取组合数据时
        config = HMMConfig(start_date="2019-01-01")
        fetcher = DataFetcher(config=config)

        # 模拟缓存无效，强制从网络获取
        mock_is_cache_valid.return_value = False

        # 模拟登录成功和查询结果
        mock_login_result = Mock()
        mock_login_result.error_code = "0"
        mock_bs.login.return_value = mock_login_result

        mock_result = Mock()
        mock_result.error_code = "0"
        mock_result.fields = ["date", "close"]

        # 正确模拟next()方法，使其返回True然后False
        # 第一次调用返回True，第二次返回False
        mock_result.next = Mock(side_effect=[True, False])
        mock_result.get_row_data.return_value = ["2023-01-01", "10.0"]
        mock_bs.query_history_k_data_plus.return_value = mock_result

        # 模拟宏观数据方法
        with patch.object(fetcher, "get_macro_data") as mock_macro, patch.object(
            fetcher, "align_data"
        ) as mock_align:

            # 设置模拟返回值 - 使用实际的Series对象
            mock_series = pd.Series(
                [3.0],
                index=pd.date_range("2023-01-01", periods=1),
                name="ebs_indicator",
            )
            mock_series2 = pd.Series(
                [100.0],
                index=pd.date_range("2023-01-01", periods=1),
                name="buffett_index",
            )
            mock_macro.side_effect = [mock_series, mock_series2]

            # 模拟对齐方法返回有效数据
            mock_aligned_data = pd.DataFrame(
                {"close": [10.0], "ebs_indicator": [3.0], "buffett_index": [100.0]},
                index=pd.date_range("2023-01-01", periods=1),
            )
            mock_align.return_value = mock_aligned_data

            # 调用组合数据方法
            result = fetcher.get_combined_data()

            # 应该调用股票数据方法并使用配置中的开始日期
            mock_bs.query_history_k_data_plus.assert_called_once()
            call_args = mock_bs.query_history_k_data_plus.call_args[1]
            assert call_args["start_date"] == "2019-01-01"

    def test_config_validation_in_data_fetcher(self):
        """测试DataFetcher中的配置验证"""
        # 当使用无效配置时
        config = HMMConfig(stock_symbol="invalid_symbol")
        fetcher = DataFetcher(config=config)

        # 配置验证应该失败
        assert fetcher.config.validate() is False
        assert len(fetcher.config.validation_errors) > 0

    def test_config_persistence(self):
        """测试配置对象在DataFetcher中的持久性"""
        # 当修改配置对象时
        config = HMMConfig()
        fetcher = DataFetcher(config=config)

        # 修改配置对象
        config.stock_symbol = "sh.601398"

        # DataFetcher中的配置应该同步更新
        assert fetcher.config.stock_symbol == "sh.601398"
        assert fetcher.symbol == "sh.601398"


class TestDataFetcherBackwardCompatibility:
    """测试DataFetcher的向后兼容性"""

    def test_backward_compatibility_symbol(self):
        """测试股票代码的向后兼容性"""
        # 当使用旧的初始化方式时
        fetcher = DataFetcher(symbol="sh.601398")

        # 应该仍然工作，并创建默认配置
        assert fetcher.symbol == "sh.601398"
        assert isinstance(fetcher.config, HMMConfig)

    def test_backward_compatibility_data_source(self):
        """测试数据源的向后兼容性"""
        # 当使用旧的初始化方式时
        fetcher = DataFetcher(data_source="test_source")

        # 应该仍然工作
        assert fetcher.data_source == "test_source"
        assert isinstance(fetcher.config, HMMConfig)

    def test_mixed_initialization(self):
        """测试混合初始化方式"""
        # 当同时提供配置对象和直接参数时
        config = HMMConfig(stock_symbol="sh.601398")
        fetcher = DataFetcher(config=config, symbol="sh.000001")

        # 应该优先使用配置对象
        assert fetcher.symbol == "sh.601398"  # 来自配置对象
        assert fetcher.config.stock_symbol == "sh.601398"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
