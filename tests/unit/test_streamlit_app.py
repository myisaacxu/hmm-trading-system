import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from src.visualization.streamlit_app import StreamlitApp


class TestStreamlitApp:
    """Streamlit应用测试类"""

    @pytest.fixture
    def sample_stock_data(self):
        """创建示例股票数据"""
        dates = pd.date_range("2020-01-01", periods=100)
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100)) + 100
        volumes = np.random.randint(1000000, 10000000, size=100)

        return pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

    @pytest.fixture
    def sample_feature_data(self):
        """创建示例特征数据"""
        dates = pd.date_range("2020-01-01", periods=100)
        np.random.seed(42)

        return pd.DataFrame(
            {
                "VOL": np.random.randn(100) * 0.01,
                "SPREAD": np.random.randn(100) + 1.0,
                "EBS": np.random.randn(100) * 0.5,
                "BUFFETT": np.random.randn(100) + 1.5,
            },
            index=dates,
        )

    @pytest.fixture
    def sample_strategy_results(self):
        """创建示例策略结果"""
        dates = pd.date_range("2020-01-01", periods=100)

        return {
            "cagr": 0.15,
            "sharpe": 1.8,
            "max_drawdown": 0.12,
            "win_rate": 0.65,
            "returns": pd.Series(np.random.randn(100) * 0.01, index=dates),
        }

    @pytest.fixture
    def app_instance(self):
        """创建StreamlitApp实例"""
        with patch("streamlit.set_page_config"):
            return StreamlitApp()

    def test_initialization(self, app_instance):
        """测试应用初始化"""
        assert app_instance is not None
        assert hasattr(app_instance, "chart_gen")
        assert hasattr(app_instance, "logger")

    def test_display_data_summary(self, app_instance, sample_stock_data):
        """测试数据摘要显示"""
        # 使用MagicMock模拟st.subheader和st.metric
        with patch("streamlit.subheader"), patch(
            "streamlit.columns",
            return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()],
        ), patch("streamlit.metric"), patch("streamlit.plotly_chart"):

            app_instance.display_data_summary(sample_stock_data, "光大银行")

    def test_display_feature_analysis(self, app_instance, sample_feature_data):
        """测试特征分析显示"""
        # 使用MagicMock模拟st.subheader和st.metric
        with patch("streamlit.subheader"), patch(
            "streamlit.columns",
            return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()],
        ), patch("streamlit.metric"), patch("streamlit.plotly_chart"):

            app_instance.display_feature_analysis(sample_feature_data)

    def test_display_strategy_performance(self, app_instance, sample_strategy_results):
        """测试策略表现显示"""
        # 使用MagicMock模拟st.subheader和st.metric
        with patch("streamlit.subheader"), patch(
            "streamlit.columns",
            return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()],
        ), patch("streamlit.metric"), patch("streamlit.plotly_chart"):

            app_instance.display_strategy_performance(sample_strategy_results)

    def test_display_strategy_performance_empty(self, app_instance):
        """测试空策略结果显示"""
        # 使用MagicMock模拟st.subheader和st.warning
        with patch("streamlit.subheader"), patch("streamlit.warning"):

            app_instance.display_strategy_performance({})

    def test_display_error_message(self, app_instance):
        """测试错误信息显示"""
        # 使用MagicMock模拟st.error和st.info
        with patch("streamlit.error"), patch("streamlit.info"):

            app_instance.display_error_message("测试错误")

    def test_create_sidebar(self, app_instance):
        """测试侧边栏创建"""
        # 使用MagicMock模拟streamlit的侧边栏组件
        with patch("streamlit.sidebar.title"), patch(
            "streamlit.sidebar.selectbox", return_value="光大银行(601818)"
        ), patch(
            "streamlit.sidebar.date_input",
            return_value=(datetime.now() - timedelta(days=365), datetime.now()),
        ), patch(
            "streamlit.sidebar.subheader"
        ), patch(
            "streamlit.sidebar.slider", return_value=3
        ), patch(
            "streamlit.sidebar.checkbox", return_value=True
        ), patch(
            "streamlit.sidebar.button", return_value=True
        ):

            params = app_instance.create_sidebar()

            # 验证返回的参数
            assert isinstance(params, dict)
            assert "stock_code" in params
            assert "stock_name" in params
            assert "start_date" in params
            assert "end_date" in params
            assert "n_states" in params
            assert "use_smoothing" in params
            assert "analyze_clicked" in params

    def test_display_market_regime(self, app_instance, sample_stock_data):
        """测试市场状态显示"""
        # 创建示例状态数据
        dates = sample_stock_data.index
        regime_series = pd.Series(np.random.randint(0, 3, size=len(dates)), index=dates)

        model_info = {"converged": True, "n_iter": 50, "final_log_likelihood": -100.5}

        # 使用MagicMock模拟st.subheader和st.metric
        with patch("streamlit.subheader"), patch(
            "streamlit.columns", return_value=[MagicMock(), MagicMock(), MagicMock()]
        ), patch("streamlit.metric"), patch("streamlit.plotly_chart"):

            app_instance.display_market_regime(
                sample_stock_data, regime_series, model_info
            )

    def test_run(self, app_instance):
        """测试应用运行"""
        # 使用MagicMock模拟整个应用流程
        with patch.object(
            app_instance,
            "create_sidebar",
            return_value={
                "stock_code": "sh.601818",
                "stock_name": "光大银行(601818)",
                "start_date": datetime.now() - timedelta(days=365),
                "end_date": datetime.now(),
                "n_states": 3,
                "use_smoothing": True,
                "analyze_clicked": False,
            },
        ), patch.object(app_instance, "display_header"), patch("streamlit.info"), patch(
            "streamlit.expander"
        ):

            app_instance.run()

    def test_validate_dataframe_integration(self, app_instance, sample_stock_data):
        """测试数据验证功能集成"""
        from src.utils.helpers import validate_dataframe

        # 测试有效的数据
        is_valid, message = validate_dataframe(
            sample_stock_data, ["open", "high", "low", "close", "volume"]
        )
        assert is_valid
        assert "数据验证通过" in message

        # 测试缺少列的数据
        missing_columns_data = sample_stock_data.drop("volume", axis=1)
        is_valid, message = validate_dataframe(
            missing_columns_data, ["open", "high", "low", "close", "volume"]
        )
        assert not is_valid
        assert "缺少列" in message

        # 测试空数据
        empty_data = pd.DataFrame()
        is_valid, message = validate_dataframe(empty_data)
        assert not is_valid
        assert "DataFrame为空" in message

    def test_format_functions_integration(self):
        """测试格式化函数集成"""
        from src.utils.helpers import (
            format_percentage,
            format_currency,
            calculate_time_period,
        )

        # 测试百分比格式化
        assert format_percentage(0.1234) == "12.34%"
        assert format_percentage(0.05) == "5.00%"

        # 测试货币格式化
        assert format_currency(1234567) == "¥1.23M"
        assert format_currency(1234) == "¥1.23K"
        assert format_currency(1234567890) == "¥1.23B"

        # 测试时间期间计算
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2021, 1, 1)
        assert "1.0年" in calculate_time_period(start_date, end_date)

        end_date = datetime(2020, 2, 1)
        assert "1.0个月" in calculate_time_period(start_date, end_date)

        # 测试少于30天的情况
        end_date = datetime(2020, 1, 10)
        assert "9天" in calculate_time_period(start_date, end_date)

    def test_safe_divide_integration(self):
        """测试安全除法函数集成"""
        from src.utils.helpers import safe_divide

        # 测试正常除法
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(5, 2) == 2.5

        # 测试除零情况
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=1.0) == 1.0

        # 测试NaN和None情况
        assert safe_divide(10, None) == 0.0
        assert safe_divide(10, np.nan) == 0.0
        assert safe_divide(10, np.inf) == 0.0
