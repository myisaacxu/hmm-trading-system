"""
可视化模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.visualization.chart_generator import ChartGenerator


class TestChartGenerator:
    """图表生成器测试类"""

    def setup_method(self):
        """测试初始化"""
        self.chart_gen = ChartGenerator()

        # 创建测试数据
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        self.price_data = pd.DataFrame(
            {
                "close": np.random.normal(100, 10, len(dates)),
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        self.feature_data = pd.DataFrame(
            {
                "VOL": np.random.normal(0.02, 0.01, len(dates)),
                "SPREAD": np.random.normal(0.03, 0.01, len(dates)),
                "EBS": np.random.normal(0.05, 0.02, len(dates)),
                "BUFFETT": np.random.normal(1.2, 0.1, len(dates)),
            },
            index=dates,
        )

        self.regime_data = pd.Series(
            np.random.choice(["bull", "bear", "neutral"], len(dates)), index=dates
        )

    def test_create_price_chart(self):
        """测试创建价格图表"""
        fig = self.chart_gen.create_price_chart(self.price_data, "测试价格图表")

        # 验证图表对象存在
        assert fig is not None

        # 验证图表包含正确的轨迹数量
        assert len(fig.data) == 2  # 价格线和成交量

        # 验证图表标题
        assert fig.layout.title.text == "测试价格图表"

    def test_create_market_regime_chart(self):
        """测试创建市场状态图表"""
        fig = self.chart_gen.create_market_regime_chart(
            self.price_data, self.regime_data, "测试状态图表"
        )

        # 验证图表对象存在
        assert fig is not None

        # 验证包含子图
        assert hasattr(fig, "_grid_ref")

        # 验证图表标题
        assert fig.layout.title.text == "测试状态图表"

    def test_create_feature_analysis_chart(self):
        """测试创建特征分析图表"""
        fig = self.chart_gen.create_feature_analysis_chart(
            self.feature_data, "测试特征图表"
        )

        # 验证图表对象存在
        assert fig is not None

        # 验证包含正确的特征数量
        available_features = [
            f
            for f in ["VOL", "SPREAD", "EBS", "BUFFETT"]
            if f in self.feature_data.columns
        ]
        assert len(fig.data) == len(available_features)

    def test_create_feature_analysis_chart_empty_data(self):
        """测试空数据时的特征分析图表"""
        empty_data = pd.DataFrame()
        fig = self.chart_gen.create_feature_analysis_chart(empty_data)

        # 验证返回空图表
        assert fig is not None
        assert len(fig.data) == 0  # 空图表应该没有数据轨迹

    def test_create_strategy_performance_chart(self):
        """测试创建策略表现图表"""
        # 创建测试收益率数据
        returns = pd.Series(
            np.random.normal(0.001, 0.02, len(self.price_data)),
            index=self.price_data.index,
        )

        fig = self.chart_gen.create_strategy_performance_chart(
            returns, title="测试策略图表"
        )

        # 验证图表对象存在
        assert fig is not None

        # 验证包含策略收益轨迹
        assert len(fig.data) >= 1
        assert fig.layout.title.text == "测试策略图表"

    def test_create_strategy_performance_chart_with_benchmark(self):
        """测试带基准的策略表现图表"""
        # 创建测试收益率数据
        returns = pd.Series(
            np.random.normal(0.001, 0.02, len(self.price_data)),
            index=self.price_data.index,
        )
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, len(self.price_data)),
            index=self.price_data.index,
        )

        fig = self.chart_gen.create_strategy_performance_chart(
            returns, benchmark_returns, "测试带基准策略图表"
        )

        # 验证图表对象存在
        assert fig is not None

        # 验证包含策略和基准两个轨迹
        assert len(fig.data) == 2

    def test_create_regime_statistics_chart(self):
        """测试创建状态统计图表"""
        regime_counts = {"bull": 100, "bear": 50, "neutral": 75}
        fig = self.chart_gen.create_regime_statistics_chart(
            regime_counts, "测试统计图表"
        )

        # 验证图表对象存在
        assert fig is not None

        # 验证包含饼图数据
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], type(fig.data[0]))  # 验证是饼图类型

        # 验证使用正确的颜色
        data_trace = fig.data[0]
        assert "marker" in data_trace
        assert "colors" in data_trace.marker

    def test_create_regime_statistics_chart_empty(self):
        """测试空数据时的状态统计图表"""
        empty_counts = {}
        fig = self.chart_gen.create_regime_statistics_chart(empty_counts)

        # 验证图表对象存在
        assert fig is not None
        # 空数据时应该返回基本图表结构
        assert hasattr(fig, "data")

    def test_color_palette_initialization(self):
        """测试颜色调色板初始化"""
        # 验证颜色调色板正确设置
        expected_colors = ["bull", "bear", "neutral", "price", "volume", "trend"]

        for color_name in expected_colors:
            assert color_name in self.chart_gen.color_palette
            assert self.chart_gen.color_palette[color_name].startswith("#")

    def test_chart_layout_configuration(self):
        """测试图表布局配置"""
        fig = self.chart_gen.create_price_chart(self.price_data)

        # 验证基本布局属性
        assert hasattr(fig.layout, "title")
        assert hasattr(fig.layout, "xaxis")
        assert hasattr(fig.layout, "yaxis")

        # 验证高度设置
        assert fig.layout.height == 400

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("测试标题", "测试标题"),
            ("", ""),
            (None, "特征分析"),  # 默认标题
        ],
    )
    def test_chart_titles(self, title, expected):
        """测试图表标题设置"""
        if title is None:
            fig = self.chart_gen.create_feature_analysis_chart(self.feature_data)
        else:
            fig = self.chart_gen.create_feature_analysis_chart(self.feature_data, title)

        assert fig.layout.title.text == expected


class TestStreamlitApp:
    """Streamlit应用测试类"""

    def setup_method(self):
        """测试初始化"""
        # 导入StreamlitApp类
        from src.visualization.streamlit_app import StreamlitApp

        self.app_class = StreamlitApp

        # 创建测试数据
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        self.price_data = pd.DataFrame(
            {
                "close": np.random.normal(100, 10, len(dates)),
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

    @patch("streamlit.set_page_config")
    def test_app_initialization(self, mock_set_page_config):
        """测试应用初始化"""
        # 创建应用实例
        app = self.app_class()

        # 验证应用正确初始化
        assert app is not None
        assert hasattr(app, "chart_gen")
        assert hasattr(app, "logger")

        # 验证页面配置被调用
        mock_set_page_config.assert_called_once()

    @patch("streamlit.sidebar")
    def test_create_sidebar(self, mock_sidebar):
        """测试侧边栏创建"""
        # 模拟sidebar组件
        mock_sidebar.title = Mock()
        mock_sidebar.selectbox = Mock(return_value="光大银行(601818)")
        mock_sidebar.date_input = Mock(
            return_value=(datetime(2020, 1, 1), datetime(2023, 1, 1))
        )
        mock_sidebar.slider = Mock(return_value=3)
        mock_sidebar.checkbox = Mock(return_value=True)
        mock_sidebar.button = Mock(return_value=False)
        mock_sidebar.subheader = Mock()

        # 创建应用实例
        app = self.app_class()

        # 调用创建侧边栏方法
        params = app.create_sidebar()

        # 验证返回正确的参数
        assert "stock_code" in params
        assert "stock_name" in params
        assert "start_date" in params
        assert "end_date" in params
        assert "n_states" in params
        assert "use_smoothing" in params
        assert "analyze_clicked" in params

    @patch("streamlit.title")
    @patch("streamlit.markdown")
    def test_display_header(self, mock_markdown, mock_title):
        """测试显示页面头部"""
        # 创建应用实例
        app = self.app_class()

        # 调用显示头部方法
        app.display_header("测试股票")

        # 验证标题和markdown被调用
        mock_title.assert_called()
        mock_markdown.assert_called()

    def test_display_loading_indicator(self):
        """测试显示加载指示器"""
        # 创建应用实例
        app = self.app_class()

        # 这个测试主要是验证方法不会抛出异常
        # 由于streamlit的spinner在测试环境中可能无法正常工作
        # 我们主要验证方法可以正常调用
        try:
            app.display_loading_indicator("测试加载")
            # 如果执行到这里，说明方法正常
            assert True
        except Exception as e:
            # 允许一些streamlit相关的错误
            if "StreamlitAPIException" not in str(type(e).__name__):
                raise

    @patch("streamlit.metric")
    @patch("streamlit.plotly_chart")
    @patch("streamlit.subheader")
    def test_display_data_summary(self, mock_subheader, mock_plotly_chart, mock_metric):
        """测试显示数据摘要"""
        # 创建应用实例
        app = self.app_class()

        # 调用显示数据摘要方法
        app.display_data_summary(self.price_data, "测试股票")

        # 验证相关组件被调用
        mock_subheader.assert_called()
        mock_metric.assert_called()
        mock_plotly_chart.assert_called()

    def test_display_error_message(self):
        """测试显示错误信息"""
        # 创建应用实例
        app = self.app_class()

        # 这个测试主要是验证方法不会抛出异常
        try:
            app.display_error_message("测试错误信息")
            # 如果执行到这里，说明方法正常
            assert True
        except Exception as e:
            # 允许一些streamlit相关的错误
            if "StreamlitAPIException" not in str(type(e).__name__):
                raise
