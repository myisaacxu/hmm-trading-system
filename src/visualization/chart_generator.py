"""
图表生成模块
负责生成各种可视化图表
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class ChartGenerator:
    """图表生成器类"""

    def __init__(self):
        self.color_palette = {
            "bull": "#00D100",  # 牛市绿色
            "bear": "#FF0000",  # 熊市红色
            "neutral": "#FFA500",  # 中性橙色
            "price": "#1f77b4",  # 价格蓝色
            "volume": "#ff7f0e",  # 成交量橙色
            "trend": "#2ca02c",  # 趋势绿色
        }

    def create_price_chart(
        self, df: pd.DataFrame, title: str = "股票价格走势"
    ) -> go.Figure:
        """
        创建价格走势图

        Args:
            df: 包含价格数据的DataFrame
            title: 图表标题

        Returns:
            go.Figure: Plotly图表对象
        """
        fig = go.Figure()

        # 添加收盘价线
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["close"],
                mode="lines",
                name="收盘价",
                line=dict(color=self.color_palette["price"], width=2),
            )
        )

        # 添加成交量（次级y轴）
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                name="成交量",
                marker=dict(color=self.color_palette["volume"], opacity=0.3),
                yaxis="y2",
            )
        )

        # 设置布局
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title="日期",
            yaxis_title="价格",
            yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=400,
        )

        return fig

    def create_market_regime_chart(
        self, df: pd.DataFrame, regime_data: pd.Series, title: str = "市场状态识别"
    ) -> go.Figure:
        """
        创建市场状态识别图表

        Args:
            df: 价格数据DataFrame
            regime_data: 状态序列
            title: 图表标题

        Returns:
            go.Figure: Plotly图表对象
        """
        # 创建子图
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("价格走势", "市场状态"),
            row_heights=[0.7, 0.3],
        )

        # 添加价格走势
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["close"],
                mode="lines",
                name="收盘价",
                line=dict(color=self.color_palette["price"], width=2),
            ),
            row=1,
            col=1,
        )

        # 添加市场状态区域
        for regime in ["bull", "bear", "neutral"]:
            if regime in regime_data.unique():
                mask = regime_data == regime
                color = self.color_palette[regime]

                fig.add_trace(
                    go.Scatter(
                        x=df.index[mask],
                        y=[regime] * mask.sum(),
                        mode="markers",
                        name=f"{regime.capitalize()}状态",
                        marker=dict(color=color, size=8, symbol="circle"),
                        showlegend=True,
                    ),
                    row=2,
                    col=1,
                )

        # 设置布局
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"), height=500, showlegend=True
        )

        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="状态", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)

        return fig

    def create_feature_analysis_chart(
        self, features_df: pd.DataFrame, title: str = "特征分析"
    ) -> go.Figure:
        """
        创建特征分析图表

        Args:
            features_df: 特征数据DataFrame
            title: 图表标题

        Returns:
            go.Figure: Plotly图表对象
        """
        # 选择要显示的特征
        selected_features = ["VOL", "SPREAD", "EBS", "BUFFETT"]
        available_features = [f for f in selected_features if f in features_df.columns]

        if not available_features:
            return self._create_empty_chart("无可用特征数据")

        # 创建子图
        n_features = len(available_features)
        fig = make_subplots(
            rows=n_features,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=available_features,
        )

        # 为每个特征添加图表
        for i, feature in enumerate(available_features, 1):
            fig.add_trace(
                go.Scatter(
                    x=features_df.index,
                    y=features_df[feature],
                    mode="lines",
                    name=feature,
                    line=dict(width=1),
                ),
                row=i,
                col=1,
            )

        # 设置布局
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            height=300 * n_features,
            showlegend=False,
        )

        return fig

    def create_strategy_performance_chart(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "策略表现",
    ) -> go.Figure:
        """
        创建策略表现图表

        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            title: 图表标题

        Returns:
            go.Figure: Plotly图表对象
        """
        # 计算累计收益
        strategy_cumulative = (1 + returns).cumprod()

        fig = go.Figure()

        # 添加策略累计收益
        fig.add_trace(
            go.Scatter(
                x=strategy_cumulative.index,
                y=strategy_cumulative.values,
                mode="lines",
                name="策略累计收益",
                line=dict(color="blue", width=2),
            )
        )

        # 添加基准累计收益（如果存在）
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    mode="lines",
                    name="基准累计收益",
                    line=dict(color="red", width=2, dash="dash"),
                )
            )

        # 设置布局
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_title="日期",
            yaxis_title="累计收益",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=400,
        )

        return fig

    def create_regime_statistics_chart(
        self, regime_counts: Dict[str, int], title: str = "状态统计"
    ) -> go.Figure:
        """
        创建状态统计图表

        Args:
            regime_counts: 状态计数字典
            title: 图表标题

        Returns:
            go.Figure: Plotly图表对象
        """
        # 准备数据
        regimes = list(regime_counts.keys())
        counts = list(regime_counts.values())
        colors = [self.color_palette.get(regime, "#808080") for regime in regimes]

        # 创建饼图
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=regimes,
                    values=counts,
                    marker=dict(colors=colors),
                    textinfo="percent+label",
                    hole=0.3,
                )
            ]
        )

        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"), height=400)

        return fig

    def _create_empty_chart(self, message: str) -> go.Figure:
        """创建空图表用于错误处理"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=200,
        )
        return fig
