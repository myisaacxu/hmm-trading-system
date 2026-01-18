"""
性能指标计算模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """风险指标数据类"""

    volatility: float  # 波动率
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR
    skewness: float  # 偏度
    kurtosis: float  # 峰度


class PerformanceMetrics:
    """性能指标计算类"""

    @staticmethod
    def calculate_risk_metrics(returns: pd.Series) -> RiskMetrics:
        """
        计算风险指标

        Args:
            returns: 收益率序列

        Returns:
            RiskMetrics: 风险指标对象
        """
        if len(returns) < 2:
            return RiskMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        # 年化波动率
        volatility = returns.std() * np.sqrt(252)

        # VaR和CVaR（95%置信水平）
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        # 偏度和峰度
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        return RiskMetrics(
            float(volatility),
            float(var_95),
            float(cvar_95),
            float(skewness),
            float(kurtosis),
        )

    @staticmethod
    def calculate_tracking_error(
        portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """
        计算跟踪误差

        Args:
            portfolio_returns: 组合收益率
            benchmark_returns: 基准收益率

        Returns:
            float: 年化跟踪误差
        """
        # 对齐数据
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1)
        aligned_data.columns = ["portfolio", "benchmark"]
        aligned_data = aligned_data.dropna()

        if len(aligned_data) < 2:
            return 0.0

        # 计算跟踪误差（超额收益的标准差）
        excess_returns = aligned_data["portfolio"] - aligned_data["benchmark"]
        tracking_error = excess_returns.std() * np.sqrt(252)

        return tracking_error

    @staticmethod
    def calculate_information_ratio(
        portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """
        计算信息比率

        Args:
            portfolio_returns: 组合收益率
            benchmark_returns: 基准收益率

        Returns:
            float: 信息比率
        """
        # 对齐数据
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1)
        aligned_data.columns = ["portfolio", "benchmark"]
        aligned_data = aligned_data.dropna()

        if len(aligned_data) < 2:
            return 0.0

        # 计算超额收益和跟踪误差
        excess_returns = aligned_data["portfolio"] - aligned_data["benchmark"]
        tracking_error = excess_returns.std() * np.sqrt(252)

        if tracking_error == 0:
            return 0.0

        # 信息比率 = 超额收益均值 / 跟踪误差
        information_ratio = excess_returns.mean() * 252 / tracking_error

        return information_ratio

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, max_drawdown: float) -> float:
        """
        计算Calmar比率

        Args:
            returns: 收益率序列
            max_drawdown: 最大回撤

        Returns:
            float: Calmar比率
        """
        if len(returns) < 2 or max_drawdown == 0:
            return 0.0

        # 年化收益率
        total_return = float((1 + returns).prod())
        cagr = total_return ** (252 / len(returns)) - 1

        # Calmar比率 = 年化收益率 / 最大回撤
        calmar_ratio = cagr / abs(max_drawdown)

        return calmar_ratio

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """
        计算Sortino比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率

        Returns:
            float: Sortino比率
        """
        if len(returns) < 2:
            return 0.0

        # 计算下行风险（只考虑负收益）
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            downside_risk = 0.0
        else:
            downside_risk = downside_returns.std() * np.sqrt(252)

        if downside_risk == 0:
            return 0.0

        # 超额收益
        excess_returns = returns - risk_free_rate / 252

        # Sortino比率 = 超额收益均值 / 下行风险
        sortino_ratio = excess_returns.mean() * 252 / downside_risk

        return sortino_ratio

    @staticmethod
    def calculate_rolling_metrics(
        returns: pd.Series, window: int = 252
    ) -> pd.DataFrame:
        """
        计算滚动指标

        Args:
            returns: 收益率序列
            window: 滚动窗口大小

        Returns:
            pd.DataFrame: 滚动指标数据
        """
        if len(returns) < window:
            return pd.DataFrame()

        rolling_metrics = []

        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i - window : i]

            # 计算滚动指标
            rolling_return = float((1 + window_returns).prod()) - 1
            rolling_volatility = window_returns.std() * np.sqrt(252)
            rolling_sharpe = window_returns.mean() / window_returns.std() * np.sqrt(252)

            # 计算滚动最大回撤
            cumulative = (1 + window_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            rolling_drawdown = (cumulative - rolling_max) / rolling_max
            rolling_max_drawdown = rolling_drawdown.min()

            rolling_metrics.append(
                {
                    "date": returns.index[i - 1],
                    "return": rolling_return,
                    "volatility": rolling_volatility,
                    "sharpe": rolling_sharpe,
                    "max_drawdown": rolling_max_drawdown,
                }
            )

        return pd.DataFrame(rolling_metrics).set_index("date")
