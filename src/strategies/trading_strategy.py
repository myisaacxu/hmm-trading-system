"""
交易策略模块
基于市场状态生成交易信号和评估策略表现
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from src.utils.logger import global_logger


@dataclass
class TradingSignal:
    """交易信号数据类"""

    date: pd.Timestamp
    regime: str
    position: int  # 1: 做多, -1: 做空, 0: 空仓
    price: float


@dataclass
class StrategyPerformance:
    """策略表现数据类"""

    total_return: float
    cagr: float  # 年化收益率
    sharpe: float  # 夏普比率
    max_drawdown: float  # 最大回撤
    win_rate: float  # 胜率
    profit_factor: float  # 盈亏比
    num_trades: int  # 交易次数


class TradingStrategy:
    """交易策略类"""

    def __init__(self, transaction_cost: float = 0.001):
        """
        初始化交易策略

        Args:
            transaction_cost: 交易成本（百分比）
        """
        self.transaction_cost = transaction_cost
        self.regime_mapping = {
            "bull": 1,  # 牛市做多
            "bear": -1,  # 熊市做空
            "neutral": 0,  # 中性空仓
        }

    def generate_signals(
        self, price_data: pd.DataFrame, regime_series: Union[pd.Series, np.ndarray]
    ) -> Dict:
        """
        基于市场状态生成交易信号

        Args:
            price_data: 价格数据DataFrame
            regime_series: 市场状态序列

        Returns:
            Dict: 包含信号和表现的字典
        """
        # 检查空数据
        if price_data.empty:
            global_logger.warning("价格数据为空，无法生成交易信号")
            return {
                "signals_df": pd.DataFrame(),
                "performance": StrategyPerformance(
                    total_return=0.0,
                    cagr=0.0,
                    sharpe=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    num_trades=0,
                ),
                "returns": pd.Series(),
            }

        # 检查regime_series是否为空
        if isinstance(regime_series, np.ndarray):
            if len(regime_series) == 0:
                global_logger.warning("状态序列为空，无法生成交易信号")
                return {
                    "signals_df": pd.DataFrame(),
                    "performance": StrategyPerformance(
                        total_return=0.0,
                        cagr=0.0,
                        sharpe=0.0,
                        max_drawdown=0.0,
                        win_rate=0.0,
                        profit_factor=0.0,
                        num_trades=0,
                    ),
                    "returns": pd.Series(),
                }
            # 将numpy.ndarray转换为pd.Series
            regime_series = pd.Series(regime_series, index=price_data.index)
        elif regime_series.empty:
            global_logger.warning("状态序列为空，无法生成交易信号")
            return {
                "signals_df": pd.DataFrame(),
                "performance": StrategyPerformance(
                    total_return=0.0,
                    cagr=0.0,
                    sharpe=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    num_trades=0,
                ),
                "returns": pd.Series(),
            }

        # 确保数据对齐
        aligned_data = price_data.copy()
        aligned_data["regime"] = regime_series
        global_logger.info(f"开始生成交易信号，数据长度: {len(aligned_data)}")

        # 生成交易信号
        signals = []
        positions = []

        for date, row in aligned_data.iterrows():
            regime = row["regime"]
            close_price = row["close"]

            # 根据状态映射确定仓位
            position = self.regime_mapping.get(regime, 0)

            signal = TradingSignal(
                date=pd.Timestamp(date),
                regime=regime,
                position=position,
                price=close_price,
            )

            signals.append(signal)
            positions.append(position)
            
            # 记录交易信号
            global_logger.log_trading_signal(signal)

        # 转换为DataFrame
        signals_df = pd.DataFrame(
            [
                {
                    "date": s.date,
                    "regime": s.regime,
                    "position": s.position,
                    "price": s.price,
                }
                for s in signals
            ]
        )
        signals_df.set_index("date", inplace=True)

        # 计算策略表现
        performance = self.calculate_performance(aligned_data, positions)
        
        # 记录策略表现
        global_logger.log_performance(performance)

        return {
            "signals_df": signals_df,
            "performance": performance,
            "returns": performance.returns if hasattr(performance, "returns") else None,
        }

    def calculate_performance(
        self, data: pd.DataFrame, positions: List[int]
    ) -> StrategyPerformance:
        """
        计算策略表现指标

        Args:
            data: 包含价格和状态的数据
            positions: 仓位序列

        Returns:
            StrategyPerformance: 策略表现对象
        """
        if len(data) < 2:
            return StrategyPerformance(
                total_return=0.0,
                cagr=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                num_trades=0,
            )

        # 计算价格收益率
        price_returns = data["close"].pct_change().dropna()

        # 计算策略收益率（考虑仓位和交易成本）
        strategy_returns = self._calculate_strategy_returns(price_returns, positions)

        # 计算各种指标
        total_return = self._calculate_total_return(strategy_returns)
        cagr = self._calculate_cagr(strategy_returns, len(data))
        sharpe = self._calculate_sharpe_ratio(strategy_returns)
        max_drawdown = self._calculate_max_drawdown(strategy_returns)
        win_rate, profit_factor = self._calculate_trade_metrics(strategy_returns)
        num_trades = self._count_trades(positions)

        # 创建策略表现对象
        performance = StrategyPerformance(
            total_return=total_return,
            cagr=cagr,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
        )

        # 添加收益率序列供图表使用
        performance.returns = strategy_returns

        return performance

    def _calculate_strategy_returns(
        self, price_returns: pd.Series, positions: List[int]
    ) -> pd.Series:
        """计算策略收益率序列"""
        # 确保positions长度匹配
        if len(positions) != len(price_returns) + 1:
            positions = positions[1:]  # 调整位置以匹配收益率序列

        # 计算策略收益率（仓位 * 价格收益率）
        strategy_returns = price_returns * positions[:-1]

        # 应用交易成本（仅在仓位变化时）
        position_changes = np.diff(positions) != 0
        
        # 记录仓位变化和交易执行
        for i, (change, date, prev_pos, curr_pos, price) in enumerate(zip(
            position_changes,
            price_returns.index,
            positions[:-1],
            positions[1:],
            price_returns.index.map(lambda x: price_returns.name if hasattr(price_returns, 'name') else x)
        )):
            if change:
                # 确定交易动作
                if prev_pos == 0 and curr_pos > 0:
                    action = "BUY"
                elif prev_pos > 0 and curr_pos == 0:
                    action = "SELL"
                elif prev_pos == 0 and curr_pos < 0:
                    action = "SHORT"
                elif prev_pos < 0 and curr_pos == 0:
                    action = "COVER"
                elif prev_pos < 0 and curr_pos > 0:
                    action = "COVER_BUY"
                elif prev_pos > 0 and curr_pos < 0:
                    action = "SELL_SHORT"
                else:
                    action = "HOLD"
                
                # 记录交易执行
                global_logger.log_trade_execution(
                    date=pd.Timestamp(date),
                    action=action,
                    price=price,
                    quantity=abs(curr_pos - prev_pos),
                    previous_position=prev_pos,
                    new_position=curr_pos
                )
        
        # 应用交易成本
        strategy_returns.iloc[position_changes] -= self.transaction_cost

        return strategy_returns

    def _calculate_total_return(self, returns: pd.Series) -> float:
        """计算总收益率"""
        return float((1 + returns).prod()) - 1

    def _calculate_cagr(self, returns: pd.Series, num_days: int) -> float:
        """计算年化收益率"""
        total_return = self._calculate_total_return(returns)
        years = num_days / 365.25
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate / 252  # 日度无风险利率

        if len(excess_returns) < 2 or excess_returns.std() == 0:
            return 0.0

        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        return drawdown.min() if not drawdown.empty else 0.0

    def _calculate_trade_metrics(self, returns: pd.Series) -> Tuple[float, float]:
        """计算交易指标（胜率、盈亏比）"""
        if len(returns) == 0:
            return 0.0, 0.0

        # 识别交易（非零收益率视为交易）
        trades = returns[returns != 0]

        if len(trades) == 0:
            return 0.0, 0.0

        # 计算胜率
        winning_trades = trades[trades > 0]
        win_rate = len(winning_trades) / len(trades)

        # 计算盈亏比
        total_profit = winning_trades.sum()
        total_loss = trades[trades < 0].sum()

        if total_loss == 0:
            profit_factor = float("inf") if total_profit > 0 else 0.0
        else:
            profit_factor = abs(total_profit / total_loss)

        return win_rate, profit_factor

    def _count_trades(self, positions: List[int]) -> int:
        """计算交易次数"""
        if len(positions) < 2:
            return 0

        # 计算仓位变化次数（不包括初始建仓）
        position_changes = np.diff(positions) != 0
        return int(position_changes.sum())

    def generate_buy_hold_signals(self, price_data: pd.DataFrame) -> Dict:
        """
        生成买入持有策略信号（作为基准）

        Args:
            price_data: 价格数据DataFrame

        Returns:
            Dict: 买入持有策略结果
        """
        # 始终持有多头仓位
        positions = [1] * len(price_data)

        # 计算买入持有表现
        performance = self.calculate_performance(price_data, positions)

        return {
            "signals_df": None,  # 买入持有没有具体信号
            "performance": performance,
            "returns": performance.returns,
        }
