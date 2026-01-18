"""
市场状态分析器模块
负责分析HMM识别的市场状态，提供状态特征分析和可视化支持
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from src.utils.logger import global_logger


@dataclass
class StateAnalysis:
    """市场状态分析结果"""

    state_id: int
    state_name: str
    duration_ratio: float
    average_return: float
    volatility: float
    transition_probabilities: List[float]
    characteristic_features: Dict[str, float]


class MarketStateAnalyzer:
    """
    市场状态分析器
    负责分析HMM识别的市场状态，提供状态特征统计和解释
    """

    def __init__(self, state_config: Optional[Dict[int, Dict[str, Any]]] = None):
        """
        初始化市场状态分析器

        Args:
            state_config: 状态配置字典，包含状态名称和颜色
        """
        # 默认状态配置
        default_config = {
            0: {"name": "下跌状态", "color": "red"},
            1: {"name": "震荡状态", "color": "orange"},
            2: {"name": "上涨状态", "color": "green"},
        }

        # 使用自定义配置或默认配置
        self.state_config = state_config or default_config

        # 提取状态名称和颜色映射
        self.state_names = {
            state_id: config["name"] for state_id, config in self.state_config.items()
        }
        self.state_colors = {
            state_id: config["color"] for state_id, config in self.state_config.items()
        }

        # 技术指标列表，用于特征分析
        self.tech_indicators = [
            "rsi",
            "macd",
            "bollinger_upper",
            "bollinger_lower",
            "atr",
            "adx",
            "cci",
        ]

    def analyze_states(
        self,
        price_data: pd.DataFrame,
        regime_series: pd.Series,
        feature_data: pd.DataFrame,
    ) -> Dict[int, StateAnalysis]:
        """
        分析市场状态特征

        Args:
            price_data: 价格数据，必须包含'close'列
            regime_series: 状态序列
            feature_data: 特征数据

        Returns:
            Dict[int, StateAnalysis]: 状态分析结果
        """
        # 参数验证
        if price_data.empty or regime_series.empty:
            global_logger.warning("价格数据或状态序列为空，无法进行状态分析")
            return {}

        if "close" not in price_data.columns:
            raise ValueError("price_data必须包含'close'列")

        # 确保索引对齐
        if not price_data.index.equals(regime_series.index):
            # 尝试对齐索引
            regime_series = regime_series.reindex(price_data.index)
            if regime_series.isnull().any():
                raise ValueError("price_data和regime_series的索引无法对齐")

        # 对齐数据
        aligned_data = price_data.copy()
        aligned_data["regime"] = regime_series

        # 确保特征数据对齐
        if not feature_data.empty:
            aligned_data = aligned_data.join(feature_data, how="inner")

        # 获取所有唯一状态
        unique_states = sorted(regime_series.unique())
        global_logger.info(f"开始分析市场状态，共检测到 {len(unique_states)} 种状态")

        analysis_results = {}

        for state_id in unique_states:
            state_data = aligned_data[aligned_data["regime"] == state_id]

            if len(state_data) == 0:
                continue

            # 计算状态持续时间比例
            total_duration = len(regime_series)
            state_duration = len(state_data)
            duration_ratio = state_duration / total_duration

            # 计算状态收益特征
            returns = state_data["close"].pct_change().dropna()
            average_return = returns.mean() if len(returns) > 0 else 0.0
            volatility = returns.std() if len(returns) > 0 else 0.0

            # 计算状态转移概率
            transition_probs = self._calculate_transition_probabilities(
                regime_series, state_id, unique_states
            )

            # 计算状态特征统计
            characteristic_features = self._calculate_state_features(state_data)

            analysis_results[state_id] = StateAnalysis(
                state_id=state_id,
                state_name=self.state_names.get(state_id, f"状态{state_id}"),
                duration_ratio=duration_ratio,
                average_return=average_return,
                volatility=volatility,
                transition_probabilities=transition_probs,
                characteristic_features=characteristic_features,
            )

            # 记录状态分析结果
            global_logger.log_market_state(
                timestamp=price_data.index[-1],
                state=self.state_names.get(state_id, f"状态{state_id}"),
                state_id=state_id,
                duration_ratio=duration_ratio,
                average_return=average_return,
                volatility=volatility,
                sample_count=len(state_data)
            )

        return analysis_results

    def _calculate_transition_probabilities(
        self, regime_series: pd.Series, target_state: int, unique_states: List[int]
    ) -> List[float]:
        """
        计算状态转移概率

        Args:
            regime_series: 状态序列
            target_state: 目标状态
            unique_states: 所有唯一状态列表

        Returns:
            List[float]: 转移概率列表
        """
        if len(regime_series) < 2:
            return [0.0] * len(unique_states)

        # 创建状态ID到索引的映射
        state_to_idx = {state: i for i, state in enumerate(unique_states)}
        transition_probs = [0.0] * len(unique_states)

        # 统计从目标状态转移到其他状态的次数
        transitions_from_target = 0
        for i in range(1, len(regime_series)):
            if regime_series.iloc[i - 1] == target_state:
                transitions_from_target += 1
                next_state = regime_series.iloc[i]
                if next_state in state_to_idx:
                    transition_probs[state_to_idx[next_state]] += 1

        # 计算转移概率
        if transitions_from_target > 0:
            transition_probs = [
                count / transitions_from_target for count in transition_probs
            ]

        return transition_probs

    def _calculate_state_features(self, state_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算状态特征统计

        Args:
            state_data: 状态数据

        Returns:
            Dict[str, float]: 特征统计
        """
        features = {}

        # 基本价格特征
        if "close" in state_data.columns:
            features["price_mean"] = state_data["close"].mean()
            features["price_std"] = state_data["close"].std()
            features["price_range"] = (
                state_data["close"].max() - state_data["close"].min()
            )

        # 成交量特征
        if "volume" in state_data.columns:
            features["volume_mean"] = state_data["volume"].mean()
            features["volume_std"] = state_data["volume"].std()

        # 技术指标特征（如果存在）
        for indicator in self.tech_indicators:
            if indicator in state_data.columns:
                features[f"{indicator}_mean"] = state_data[indicator].mean()
                features[f"{indicator}_std"] = state_data[indicator].std()

        return features

    def get_state_summary(
        self, analysis_results: Dict[int, StateAnalysis]
    ) -> pd.DataFrame:
        """
        生成状态分析摘要

        Args:
            analysis_results: 状态分析结果

        Returns:
            pd.DataFrame: 状态摘要表格
        """
        if not analysis_results:
            return pd.DataFrame(
                columns=[
                    "状态ID",
                    "状态名称",
                    "持续时间比例",
                    "平均收益率",
                    "波动率",
                    "样本数量",
                ]
            )

        summary_data = []

        for state_id, analysis in analysis_results.items():
            # 计算实际样本数量（基于特征数据行数）
            sample_count = (
                len(analysis.characteristic_features)
                if analysis.characteristic_features
                else 0
            )

            summary_data.append(
                {
                    "状态ID": state_id,
                    "状态名称": analysis.state_name,
                    "持续时间比例": f"{analysis.duration_ratio:.2%}",
                    "平均收益率": f"{analysis.average_return:.4f}",
                    "波动率": f"{analysis.volatility:.4f}",
                    "样本数量": sample_count,
                }
            )

        return pd.DataFrame(summary_data)

    def identify_dominant_state(
        self, analysis_results: Dict[int, StateAnalysis]
    ) -> Optional[int]:
        """
        识别主导状态

        Args:
            analysis_results: 状态分析结果

        Returns:
            Optional[int]: 主导状态ID，基于持续时间比例
        """
        if not analysis_results:
            return None

        # 根据持续时间比例识别主导状态
        dominant_state = max(
            analysis_results.items(), key=lambda x: x[1].duration_ratio
        )[0]

        return dominant_state

    def detect_state_transitions(self, regime_series: pd.Series) -> List[Dict]:
        """
        检测状态转换点

        Args:
            regime_series: 状态序列

        Returns:
            List[Dict]: 状态转换点列表，包含转换前后状态、转换索引和持续时间
        """
        transitions: List[Dict[str, Any]] = []

        if len(regime_series) < 2:
            global_logger.debug("状态序列长度不足，无法检测状态转换")
            return transitions

        current_state = regime_series.iloc[0]
        start_index = 0
        
        # 记录初始状态
        initial_date = regime_series.index[0] if isinstance(regime_series.index, pd.DatetimeIndex) else None
        if initial_date:
            global_logger.log_market_state(
                timestamp=initial_date,
                state=self.state_names.get(current_state, f"状态{current_state}"),
                state_id=current_state,
                event="initial_state"
            )

        for i in range(1, len(regime_series)):
            next_state = regime_series.iloc[i]
            transition_date = regime_series.index[i] if isinstance(regime_series.index, pd.DatetimeIndex) else None

            if next_state != current_state:
                # 记录状态转换
                transition = {
                    "from_state": current_state,
                    "to_state": next_state,
                    "transition_index": i,
                    "transition_date": transition_date,
                    "duration": i - start_index,
                    "from_state_name": self.state_names.get(
                        current_state, f"状态{current_state}"
                    ),
                    "to_state_name": self.state_names.get(
                        next_state, f"状态{next_state}"
                    ),
                }
                transitions.append(transition)
                
                # 记录状态转换日志
                if transition_date:
                    global_logger.log_market_state(
                        timestamp=transition_date,
                        state=self.state_names.get(next_state, f"状态{next_state}"),
                        state_id=next_state,
                        from_state=current_state,
                        from_state_name=transition["from_state_name"],
                        event="state_transition",
                        duration=i - start_index
                    )

                current_state = next_state
                start_index = i

        return transitions

    def get_state_colors(self) -> Dict[int, str]:
        """
        获取状态颜色映射

        Returns:
            Dict[int, str]: 状态ID到颜色的映射
        """
        return self.state_colors

    def get_state_names(self) -> Dict[int, str]:
        """
        获取状态名称映射

        Returns:
            Dict[int, str]: 状态ID到名称的映射
        """
        return self.state_names
