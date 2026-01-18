"""
HMM模型模块 - 按照原有程序逻辑实现
负责市场状态识别和状态平滑处理
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional, Dict, List
from hmmlearn import hmm
from src.config.config import HMMConfig


class HMMRegimeDetector:
    """HMM市场状态识别器，按照原有程序逻辑实现"""

    def __init__(
        self,
        config: Optional[HMMConfig] = None,
        n_states: int = None,
        covariance_type: str = None,
        n_iter: int = None,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """
        初始化HMM状态识别器，与参考文件保持一致

        Args:
            config: HMM配置对象，优先使用配置参数
            n_states: 状态数量（如果未提供config则使用此参数）
            covariance_type: 协方差类型（如果未提供config则使用此参数）
            n_iter: 最大迭代次数（如果未提供config则使用此参数）
            tol: 收敛容忍度
            random_state: 随机种子
        """
        # 优先使用配置对象
        if config is not None:
            self._config = config
            self._n_states = config.n_states
            self._covariance_type = config.covariance_type
            self._n_iter = config.n_iter
        else:
            # 向后兼容：使用直接参数
            self._config = HMMConfig()
            # 如果提供了直接参数，更新配置对象
            if n_states is not None:
                self._config.n_states = n_states
            if covariance_type is not None:
                self._config.covariance_type = covariance_type
            if n_iter is not None:
                self._config.n_iter = n_iter
            self._n_states = self._config.n_states
            self._covariance_type = self._config.covariance_type
            self._n_iter = self._config.n_iter

        self.model = hmm.GaussianHMM(
            n_components=self._n_states,
            covariance_type=self._covariance_type,
            n_iter=self._n_iter,
            tol=tol,
            random_state=random_state,
        )
        self._is_fitted = False

    @property
    def config(self):
        """配置对象属性"""
        return self._config

    @property
    def n_states(self):
        """状态数量属性（与配置对象同步）"""
        return self._config.n_states

    @property
    def is_fitted(self) -> bool:
        """检查模型是否已拟合"""
        return self._is_fitted

    @staticmethod
    def enforce_min_duration(labels: np.ndarray, min_len: int = 10) -> np.ndarray:
        """
        合并短状态运行（< min_len）到较长的相邻状态，与参考文件完全一致

        Args:
            labels: 原始状态标签数组
            min_len: 最小状态持续时间

        Returns:
            平滑后的状态标签数组
        """
        if len(labels) == 0:
            return labels

        s = np.array(labels, copy=True)
        n = len(s)
        i = 0
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            run_len = j - i
            if run_len < min_len:
                left = s[i - 1] if i > 0 else None
                right = s[j] if j < n else None
                if left is None and right is not None:
                    s[i:j] = right
                elif right is None and left is not None:
                    s[i:j] = left
                elif left is not None and right is not None:
                    # 比较相邻运行的长度
                    L = i - 1
                    while L - 1 >= 0 and s[L - 1] == left:
                        L -= 1
                    left_len = i - L
                    R = j
                    while R + 1 < n and s[R + 1] == right:
                        R += 1
                    right_len = R - j + 1
                    s[i:j] = left if left_len >= right_len else right
            i = j
        return s

    def fit(self, X: np.ndarray) -> "HMMRegimeDetector":
        """
        训练HMM模型

        Args:
            X: 特征矩阵

        Returns:
            训练后的模型实例
        """
        # 检查输入数据是否有效
        if len(X) == 0:
            raise ValueError("无法训练HMM模型：特征矩阵为空")

        if len(X) < 10:
            warnings.warn(f"训练数据量较少（{len(X)}个样本），可能影响模型质量")

        # 检查数据是否有足够的变异性
        if X.std(axis=0).sum() < 1e-10:
            raise ValueError("特征矩阵缺乏变异性，无法训练HMM模型")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model.fit(X)

        self._is_fitted = True
        return self

    def make_sticky(self, strength: float = 10.0) -> "HMMRegimeDetector":
        """
        通过增强转移矩阵对角线使状态更倾向于保持不变

        Args:
            strength: 粘性强度

        Returns:
            修改后的模型实例
        """
        if not self._is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        A = self.model.transmat_
        A = A + strength * np.eye(self.n_states)
        self.model.transmat_ = A / A.sum(axis=1, keepdims=True)

        return self

    def predict(
        self, X: np.ndarray, min_len: int = 10, sticky_strength: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测市场状态，与参考文件完全一致

        Args:
            X: 特征矩阵
            min_len: 最小状态持续时间
            sticky_strength: 粘性强度

        Returns:
            状态标签和状态概率
        """
        if not self._is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        # 检查输入数据
        if len(X) == 0:
            return np.array([]), np.array([])

        if sticky_strength is not None:
            self.make_sticky(sticky_strength)

        states = self.model.predict(X)

        # 应用状态平滑，与参考文件一致，总是应用
        states = self.enforce_min_duration(states, min_len=min_len)

        proba = self.model.predict_proba(X)

        return states, proba

    def smooth_regimes(self, regimes: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        平滑状态序列，使用移动平均窗口

        Args:
            regimes: 原始状态序列
            window_size: 平滑窗口大小

        Returns:
            平滑后的状态序列
        """
        if len(regimes) < window_size:
            return regimes

        # 使用移动平均平滑状态
        smoothed = []
        for i in range(len(regimes)):
            start = max(0, i - window_size // 2)
            end = min(len(regimes), i + window_size // 2 + 1)
            window = regimes[start:end]

            # 选择窗口中最常见的状态
            if len(window) > 0:
                mode = np.bincount(window).argmax()
                smoothed.append(mode)
            else:
                smoothed.append(regimes[i])

        return np.array(smoothed)

    def get_state_labels(
        self, returns: pd.Series, states: np.ndarray
    ) -> Dict[int, str]:
        """
        根据收益率排序状态并映射到市场状态标签

        Args:
            returns: 收益率序列
            states: 状态标签数组

        Returns:
            状态标签映射字典
        """
        if len(returns) != len(states):
            raise ValueError("收益率序列和状态标签长度不匹配")

        # 按平均收益率排序状态
        state_returns = []
        for state in range(self.n_states):
            state_mask = states == state
            if state_mask.any():
                mean_return = returns[state_mask].mean()
                state_returns.append((state, mean_return))

        # 按收益率降序排序
        state_returns.sort(key=lambda x: x[1], reverse=True)

        # 映射到市场状态标签
        labels = {}
        if state_returns:
            # 最高收益率为牛市，最低为熊市
            labels[state_returns[0][0]] = "Bull"
            labels[state_returns[-1][0]] = "Bear"

            # 其余为中性
            for state, _ in state_returns[1:-1]:
                labels[state] = "Neutral"

        return labels

    def get_model_summary(self) -> Dict:
        """获取模型摘要信息"""
        if not self._is_fitted:
            return {"status": "未训练"}

        summary = {
            "status": "已训练",
            "n_states": self.n_states,
            "covariance_type": self.model.covariance_type,
            "converged": self.model.monitor_.converged,
            "n_iter": self.model.n_iter,
            "log_likelihood": (
                self.model.score(self.model._X) if hasattr(self.model, "_X") else None
            ),
        }

        return summary

    def validate_model(self) -> bool:
        """验证模型的有效性"""
        if not self._is_fitted:
            return False

        # 检查转移矩阵
        if not hasattr(self.model, "transmat_"):
            return False

        # 检查转移矩阵是否有效
        transmat = self.model.transmat_
        if np.any(transmat < 0) or not np.allclose(transmat.sum(axis=1), 1.0):
            return False

        # 检查是否收敛
        if not self.model.monitor_.converged:
            return False

        return True


class MarketStateAnalyzer:
    """市场状态分析器，按照原有程序逻辑实现"""

    def __init__(self, config: Optional[HMMConfig] = None):
        """
        初始化市场状态分析器

        Args:
            config: HMM配置对象，如果未提供则创建默认配置
        """
        # 使用配置对象或创建默认配置
        if config is not None:
            self._config = config
        else:
            self._config = HMMConfig()

        self._state_data = None

    @property
    def config(self):
        """配置对象属性"""
        return self._config

    def analyze_regime_performance(
        self, data: pd.DataFrame, states: np.ndarray, state_labels: Dict[int, str]
    ) -> pd.DataFrame:
        """
        分析各市场状态的性能表现

        Args:
            data: 包含价格和收益率的数据框
            states: 状态标签数组
            state_labels: 状态标签映射

        Returns:
            各状态性能分析数据框
        """
        if len(data) != len(states):
            raise ValueError("数据框和状态标签长度不匹配")

        analysis_data = data.copy()
        analysis_data["state"] = states
        analysis_data["regime"] = analysis_data["state"].map(state_labels)

        # 计算各状态的统计指标
        regime_stats = []
        for regime in ["Bull", "Bear", "Neutral"]:
            regime_data = analysis_data[analysis_data["regime"] == regime]
            if not regime_data.empty:
                stats = {
                    "regime": regime,
                    "count": len(regime_data),
                    "mean_return": regime_data["log_ret"].mean(),
                    "std_return": regime_data["log_ret"].std(),
                    "volatility": regime_data["VOL"].mean(),
                    "avg_price": regime_data["PX"].mean(),
                    "duration_days": len(regime_data),
                }
                regime_stats.append(stats)

        return pd.DataFrame(regime_stats)

    def calculate_transition_matrix(
        self, states: np.ndarray, state_labels: Dict[int, str]
    ) -> pd.DataFrame:
        """
        计算状态转换矩阵

        Args:
            states: 状态标签数组
            state_labels: 状态标签映射

        Returns:
            状态转换矩阵
        """
        # 将数值状态转换为标签
        regime_states = [state_labels.get(state, "Unknown") for state in states]

        # 计算状态转换
        transitions: list[tuple[str, str]] = []
        prev_state = None

        for state in regime_states:
            if prev_state is not None and prev_state != state:
                transitions.append((prev_state, state))
            prev_state = state

        # 创建转换矩阵
        if transitions:
            transition_df = pd.DataFrame(transitions, columns=["From", "To"])
            transition_matrix = pd.crosstab(
                transition_df["From"], transition_df["To"], normalize="index"
            )
            return transition_matrix
        else:
            # 如果没有转换，返回空矩阵
            regime_types = list(set(state_labels.values()))
            return pd.DataFrame(0, index=regime_types, columns=regime_types)

    def generate_trading_signals(
        self, data: pd.DataFrame, states: np.ndarray, state_labels: Dict[int, str]
    ) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            data: 数据框
            states: 状态标签
            state_labels: 状态标签映射

        Returns:
            包含交易信号的数据框
        """
        signals_data = data.copy()
        signals_data["state"] = states
        signals_data["regime"] = signals_data["state"].map(state_labels)

        # 生成交易信号：牛市做多，熊市做空，中性观望
        signals_data["position"] = 0
        signals_data.loc[signals_data["regime"] == "Bull", "position"] = 1
        signals_data.loc[signals_data["regime"] == "Bear", "position"] = -1

        # 应用次日执行（移位持仓）
        signals_data["position"] = signals_data["position"].shift(1).fillna(0)

        # 计算策略收益率
        signals_data["strat_ret"] = signals_data["position"] * signals_data["log_ret"]

        return signals_data
