"""
HMM模型模块单元测试
"""

from src.models.hmm_regime_detector import HMMRegimeDetector, MarketStateAnalyzer
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# 设置pandas选项以避免FutureWarning
pd.set_option("future.no_silent_downcasting", True)


class TestHMMRegimeDetector:
    """HMM状态识别器测试类"""

    def setup_method(self):
        """测试初始化"""
        self.detector = HMMRegimeDetector(n_states=3, random_state=42)

        # 创建模拟特征数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 5

        # 创建三个不同状态的数据（高、中、低波动）
        self.X = np.zeros((n_samples, n_features))

        # 状态1：高波动，正收益
        self.X[:300] = np.random.normal(0.1, 0.5, (300, n_features))

        # 状态2：中波动，中性收益
        self.X[300:700] = np.random.normal(0.0, 0.2, (400, n_features))

        # 状态3：低波动，负收益
        self.X[700:] = np.random.normal(-0.05, 0.1, (300, n_features))

        # 创建模拟价格和收益率数据
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
        prices = 100 * np.exp(np.cumsum(self.X[:, 0] / 100))  # 使用第一个特征作为收益率

        self.data = pd.DataFrame(
            {
                "PX": prices,
                "log_ret": self.X[:, 0],
                "VOL": np.abs(self.X[:, 1]),
                "SPREAD": self.X[:, 2],
                "EBS": self.X[:, 3],
                "BUFFETT": self.X[:, 4],
            },
            index=dates,
        )

    def test_initialization(self):
        """测试初始化参数"""
        detector = HMMRegimeDetector(
            n_states=4, covariance_type="full", n_iter=500, tol=1e-6, random_state=123
        )

        assert detector.n_states == 4
        assert detector.model.covariance_type == "full"
        assert detector.model.n_iter == 500
        assert detector.model.tol == 1e-6
        assert not detector._is_fitted

    def test_enforce_min_duration(self):
        """测试最小状态持续时间强制"""
        # 创建有短状态运行的序列
        labels = np.array([0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0])  # 状态1和2都很短

        smoothed = HMMRegimeDetector.enforce_min_duration(labels, min_len=3)

        # 验证短状态被合并
        assert len(np.unique(smoothed)) <= len(np.unique(labels))

        # 验证至少有一些短状态被合并（状态数量减少）
        original_unique_states = len(np.unique(labels))
        smoothed_unique_states = len(np.unique(smoothed))
        assert smoothed_unique_states < original_unique_states

        # 验证结果长度不变
        assert len(smoothed) == len(labels)

    def test_enforce_min_duration_edge_cases(self):
        """测试边界情况"""
        # 测试空数组
        empty_labels = np.array([])
        result = HMMRegimeDetector.enforce_min_duration(empty_labels, min_len=5)
        assert len(result) == 0

        # 测试单元素数组
        single_label = np.array([0])
        result = HMMRegimeDetector.enforce_min_duration(single_label, min_len=5)
        assert len(result) == 1
        assert result[0] == 0

        # 测试所有状态都满足最小长度
        long_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        result = HMMRegimeDetector.enforce_min_duration(long_labels, min_len=2)
        assert np.array_equal(result, long_labels)

    def test_fit_model(self):
        """测试模型训练"""
        # 训练模型
        self.detector.fit(self.X)

        # 验证训练状态
        assert self.detector._is_fitted is True
        assert hasattr(self.detector.model, "means_")
        assert hasattr(self.detector.model, "covars_")
        assert hasattr(self.detector.model, "transmat_")

        # 验证转移矩阵
        transmat = self.detector.model.transmat_
        assert transmat.shape == (3, 3)
        assert np.allclose(transmat.sum(axis=1), 1.0)

    def test_make_sticky(self):
        """测试粘性增强"""
        # 先训练模型
        self.detector.fit(self.X)

        original_transmat = self.detector.model.transmat_.copy()

        # 应用粘性增强
        self.detector.make_sticky(strength=10.0)

        sticky_transmat = self.detector.model.transmat_

        # 验证对角线增强
        for i in range(3):
            assert sticky_transmat[i, i] > original_transmat[i, i]

        # 验证转移矩阵仍然有效
        assert np.allclose(sticky_transmat.sum(axis=1), 1.0)

    def test_make_sticky_without_fit(self):
        """测试未训练模型应用粘性增强"""
        with pytest.raises(ValueError, match="模型尚未训练"):
            self.detector.make_sticky(strength=10.0)

    def test_predict_without_fit(self):
        """测试未训练模型预测"""
        with pytest.raises(ValueError, match="模型尚未训练"):
            self.detector.predict(self.X)

    def test_predict_with_smoothing(self):
        """测试带平滑的预测"""
        # 训练模型
        self.detector.fit(self.X)

        # 预测状态
        states, proba = self.detector.predict(self.X, min_len=20, sticky_strength=5.0)

        # 验证输出形状
        assert len(states) == len(self.X)
        assert proba.shape == (len(self.X), 3)

        # 验证概率有效
        assert np.all(proba >= 0)
        assert np.allclose(proba.sum(axis=1), 1.0)

        # 验证状态标签在有效范围内
        assert np.all(states >= 0)
        assert np.all(states < 3)

    def test_get_state_labels(self):
        """测试状态标签生成"""
        # 创建模拟状态和收益率
        # 状态0:高收益, 状态1:中性, 状态2:低收益
        returns = pd.Series([0.1, 0.1, 0.0, 0.0, -0.1, -0.1])
        states = np.array([0, 0, 1, 1, 2, 2])

        labels = self.detector.get_state_labels(returns, states)

        # 验证标签映射
        assert len(labels) == 3
        assert labels[0] == "Bull"  # 最高收益
        assert labels[2] == "Bear"  # 最低收益
        assert labels[1] == "Neutral"  # 中间收益

    def test_get_state_labels_invalid_length(self):
        """测试长度不匹配的状态标签生成"""
        returns = pd.Series([0.1, 0.1])
        states = np.array([0, 0, 0])

        with pytest.raises(ValueError, match="长度不匹配"):
            self.detector.get_state_labels(returns, states)

    def test_get_model_summary(self):
        """测试模型摘要"""
        # 未训练模型
        summary = self.detector.get_model_summary()
        assert summary["status"] == "未训练"

        # 训练后模型
        self.detector.fit(self.X)
        summary = self.detector.get_model_summary()

        assert summary["status"] == "已训练"
        assert summary["n_states"] == 3
        assert summary["covariance_type"] == "diag"
        assert "converged" in summary
        assert "n_iter" in summary

    def test_validate_model(self):
        """测试模型验证"""
        # 未训练模型
        assert self.detector.validate_model() is False

        # 训练后模型
        self.detector.fit(self.X)
        assert self.detector.validate_model() is True


class TestMarketStateAnalyzer:
    """市场状态分析器测试类"""

    def setup_method(self):
        """测试初始化"""
        self.analyzer = MarketStateAnalyzer()

        # 创建模拟数据
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")

        # 创建三个状态的数据
        states = np.array([0] * 150 + [1] * 200 + [2] * 150)

        self.data = pd.DataFrame(
            {
                "PX": np.random.normal(100, 20, n_samples),
                "log_ret": np.random.normal(0, 0.1, n_samples),
                "VOL": np.random.uniform(0.1, 0.5, n_samples),
                "SPREAD": np.random.uniform(-0.1, 0.1, n_samples),
                "EBS": np.random.uniform(1, 5, n_samples),
                "BUFFETT": np.random.uniform(50, 150, n_samples),
            },
            index=dates,
        )

        self.states = states
        self.state_labels = {0: "Bull", 1: "Neutral", 2: "Bear"}

    def test_analyze_regime_performance(self):
        """测试市场状态性能分析"""
        result = self.analyzer.analyze_regime_performance(
            self.data, self.states, self.state_labels
        )

        # 验证结果结构
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 三个状态

        # 验证列名
        expected_columns = [
            "regime",
            "count",
            "mean_return",
            "std_return",
            "volatility",
            "avg_price",
            "duration_days",
        ]
        assert all(col in result.columns for col in expected_columns)

        # 验证状态完整性
        regimes_in_result = set(result["regime"])
        assert regimes_in_result == {"Bull", "Neutral", "Bear"}

    def test_analyze_regime_performance_invalid_length(self):
        """测试长度不匹配的性能分析"""
        short_states = self.states[:100]

        with pytest.raises(ValueError, match="长度不匹配"):
            self.analyzer.analyze_regime_performance(
                self.data, short_states, self.state_labels
            )

    def test_calculate_transition_matrix(self):
        """测试状态转换矩阵计算"""
        # 创建有明显转换的状态序列
        transition_states = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1])

        matrix = self.analyzer.calculate_transition_matrix(
            transition_states, self.state_labels
        )

        # 验证矩阵结构
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape[0] == 3  # 三个状态
        assert matrix.shape[1] == 3

        # 验证行和为1（或接近1）
        row_sums = matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_calculate_transition_matrix_no_transitions(self):
        """测试没有状态转换的情况"""
        # 创建没有转换的状态序列
        constant_states = np.array([0, 0, 0, 0, 0])

        matrix = self.analyzer.calculate_transition_matrix(
            constant_states, self.state_labels
        )

        # 验证矩阵结构
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] == 3

        # 验证所有值为0（因为没有转换）
        assert (matrix == 0).all().all()

    def test_generate_trading_signals(self):
        """测试交易信号生成"""
        signals = self.analyzer.generate_trading_signals(
            self.data, self.states, self.state_labels
        )

        # 验证结果结构
        assert isinstance(signals, pd.DataFrame)

        # 验证新增列
        expected_columns = ["state", "regime", "position", "strat_ret"]
        assert all(col in signals.columns for col in expected_columns)

        # 验证交易信号
        assert set(signals["position"].unique()) == {-1, 0, 1}

        # 验证策略收益率计算
        bull_mask = signals["regime"] == "Bull"
        bear_mask = signals["regime"] == "Bear"

        # 牛市应该做多（但注意移位）
        shifted_bull_mask = bull_mask.shift(1).fillna(False)
        shifted_bull_mask = shifted_bull_mask.infer_objects(copy=False)
        assert signals.loc[shifted_bull_mask, "position"].eq(1).all()

        # 熊市应该做空（但注意移位）
        shifted_bear_mask = bear_mask.shift(1).fillna(False)
        shifted_bear_mask = shifted_bear_mask.infer_objects(copy=False)
        assert signals.loc[shifted_bear_mask, "position"].eq(-1).all()

        # 验证策略收益率计算正确
        expected_strat_ret = signals["position"] * signals["log_ret"]
        assert np.allclose(signals["strat_ret"], expected_strat_ret)

    def test_generate_trading_signals_edge_cases(self):
        """测试交易信号生成的边界情况"""
        # 创建只有一种状态的数据
        single_state_data = self.data.iloc[:50].copy()
        single_states = np.array([0] * 50)
        single_labels = {0: "Bull"}

        signals = self.analyzer.generate_trading_signals(
            single_state_data, single_states, single_labels
        )

        # 验证只有牛市信号
        assert (signals["regime"] == "Bull").all()
        assert (signals["position"].isin([0, 1])).all()


class TestHMMIntegration:
    """HMM模型集成测试"""

    def test_complete_hmm_workflow(self):
        """测试完整的HMM工作流程"""
        # 创建更真实的数据
        np.random.seed(42)
        n_samples = 2000
        dates = pd.date_range(start="2018-01-01", periods=n_samples, freq="D")

        # 创建三个不同状态的特征数据
        X = np.zeros((n_samples, 5))

        # 状态0：牛市特征（高收益，低波动）
        X[:600] = np.column_stack(
            [
                np.random.normal(0.001, 0.01, 600),  # 收益率
                np.random.normal(0.1, 0.05, 600),  # 波动率
                np.random.normal(0.02, 0.01, 600),  # 趋势
                np.random.normal(3.5, 0.5, 600),  # 股债利差
                np.random.normal(120, 10, 600),  # 巴菲特指数
            ]
        )

        # 状态1：中性市场
        X[600:1400] = np.column_stack(
            [
                np.random.normal(0.000, 0.015, 800),
                np.random.normal(0.2, 0.08, 800),
                np.random.normal(0.00, 0.02, 800),
                np.random.normal(2.5, 0.8, 800),
                np.random.normal(100, 15, 800),
            ]
        )

        # 状态2：熊市特征（负收益，高波动）
        X[1400:] = np.column_stack(
            [
                np.random.normal(-0.001, 0.02, 600),
                np.random.normal(0.3, 0.1, 600),
                np.random.normal(-0.02, 0.015, 600),
                np.random.normal(1.5, 0.6, 600),
                np.random.normal(80, 12, 600),
            ]
        )

        # 创建数据框
        prices = 100 * np.exp(np.cumsum(X[:, 0]))
        data = pd.DataFrame(
            {
                "PX": prices,
                "log_ret": X[:, 0],
                "VOL": X[:, 1],
                "SPREAD": X[:, 2],
                "EBS": X[:, 3],
                "BUFFETT": X[:, 4],
            },
            index=dates,
        )

        # 完整工作流程
        detector = HMMRegimeDetector(n_states=3, random_state=42)
        detector.fit(X)

        states, proba = detector.predict(X, min_len=30, sticky_strength=8.0)

        state_labels = detector.get_state_labels(data["log_ret"], states)

        analyzer = MarketStateAnalyzer()
        performance = analyzer.analyze_regime_performance(data, states, state_labels)
        signals = analyzer.generate_trading_signals(data, states, state_labels)

        # 验证结果
        assert detector.validate_model() is True
        assert len(performance) == 3
        assert len(signals) == n_samples

        # 验证交易信号有效性
        assert signals["strat_ret"].std() > 0
        assert signals["position"].nunique() == 3  # 三种仓位
