"""
HMM模型配置集成测试
测试重构后的HMM模型模块如何使用配置模块
"""

from src.models.hmm_regime_detector import HMMRegimeDetector, MarketStateAnalyzer
from src.config.config import HMMConfig
import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestHMMRegimeDetectorWithConfig:
    """测试使用配置模块的HMMRegimeDetector"""

    def test_hmm_regime_detector_with_config_object(self):
        """测试使用配置对象初始化HMMRegimeDetector"""
        # 当使用配置对象创建HMMRegimeDetector时
        config = HMMConfig(n_states=4, covariance_type="full", n_iter=500)
        detector = HMMRegimeDetector(config=config)

        # 应该使用配置对象中的参数
        assert detector.config == config
        assert detector.n_states == 4
        assert detector.model.n_components == 4
        assert detector.model.covariance_type == "full"
        assert detector.model.n_iter == 500

    def test_hmm_regime_detector_without_config(self):
        """测试不使用配置对象初始化HMMRegimeDetector（向后兼容）"""
        # 当不使用配置对象创建HMMRegimeDetector时
        detector = HMMRegimeDetector(
            n_states=5, covariance_type="spherical", n_iter=200
        )

        # 应该使用直接参数并创建默认配置
        assert isinstance(detector.config, HMMConfig)
        assert detector.n_states == 5
        assert detector.model.n_components == 5
        assert detector.model.covariance_type == "spherical"
        assert detector.model.n_iter == 200

    def test_hmm_regime_detector_default_values(self):
        """测试HMMRegimeDetector的默认值"""
        # 当使用默认配置创建HMMRegimeDetector时
        detector = HMMRegimeDetector()

        # 应该使用配置的默认值
        assert detector.n_states == 3
        assert detector.model.n_components == 3
        assert detector.model.covariance_type == "diag"
        assert detector.model.n_iter == 300

    def test_hmm_regime_detector_fit_with_config(self):
        """测试使用配置参数进行模型训练"""
        # 当使用配置对象训练模型时
        config = HMMConfig(n_states=3, covariance_type="diag", n_iter=100)
        detector = HMMRegimeDetector(config=config)

        # 创建测试数据
        np.random.seed(42)
        X = np.random.randn(100, 3)  # 100个样本，3个特征

        # 训练模型
        detector.fit(X)

        # 验证模型已训练
        assert detector.is_fitted is True
        assert detector.model.transmat_.shape == (3, 3)
        assert detector.model.means_.shape == (3, 3)

    def test_hmm_regime_detector_predict_with_config(self):
        """测试使用配置参数进行预测"""
        # 当使用配置对象进行预测时
        config = HMMConfig(n_states=3, min_duration=15, stickiness=8.0)
        detector = HMMRegimeDetector(config=config)

        # 创建测试数据并训练模型
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        X_test = np.random.randn(50, 3)

        detector.fit(X_train)

        # 进行预测
        states, proba = detector.predict(X_test)

        # 验证预测结果
        assert len(states) == 50
        assert proba.shape == (50, 3)
        assert np.all(states >= 0) and np.all(states < 3)

    def test_config_validation_in_hmm_detector(self):
        """测试HMMRegimeDetector中的配置验证"""
        # 当使用无效配置时
        config = HMMConfig(n_states=1)  # 状态数太少
        detector = HMMRegimeDetector(config=config)

        # 配置验证应该失败
        assert detector.config.validate() is False
        assert len(detector.config.validation_errors) > 0

    def test_config_persistence_in_hmm_detector(self):
        """测试配置对象在HMMRegimeDetector中的持久性"""
        # 当修改配置对象时
        config = HMMConfig()
        detector = HMMRegimeDetector(config=config)

        # 修改配置对象
        config.n_states = 4

        # HMMRegimeDetector中的配置应该同步更新
        assert detector.config.n_states == 4
        # 注意：模型参数需要重新初始化才能生效
        assert detector.n_states == 4


class TestMarketStateAnalyzerWithConfig:
    """测试使用配置模块的MarketStateAnalyzer"""

    def test_market_state_analyzer_with_config_object(self):
        """测试使用配置对象初始化MarketStateAnalyzer"""
        # 当使用配置对象创建MarketStateAnalyzer时
        config = HMMConfig()
        analyzer = MarketStateAnalyzer(config=config)

        # 应该使用配置对象
        assert analyzer.config == config

    def test_market_state_analyzer_without_config(self):
        """测试不使用配置对象初始化MarketStateAnalyzer（向后兼容）"""
        # 当不使用配置对象创建MarketStateAnalyzer时
        analyzer = MarketStateAnalyzer()

        # 应该创建默认配置
        assert isinstance(analyzer.config, HMMConfig)

    def test_analyze_regime_performance_with_config(self):
        """测试使用配置分析市场状态性能"""
        # 当使用配置分析市场状态时
        config = HMMConfig()
        analyzer = MarketStateAnalyzer(config=config)

        # 创建测试数据
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "log_ret": np.random.normal(0, 0.02, 100),
                "VOL": np.random.normal(1, 0.1, 100),
                "PX": np.random.normal(10, 1, 100),
            },
            index=dates,
        )

        states = np.random.randint(0, 3, 100)
        state_labels = {0: "Bull", 1: "Neutral", 2: "Bear"}

        # 分析性能
        result = analyzer.analyze_regime_performance(data, states, state_labels)

        # 验证分析结果
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "regime" in result.columns
        assert "mean_return" in result.columns

    def test_calculate_transition_matrix_with_config(self):
        """测试使用配置计算状态转换矩阵"""
        # 当使用配置计算转换矩阵时
        config = HMMConfig()
        analyzer = MarketStateAnalyzer(config=config)

        # 创建测试数据
        states = np.array([0, 0, 1, 1, 2, 2, 0, 0])
        state_labels = {0: "Bull", 1: "Neutral", 2: "Bear"}

        # 计算转换矩阵
        result = analyzer.calculate_transition_matrix(states, state_labels)

        # 验证转换矩阵
        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in ["Bull", "Neutral", "Bear"])
        assert all(idx in result.index for idx in ["Bull", "Neutral", "Bear"])


class TestHMMBackwardCompatibility:
    """测试HMM模型的向后兼容性"""

    def test_backward_compatibility_n_states(self):
        """测试状态数的向后兼容性"""
        # 当使用旧的初始化方式时
        detector = HMMRegimeDetector(n_states=4)

        # 应该仍然工作，并创建默认配置
        assert detector.n_states == 4
        assert isinstance(detector.config, HMMConfig)

    def test_backward_compatibility_covariance_type(self):
        """测试协方差类型的向后兼容性"""
        # 当使用旧的初始化方式时
        detector = HMMRegimeDetector(covariance_type="full")

        # 应该仍然工作
        assert detector.model.covariance_type == "full"
        assert isinstance(detector.config, HMMConfig)

    def test_mixed_initialization_hmm(self):
        """测试混合初始化方式"""
        # 当同时提供配置对象和直接参数时
        config = HMMConfig(n_states=4)
        detector = HMMRegimeDetector(config=config, n_states=5)

        # 应该优先使用配置对象
        assert detector.n_states == 4  # 来自配置对象
        assert detector.config.n_states == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
