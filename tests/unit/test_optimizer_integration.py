"""
优化参数模块集成测试
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.models.hmm_optimizer import HMMOptimizer, GridSearchOptimizer
from src.models.hmm_regime_detector import HMMRegimeDetector
from src.models.model_manager import ModelManager
from src.models.cross_validation import TimeSeriesCrossValidator


class TestOptimizerIntegration:
    """优化器集成测试类"""

    def setup_method(self):
        """测试初始化"""
        # 创建模拟特征数据
        np.random.seed(42)
        n_samples = 500
        n_features = 5

        # 创建三个不同状态的数据（高、中、低波动）
        self.X = np.zeros((n_samples, n_features))

        # 状态1：高波动，正收益
        self.X[:150] = np.random.normal(0.1, 0.5, (150, n_features))

        # 状态2：中波动，中性收益
        self.X[150:350] = np.random.normal(0.0, 0.2, (200, n_features))

        # 状态3：低波动，负收益
        self.X[350:] = np.random.normal(-0.05, 0.1, (150, n_features))

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

        # 创建优化器实例
        self.optimizer = HMMOptimizer()
        self.model_manager = ModelManager()

    def test_optimizer_and_model_manager_integration(self):
        """测试优化器和模型管理器的集成"""
        # 定义参数搜索空间
        param_grid = {"n_states": [2, 3], "covariance_type": ["diag"], "n_iter": [100]}

        # 执行优化
        optimization_result = self.optimizer.optimize_parameters(
            self.X, param_grid, n_splits=2
        )

        # 验证优化结果
        assert isinstance(optimization_result, dict)
        assert "best_params" in optimization_result
        assert "best_score" in optimization_result
        assert "cv_results" in optimization_result

        # 使用最佳参数训练模型
        best_params = optimization_result["best_params"]
        detector = HMMRegimeDetector(
            n_states=best_params.get("n_states", 3),
            covariance_type=best_params.get("covariance_type", "diag"),
            n_iter=best_params.get("n_iter", 100),
        )
        detector.fit(self.X)

        # 预测状态
        states, proba = detector.predict(self.X)

        # 生成性能指标
        performance_metrics = {
            "cagr": 0.15,
            "sharpe": 1.2,
            "mdd": -0.1,
            "params": best_params,
        }

        # 保存模型
        model_name = self.model_manager.save_model(detector, performance_metrics)

        # 验证模型保存成功
        assert isinstance(model_name, str)
        assert model_name

        # 加载模型
        loaded_model, loaded_metrics = self.model_manager.load_model(model_name)

        # 验证模型加载成功
        assert loaded_model is not None
        assert loaded_metrics is not None
        assert loaded_metrics["params"] == best_params

        # 清理测试模型
        self.model_manager.delete_model(model_name)

    def test_feature_optimization_integration(self):
        """测试特征优化集成"""
        # 定义特征列表
        feature_columns = ["log_ret", "VOL", "SPREAD", "EBS", "BUFFETT"]

        # 执行特征选择
        feature_result = self.optimizer.optimize_features(
            self.data, feature_columns, n_states=3
        )

        # 验证特征选择结果
        assert isinstance(feature_result, dict)
        assert "best_features" in feature_result
        assert "best_score" in feature_result
        assert "feature_importance" in feature_result
        assert "feature_scores" in feature_result

        # 使用最佳特征训练模型
        best_features = feature_result["best_features"]
        assert best_features

        # 提取最佳特征
        X_best = self.data[best_features].values

        # 训练模型
        detector = HMMRegimeDetector(n_states=3)
        detector.fit(X_best)

        # 验证模型训练成功
        states, proba = detector.predict(X_best)
        assert len(states) == len(X_best)

    def test_smoothing_parameters_optimization_integration(self):
        """测试平滑参数优化集成"""
        # 执行平滑参数优化
        smoothing_result = self.optimizer.optimize_smoothing_parameters(
            self.X, n_states=3
        )

        # 验证平滑参数优化结果
        assert isinstance(smoothing_result, dict)
        assert "best_params" in smoothing_result
        assert "best_score" in smoothing_result

        # 使用最佳平滑参数
        best_smoothing_params = smoothing_result["best_params"]
        assert best_smoothing_params
        assert "min_duration" in best_smoothing_params
        assert "sticky_strength" in best_smoothing_params

        # 训练模型
        detector = HMMRegimeDetector(n_states=3)
        detector.fit(self.X)

        # 使用最佳平滑参数预测
        states, proba = detector.predict(
            self.X,
            min_len=best_smoothing_params["min_duration"],
            sticky_strength=best_smoothing_params["sticky_strength"],
        )

        # 验证预测结果
        assert len(states) == len(self.X)
        assert proba.shape == (len(self.X), 3)

    def test_full_optimization_workflow(self):
        """测试完整的优化工作流程"""
        # 1. 特征选择
        feature_columns = ["log_ret", "VOL", "SPREAD", "EBS", "BUFFETT"]
        feature_result = self.optimizer.optimize_features(
            self.data, feature_columns, n_states=3
        )
        best_features = feature_result["best_features"]
        X_best = self.data[best_features].values

        # 2. 参数优化
        param_grid = {"n_states": [2, 3], "covariance_type": ["diag"], "n_iter": [100]}
        param_result = self.optimizer.optimize_parameters(
            X_best, param_grid, n_splits=2
        )
        best_params = param_result["best_params"]

        # 3. 平滑参数优化
        smoothing_result = self.optimizer.optimize_smoothing_parameters(
            X_best, n_states=best_params.get("n_states", 3)
        )
        best_smoothing_params = smoothing_result["best_params"]

        # 4. 训练最终模型
        detector = HMMRegimeDetector(
            n_states=best_params.get("n_states", 3),
            covariance_type=best_params.get("covariance_type", "diag"),
            n_iter=best_params.get("n_iter", 100),
        )
        detector.fit(X_best)

        # 5. 使用最佳参数预测
        states, proba = detector.predict(
            X_best,
            min_len=best_smoothing_params["min_duration"],
            sticky_strength=best_smoothing_params["sticky_strength"],
        )

        # 6. 生成综合性能指标
        combined_params = {
            **best_params,
            **best_smoothing_params,
            "features": best_features,
        }

        performance_metrics = {
            "cagr": 0.18,
            "sharpe": 1.5,
            "mdd": -0.08,
            "params": combined_params,
        }

        # 7. 保存最终模型
        model_name = self.model_manager.save_model(detector, performance_metrics)

        # 8. 验证完整流程
        assert isinstance(model_name, str)
        assert model_name

        # 9. 清理测试模型
        self.model_manager.delete_model(model_name)
