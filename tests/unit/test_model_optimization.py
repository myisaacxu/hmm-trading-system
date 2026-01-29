"""
模型优化和选择功能测试
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.models.hmm_optimizer import HMMOptimizer
from src.models.hmm_regime_detector import HMMRegimeDetector
from src.models.model_manager import ModelManager


class TestModelOptimization:
    """模型优化测试类"""

    def setup_method(self):
        """测试初始化"""
        # 创建临时目录作为模型目录
        import tempfile
        import shutil

        self.temp_dir = tempfile.mkdtemp()

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

        # 创建优化器和模型管理器实例
        self.optimizer = HMMOptimizer()
        self.model_manager = ModelManager(models_dir=self.temp_dir)

    def test_model_optimization_with_different_state_counts(self):
        """测试不同状态数量的模型优化"""
        # 定义不同状态数量的参数网格
        param_grid = {
            "n_states": [2, 3, 4],
            "covariance_type": ["diag"],
            "n_iter": [100],
        }

        # 执行优化
        optimization_result = self.optimizer.optimize_parameters(
            self.X, param_grid, n_splits=2
        )

        # 验证优化结果
        assert isinstance(optimization_result, dict)
        assert "best_params" in optimization_result
        assert "best_score" in optimization_result
        assert "cv_results" in optimization_result

        # 验证最佳参数包含状态数量
        best_params = optimization_result["best_params"]
        assert "n_states" in best_params
        assert best_params["n_states"] in [2, 3, 4]

        # 验证交叉验证结果包含所有参数组合
        cv_results = optimization_result["cv_results"]
        assert len(cv_results) >= len(param_grid["n_states"])

    def test_model_optimization_with_different_covariance_types(self):
        """测试不同协方差类型的模型优化"""
        # 定义不同协方差类型的参数网格
        param_grid = {
            "n_states": [3],
            "covariance_type": ["diag", "full"],
            "n_iter": [100],
        }

        # 执行优化
        optimization_result = self.optimizer.optimize_parameters(
            self.X, param_grid, n_splits=2
        )

        # 验证优化结果
        assert isinstance(optimization_result, dict)
        assert "best_params" in optimization_result
        assert "best_score" in optimization_result
        assert "cv_results" in optimization_result

        # 验证最佳参数包含协方差类型
        best_params = optimization_result["best_params"]
        assert "covariance_type" in best_params
        assert best_params["covariance_type"] in ["diag", "full"]

    def test_model_optimization_with_feature_selection(self):
        """测试带特征选择的模型优化"""
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

        # 使用最佳特征进行参数优化
        best_features = feature_result["best_features"]
        X_best = self.data[best_features].values

        param_grid = {"n_states": [2, 3], "covariance_type": ["diag"], "n_iter": [100]}

        optimization_result = self.optimizer.optimize_parameters(
            X_best, param_grid, n_splits=2
        )

        # 验证优化结果
        assert isinstance(optimization_result, dict)
        assert "best_params" in optimization_result
        assert "best_score" in optimization_result

    def test_model_optimization_with_smoothing_parameters(self):
        """测试带平滑参数的模型优化"""
        # 首先优化基本参数
        param_grid = {"n_states": [3], "covariance_type": ["diag"], "n_iter": [100]}

        optimization_result = self.optimizer.optimize_parameters(
            self.X, param_grid, n_splits=2
        )

        best_params = optimization_result["best_params"]

        # 然后优化平滑参数
        smoothing_result = self.optimizer.optimize_smoothing_parameters(
            self.X, n_states=best_params.get("n_states", 3)
        )

        # 验证平滑参数优化结果
        assert isinstance(smoothing_result, dict)
        assert "best_params" in smoothing_result
        assert "best_score" in smoothing_result

        # 验证平滑参数包含最小持续时间和粘性强度
        best_smoothing_params = smoothing_result["best_params"]
        assert "min_duration" in best_smoothing_params
        assert "sticky_strength" in best_smoothing_params
        assert best_smoothing_params["min_duration"] > 0
        assert best_smoothing_params["sticky_strength"] >= 0

    def test_model_selection_based_on_performance(self):
        """测试基于性能的模型选择"""
        # 训练多个不同参数的模型并保存
        model_names = []

        # 训练不同状态数量的模型
        for n_states in [2, 3, 4]:
            detector = HMMRegimeDetector(
                n_states=n_states,
                covariance_type="diag",
                n_iter=100,
            )
            detector.fit(self.X)

            # 生成不同的性能指标
            performance_metrics = {
                "cagr": 0.1 + (n_states - 2) * 0.02,
                "sharpe": 0.8 + (n_states - 2) * 0.1,
                "mdd": -0.12 + (n_states - 2) * 0.01,
                "params": {
                    "n_states": n_states,
                    "covariance_type": "diag",
                    "n_iter": 100,
                },
            }

            # 保存模型
            model_name = self.model_manager.save_model(detector, performance_metrics)
            model_names.append(model_name)

        # 获取保存的模型列表
        saved_models = self.model_manager.list_saved_models()

        # 验证模型按夏普比率降序排序
        assert len(saved_models) >= len(model_names)
        if len(saved_models) > 0:
            # 验证模型按夏普比率降序排序
            sharpe_values = [model["sharpe"] for model in saved_models]
            assert all(
                sharpe_values[i] >= sharpe_values[i + 1]
                for i in range(len(sharpe_values) - 1)
            )

            # 获取最佳模型
            best_model = saved_models[0]
            assert "name" in best_model
            assert "sharpe" in best_model
            assert "cagr" in best_model
            assert "mdd" in best_model

        # 清理测试模型
        for model_name in model_names:
            self.model_manager.delete_model(model_name)

    def test_model_comparison_functionality(self):
        """测试模型比较功能"""
        # 训练并保存一个基准模型
        detector = HMMRegimeDetector(
            n_states=3,
            covariance_type="diag",
            n_iter=100,
        )
        detector.fit(self.X)

        # 生成基准性能指标
        baseline_metrics = {
            "cagr": 0.12,
            "sharpe": 1.0,
            "mdd": -0.1,
            "params": {"n_states": 3, "covariance_type": "diag", "n_iter": 100},
        }

        # 保存基准模型
        baseline_model_name = self.model_manager.save_model(detector, baseline_metrics)

        # 测试新模型性能优于基准
        better_metrics = {
            "cagr": 0.15,
            "sharpe": 1.3,
            "mdd": -0.08,
            "params": {"n_states": 3, "covariance_type": "diag", "n_iter": 100},
        }

        comparison_result = self.model_manager.compare_models(
            better_metrics, threshold=0.1
        )

        # 验证比较结果
        assert isinstance(comparison_result, dict)
        assert "has_better_model" in comparison_result
        assert "best_model_sharpe" in comparison_result
        assert "current_sharpe" in comparison_result
        assert "improvement" in comparison_result
        assert "should_save" in comparison_result

        # 验证应该保存更好的模型
        assert comparison_result["should_save"] is True
        assert comparison_result["improvement"] > 0

        # 测试新模型性能不如基准
        worse_metrics = {
            "cagr": 0.09,
            "sharpe": 0.7,
            "mdd": -0.15,
            "params": {"n_states": 3, "covariance_type": "diag", "n_iter": 100},
        }

        comparison_result_worse = self.model_manager.compare_models(
            worse_metrics, threshold=0.1
        )

        # 验证不应该保存更差的模型
        assert comparison_result_worse["should_save"] is False
        assert comparison_result_worse["improvement"] < 0

        # 清理测试模型
        self.model_manager.delete_model(baseline_model_name)

    def test_optimization_workflow_with_feature_selection_and_smoothing(self):
        """测试完整的优化工作流程，包括特征选择和平滑参数优化"""
        # 1. 特征选择
        feature_columns = ["log_ret", "VOL", "SPREAD", "EBS", "BUFFETT"]
        feature_result = self.optimizer.optimize_features(
            self.data, feature_columns, n_states=3
        )
        best_features = feature_result["best_features"]
        X_best = self.data[best_features].values

        # 2. 基本参数优化
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

        # 5. 使用最佳平滑参数预测
        states, proba = detector.predict(
            X_best,
            min_len=best_smoothing_params["min_duration"],
            sticky_strength=best_smoothing_params["sticky_strength"],
        )

        # 6. 验证预测结果
        assert len(states) == len(X_best)
        assert proba.shape == (len(X_best), best_params.get("n_states", 3))

        # 7. 生成综合性能指标
        combined_params = {
            **best_params,
            **best_smoothing_params,
            "features": best_features,
        }

        performance_metrics = {
            "cagr": 0.16,
            "sharpe": 1.4,
            "mdd": -0.09,
            "params": combined_params,
        }

        # 8. 保存最终模型
        model_name = self.model_manager.save_model(detector, performance_metrics)

        # 9. 验证模型保存成功
        assert isinstance(model_name, str)
        assert model_name

        # 10. 清理测试模型
        self.model_manager.delete_model(model_name)

    def teardown_method(self):
        """测试清理"""
        # 清理临时目录
        import shutil

        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
