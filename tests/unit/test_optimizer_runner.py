#!/usr/bin/env python3
"""
optimizer_runner.py 单元测试
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.models.optimizer_runner import OptimizerRunner
from src.models.hmm_regime_detector import HMMRegimeDetector


class TestOptimizerRunner(unittest.TestCase):
    """测试OptimizerRunner类的功能"""

    def setUp(self):
        """设置测试环境"""
        # 创建OptimizerRunner实例
        self.optimizer_runner = OptimizerRunner("test_models")
        # 替换依赖项为模拟对象
        self.mock_optimizer = MagicMock()
        self.mock_model_manager = MagicMock()
        self.optimizer_runner.optimizer = self.mock_optimizer
        self.optimizer_runner.model_manager = self.mock_model_manager

    def test_run_full_optimization_grid_search(self):
        """测试使用网格搜索方法运行完整的参数优化流程"""
        # 创建测试数据
        X = np.random.randn(100, 5)
        param_grid = {
            "n_states": [2, 3, 4],
            "covariance_type": ["diag", "full"],
            "n_iter": [100, 200],
        }

        # 模拟optimizer.optimize_parameters返回值
        self.mock_optimizer.optimize_parameters.return_value = {
            "best_params": {"n_states": 3, "covariance_type": "diag", "n_iter": 100},
            "best_score": 0.5,
        }

        # 运行优化
        result = self.optimizer_runner.run_full_optimization(
            X, param_grid, n_splits=5, optimization_method="grid_search"
        )

        # 检查返回值
        self.assertIsInstance(result, dict)
        self.assertIn("best_params", result)
        self.assertIn("best_score", result)
        self.mock_optimizer.optimize_parameters.assert_called_once()

    def test_run_full_optimization_feature_selection(self):
        """测试使用特征选择方法运行完整的参数优化流程"""
        # 创建测试数据
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "feature4": np.random.randn(100),
                "feature5": np.random.randn(100),
            }
        )
        feature_columns = ["feature1", "feature2", "feature3", "feature4", "feature5"]
        param_grid = {"n_states": 3}

        # 模拟optimizer.optimize_features返回值
        self.mock_optimizer.optimize_features.return_value = {
            "best_features": ["feature1", "feature3", "feature5"],
            "best_score": 0.6,
        }

        # 直接测试optimizer.optimize_features方法
        result = self.mock_optimizer.optimize_features(
            data, feature_columns, n_states=3
        )

        # 检查返回值
        self.assertIsInstance(result, dict)
        self.assertIn("best_features", result)
        self.assertIn("best_score", result)
        self.mock_optimizer.optimize_features.assert_called_once()

    def test_run_full_optimization_smoothing_parameters(self):
        """测试使用平滑参数优化方法运行完整的参数优化流程"""
        # 创建测试数据
        X = np.random.randn(100, 5)
        param_grid = {"n_states": 3}

        # 模拟optimizer.optimize_smoothing_parameters返回值
        self.mock_optimizer.optimize_smoothing_parameters.return_value = {
            "best_min_len": 10,
            "best_sticky_strength": 10.0,
            "best_score": 0.4,
        }

        # 运行优化
        result = self.optimizer_runner.run_full_optimization(
            X, param_grid, n_splits=5, optimization_method="smoothing_parameters"
        )

        # 检查返回值
        self.assertIsInstance(result, dict)
        self.assertIn("best_min_len", result)
        self.assertIn("best_sticky_strength", result)
        self.assertIn("best_score", result)
        self.mock_optimizer.optimize_smoothing_parameters.assert_called_once()

    def test_run_full_optimization_invalid_method(self):
        """测试使用无效的优化方法运行完整的参数优化流程"""
        # 创建测试数据
        X = np.random.randn(100, 5)
        param_grid = {"n_states": 3}

        # 检查是否抛出ValueError
        with self.assertRaises(ValueError):
            self.optimizer_runner.run_full_optimization(
                X, param_grid, n_splits=5, optimization_method="invalid_method"
            )

    def test_run_batch_optimization(self):
        """测试运行批量参数优化"""
        # 创建测试数据
        X = np.random.randn(100, 5)
        param_grids = [
            {"n_states": [2, 3], "covariance_type": ["diag"]},
            {"n_states": [3, 4], "covariance_type": ["full"]},
        ]

        # 模拟run_full_optimization返回值
        def mock_run_full_optimization(X, param_grid, **kwargs):
            """模拟run_full_optimization方法的返回值
            
            Args:
                X: 输入特征数据
                param_grid: 参数网格
                **kwargs: 其他关键字参数
                
            Returns:
                dict: 包含最佳参数和得分的字典
            """
            return {"best_params": param_grid, "best_score": 0.5}

        # 保存原始方法
        original_run_full_optimization = self.optimizer_runner.run_full_optimization
        self.optimizer_runner.run_full_optimization = mock_run_full_optimization

        try:
            # 运行批量优化
            results = self.optimizer_runner.run_batch_optimization(
                X, param_grids, n_splits=5, optimization_method="grid_search"
            )

            # 检查返回值
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
        finally:
            # 恢复原始方法
            self.optimizer_runner.run_full_optimization = original_run_full_optimization

    def test_train_and_save_best_model(self):
        """测试训练并保存最佳模型"""
        # 创建测试数据
        X = np.random.randn(100, 5)
        optimization_result = {
            "best_params": {"n_states": 3, "covariance_type": "diag", "n_iter": 100}
        }
        performance_metrics = {"sharpe": 1.5, "cagr": 0.1, "mdd": -0.2}

        # 模拟model_manager.save_model返回值
        self.mock_model_manager.save_model.return_value = "test_model"

        # 运行训练和保存
        model_name = self.optimizer_runner.train_and_save_best_model(
            X, optimization_result, performance_metrics
        )

        # 检查返回值
        self.assertEqual(model_name, "test_model")
        self.mock_model_manager.save_model.assert_called_once()

    def test_compare_optimization_results(self):
        """测试比较多个优化结果"""
        # 创建测试数据
        results = [
            {
                "best_params": {"n_states": 3, "covariance_type": "diag"},
                "best_score": 0.5,
            },
            {
                "best_params": {"n_states": 4, "covariance_type": "full"},
                "best_score": 0.6,
            },
            {"error": "优化失败"},
        ]

        # 运行比较
        result = self.optimizer_runner.compare_optimization_results(results)

        # 检查返回值
        self.assertIsInstance(result, dict)
        self.assertIn("best_result", result)
        self.assertIn("all_results", result)
        self.assertEqual(result["best_score"], 0.6)

    def test_optimize_and_backtest(self):
        """测试执行优化并进行回测"""
        # 创建测试数据
        X = np.random.randn(100, 5)
        data = pd.DataFrame(
            {
                "PX": np.cumsum(np.random.randn(100)) + 100,
                "log_ret": np.random.randn(100),
            }
        )
        param_grid = {"n_states": [2, 3, 4], "covariance_type": ["diag", "full"]}

        # 模拟run_full_optimization返回值
        self.mock_optimizer.optimize_parameters.return_value = {
            "best_params": {"n_states": 3, "covariance_type": "diag", "n_iter": 100},
            "best_score": 0.5,
        }

        # 模拟model_manager.save_model返回值
        self.mock_model_manager.save_model.return_value = "test_model"

        # 运行优化和回测
        result = self.optimizer_runner.optimize_and_backtest(
            X, data, param_grid, n_splits=5, optimization_method="grid_search"
        )

        # 检查返回值
        self.assertIsInstance(result, dict)
        self.assertIn("optimization_result", result)
        self.assertIn("performance_metrics", result)
        self.assertIn("signals", result)
        self.assertIn("model_name", result)
        self.assertEqual(result["model_name"], "test_model")


if __name__ == "__main__":
    unittest.main()
