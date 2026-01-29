"""
HMM参数优化测试
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.models.hmm_optimizer import HMMOptimizer, GridSearchOptimizer
from src.models.hmm_regime_detector import HMMRegimeDetector


class TestHMMOptimizer:
    """HMM优化器测试类"""

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

    def test_initialization(self):
        """测试初始化"""
        optimizer = HMMOptimizer()
        assert optimizer is not None

    def test_optimize_parameters(self):
        """测试参数优化"""
        optimizer = HMMOptimizer()

        # 定义参数搜索空间
        param_grid = {
            "n_states": [2, 3],
            "covariance_type": ["diag", "full"],
            "n_iter": [100, 200],
        }

        # 执行优化
        result = optimizer.optimize_parameters(self.X, param_grid, n_splits=2)

        # 验证结果结构
        assert isinstance(result, dict)
        assert "best_params" in result
        assert "best_score" in result
        assert "cv_results" in result

    def test_feature_selection(self):
        """测试特征选择"""
        optimizer = HMMOptimizer()

        # 定义特征列表
        feature_columns = ["log_ret", "VOL", "SPREAD", "EBS", "BUFFETT"]

        # 执行特征选择
        result = optimizer.optimize_features(self.data, feature_columns, n_states=3)

        # 验证结果结构
        assert isinstance(result, dict)
        assert "best_features" in result
        assert "best_score" in result
        assert "feature_importance" in result


class TestGridSearchOptimizer:
    """网格搜索优化器测试类"""

    def setup_method(self):
        """测试初始化"""
        # 创建模拟特征数据
        np.random.seed(42)
        n_samples = 300
        n_features = 5

        self.X = np.random.normal(0, 1, (n_samples, n_features))

    def test_initialization(self):
        """测试初始化"""
        optimizer = GridSearchOptimizer()
        assert optimizer is not None

    def test_grid_search(self):
        """测试网格搜索"""
        optimizer = GridSearchOptimizer()

        # 定义参数网格
        param_grid = {"n_states": [2, 3], "covariance_type": ["diag"]}

        # 执行网格搜索
        result = optimizer.grid_search(self.X, param_grid, n_splits=2)

        # 验证结果结构
        assert isinstance(result, dict)
        assert "best_params" in result
        assert "best_score" in result
        assert "cv_results" in result

    def test_evaluate_model(self):
        """测试模型评估"""
        optimizer = GridSearchOptimizer()

        # 创建一个简单的HMM模型
        detector = HMMRegimeDetector(n_states=3)
        detector.fit(self.X)

        # 评估模型
        score = optimizer.evaluate_model(detector, self.X)

        # 验证评分结果
        assert isinstance(score, float)
        assert score is not None
