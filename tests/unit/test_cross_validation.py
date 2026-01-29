"""
交叉验证测试
"""

import pytest
import numpy as np
import pandas as pd
from src.models.cross_validation import TimeSeriesCrossValidator, RollingWindowValidator


class TestTimeSeriesCrossValidator:
    """时间序列交叉验证测试类"""

    def setup_method(self):
        """测试初始化"""
        # 创建模拟特征数据
        np.random.seed(42)
        n_samples = 500
        n_features = 5

        self.X = np.random.normal(0, 1, (n_samples, n_features))
        self.y = np.random.normal(0, 1, n_samples)

    def test_initialization(self):
        """测试初始化"""
        validator = TimeSeriesCrossValidator(n_splits=5)
        assert validator is not None
        assert validator.n_splits == 5

    def test_split(self):
        """测试数据分割"""
        validator = TimeSeriesCrossValidator(n_splits=3)

        # 执行分割
        splits = list(validator.split(self.X))

        # 验证分割结果
        assert len(splits) == 3

        for train_idx, test_idx in splits:
            # 训练集大小应小于测试集起始索引
            assert max(train_idx) < min(test_idx)
            # 确保索引连续
            assert np.array_equal(train_idx, np.arange(len(train_idx)))
            assert np.array_equal(
                test_idx, np.arange(len(train_idx), len(train_idx) + len(test_idx))
            )

    def test_split_with_min_train_size(self):
        """测试带最小训练集大小的分割"""
        validator = TimeSeriesCrossValidator(n_splits=3, min_train_size=100)

        # 执行分割
        splits = list(validator.split(self.X))

        # 验证分割结果
        assert len(splits) == 3

        for train_idx, test_idx in splits:
            # 训练集大小应至少为min_train_size
            assert len(train_idx) >= 100
            # 训练集大小应小于测试集起始索引
            assert max(train_idx) < min(test_idx)


class TestRollingWindowValidator:
    """滚动窗口交叉验证测试类"""

    def setup_method(self):
        """测试初始化"""
        # 创建模拟特征数据
        np.random.seed(42)
        n_samples = 500
        n_features = 5

        self.X = np.random.normal(0, 1, (n_samples, n_features))
        self.y = np.random.normal(0, 1, n_samples)

    def test_initialization(self):
        """测试初始化"""
        validator = RollingWindowValidator(
            window_size=200, step_size=100, test_size=100
        )
        assert validator is not None
        assert validator.window_size == 200
        assert validator.step_size == 100
        assert validator.test_size == 100

    def test_split(self):
        """测试滚动窗口分割"""
        validator = RollingWindowValidator(
            window_size=200, step_size=100, test_size=100
        )

        # 执行分割
        splits = list(validator.split(self.X))

        # 验证分割结果
        assert len(splits) > 0

        for i, (train_idx, test_idx) in enumerate(splits):
            # 训练集大小应等于window_size
            assert len(train_idx) == 200
            # 测试集大小应等于test_size
            assert len(test_idx) == 100
            # 训练集和测试集应连续
            assert max(train_idx) + 1 == min(test_idx)

    def test_split_with_large_window(self):
        """测试大窗口大小的分割"""
        # 窗口大小大于数据长度
        validator = RollingWindowValidator(
            window_size=600, step_size=100, test_size=100
        )

        # 执行分割
        splits = list(validator.split(self.X))

        # 验证分割结果
        assert len(splits) == 0  # 应返回空列表，因为窗口大小大于数据长度

    def test_split_with_invalid_parameters(self):
        """测试无效参数的分割"""
        # 测试无效的窗口大小和步长
        validator = RollingWindowValidator(window_size=50, step_size=100, test_size=100)

        # 执行分割
        splits = list(validator.split(self.X))

        # 验证分割结果
        assert len(splits) > 0  # 即使参数无效，也应返回至少一个分割
