"""
性能评估测试
"""

import pytest
import numpy as np
import pandas as pd
from src.models.performance_evaluator import PerformanceEvaluator


class TestPerformanceEvaluator:
    """性能评估器测试类"""

    def setup_method(self):
        """测试初始化"""
        # 创建模拟交易信号数据
        np.random.seed(42)
        n_samples = 252  # 一年的交易日
        dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="B")

        # 创建模拟价格和收益率数据
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_samples)))
        log_ret = np.log(prices[1:]) - np.log(prices[:-1])
        log_ret = np.insert(log_ret, 0, 0)  # 添加初始值

        # 创建模拟交易信号
        position = np.random.choice([-1, 0, 1], size=n_samples, p=[0.2, 0.4, 0.4])
        strat_ret = position * log_ret

        self.data = pd.DataFrame(
            {
                "PX": prices,
                "log_ret": log_ret,
                "position": position,
                "strat_ret": strat_ret,
            },
            index=dates,
        )

    def test_initialization(self):
        """测试初始化"""
        evaluator = PerformanceEvaluator()
        assert evaluator is not None

    def test_calculate_metrics(self):
        """测试指标计算"""
        evaluator = PerformanceEvaluator()

        # 计算性能指标
        metrics = evaluator.calculate_metrics(self.data)

        # 验证指标计算结果
        assert isinstance(metrics, dict)
        assert "cagr" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "sortino_ratio" in metrics
        assert "information_ratio" in metrics

    def test_calculate_metrics_with_benchmark(self):
        """测试带基准的指标计算"""
        evaluator = PerformanceEvaluator()

        # 创建基准收益率
        benchmark_ret = np.random.normal(0.0005, 0.015, len(self.data))
        self.data["benchmark_ret"] = benchmark_ret

        # 计算性能指标
        metrics = evaluator.calculate_metrics(self.data, benchmark_col="benchmark_ret")

        # 验证指标计算结果
        assert isinstance(metrics, dict)
        assert "cagr" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "information_ratio" in metrics

    def test_calculate_drawdown(self):
        """测试最大回撤计算"""
        evaluator = PerformanceEvaluator()

        # 计算累计收益
        cum_ret = np.exp(self.data["strat_ret"].cumsum())

        # 计算最大回撤
        drawdown = evaluator.calculate_drawdown(cum_ret)

        # 验证最大回撤计算结果
        assert isinstance(drawdown, pd.Series)
        assert not drawdown.isnull().any()
        assert (drawdown <= 0).all()  # 回撤应小于等于0

    def test_calculate_max_drawdown(self):
        """测试最大回撤计算"""
        evaluator = PerformanceEvaluator()

        # 计算累计收益
        cum_ret = np.exp(self.data["strat_ret"].cumsum())

        # 计算最大回撤
        max_dd = evaluator.calculate_max_drawdown(cum_ret)

        # 验证最大回撤计算结果
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # 最大回撤应小于等于0

    def test_evaluate_model_performance(self):
        """测试模型性能评估"""
        evaluator = PerformanceEvaluator()

        # 评估模型性能
        result = evaluator.evaluate_model_performance(self.data)

        # 验证评估结果
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "cumulative_return" in result
        assert "drawdown" in result
        assert "benchmark_comparison" in result

    def test_compare_models(self):
        """测试模型比较"""
        evaluator = PerformanceEvaluator()

        # 创建两个模型的模拟数据
        model1_data = self.data.copy()
        model2_data = self.data.copy()
        model2_data["strat_ret"] = model2_data["strat_ret"] * 1.1  # 假设模型2表现更好

        # 比较模型性能
        result = evaluator.compare_models(
            {"model1": model1_data, "model2": model2_data}
        )

        # 验证比较结果
        assert isinstance(result, dict)
        assert "model_results" in result
        assert "model1" in result["model_results"]
        assert "model2" in result["model_results"]
        assert "best_model" in result
