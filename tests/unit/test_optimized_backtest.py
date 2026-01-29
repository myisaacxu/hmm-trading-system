"""
使用最优模型进行回测的功能测试
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.models.hmm_optimizer import HMMOptimizer
from src.models.hmm_regime_detector import HMMRegimeDetector
from src.models.model_manager import ModelManager


class TestOptimizedBacktest:
    """使用最优模型进行回测的测试类"""

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

        # 创建优化器和模型管理器实例
        self.optimizer = HMMOptimizer()
        self.model_manager = ModelManager()

    def test_optimized_model_backtest(self):
        """测试使用优化模型进行回测"""
        # 定义参数搜索空间
        param_grid = {"n_states": [2, 3], "covariance_type": ["diag"], "n_iter": [100]}

        # 执行优化
        optimization_result = self.optimizer.optimize_parameters(
            self.X, param_grid, n_splits=2
        )

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

        # 生成交易信号
        signals = self.data.copy()
        signals["state"] = states

        # 映射状态到交易信号
        state_means = (
            signals.groupby("state")["log_ret"].mean().sort_values(ascending=False)
        )
        ranked_states = state_means.index.tolist()
        state_map = {ranked_states[0]: "Bull", ranked_states[-1]: "Bear"}
        for s in set(range(best_params.get("n_states", 3))) - set(state_map.keys()):
            state_map[s] = "Neutral"
        signals["regime"] = signals["state"].map(state_map)

        # 生成交易信号
        signals["position"] = 0
        signals.loc[signals["regime"] == "Bull", "position"] = 1
        signals.loc[signals["regime"] == "Bear", "position"] = -1

        # 计算策略收益率
        signals["log_ret"] = np.log(signals["PX"]).diff().fillna(0.0)
        signals["strat_ret"] = signals["position"] * signals["log_ret"]

        # 计算累积收益率
        signals["cum_bh"] = np.exp(signals["log_ret"].cumsum())
        signals["cum_strat"] = np.exp(signals["strat_ret"].cumsum())

        # 验证回测结果
        assert "cum_bh" in signals.columns
        assert "cum_strat" in signals.columns
        assert len(signals["cum_bh"]) == len(signals)
        assert len(signals["cum_strat"]) == len(signals)

        # 验证策略收益率计算正确
        assert not signals["strat_ret"].isnull().any()
        assert not signals["cum_strat"].isnull().any()

    def test_backtest_with_saved_optimized_model(self):
        """测试使用保存的优化模型进行回测"""
        # 定义参数搜索空间
        param_grid = {"n_states": [3], "covariance_type": ["diag"], "n_iter": [100]}

        # 执行优化
        optimization_result = self.optimizer.optimize_parameters(
            self.X, param_grid, n_splits=2
        )

        # 使用最佳参数训练模型
        best_params = optimization_result["best_params"]
        detector = HMMRegimeDetector(
            n_states=best_params.get("n_states", 3),
            covariance_type=best_params.get("covariance_type", "diag"),
            n_iter=best_params.get("n_iter", 100),
        )
        detector.fit(self.X)

        # 生成性能指标
        performance_metrics = {
            "cagr": 0.14,
            "sharpe": 1.3,
            "mdd": -0.09,
            "params": best_params,
        }

        # 保存模型
        model_name = self.model_manager.save_model(detector, performance_metrics)

        # 加载模型
        loaded_model, loaded_metrics = self.model_manager.load_model(model_name)

        # 验证模型加载成功
        assert loaded_model is not None
        assert loaded_metrics is not None

        # 使用加载的模型进行回测
        loaded_states, loaded_proba = loaded_model.predict(self.X)

        # 生成交易信号
        signals = self.data.copy()
        signals["state"] = loaded_states

        # 映射状态到交易信号
        state_means = (
            signals.groupby("state")["log_ret"].mean().sort_values(ascending=False)
        )
        ranked_states = state_means.index.tolist()
        state_map = {ranked_states[0]: "Bull", ranked_states[-1]: "Bear"}
        for s in set(range(best_params.get("n_states", 3))) - set(state_map.keys()):
            state_map[s] = "Neutral"
        signals["regime"] = signals["state"].map(state_map)

        # 生成交易信号
        signals["position"] = 0
        signals.loc[signals["regime"] == "Bull", "position"] = 1
        signals.loc[signals["regime"] == "Bear", "position"] = -1

        # 计算策略收益率
        signals["log_ret"] = np.log(signals["PX"]).diff().fillna(0.0)
        signals["strat_ret"] = signals["position"] * signals["log_ret"]

        # 计算累积收益率
        signals["cum_bh"] = np.exp(signals["log_ret"].cumsum())
        signals["cum_strat"] = np.exp(signals["strat_ret"].cumsum())

        # 验证回测结果
        assert "cum_bh" in signals.columns
        assert "cum_strat" in signals.columns
        assert len(signals["cum_bh"]) == len(signals)
        assert len(signals["cum_strat"]) == len(signals)

        # 清理测试模型
        self.model_manager.delete_model(model_name)

    def test_backtest_performance_comparison(self):
        """测试回测性能比较"""
        # 训练多个不同参数的模型
        models = []
        model_names = []

        # 训练不同状态数量的模型
        for n_states in [2, 3, 4]:
            detector = HMMRegimeDetector(
                n_states=n_states,
                covariance_type="diag",
                n_iter=100,
            )
            detector.fit(self.X)

            # 预测状态并计算回测结果
            states, proba = detector.predict(self.X)

            # 生成交易信号
            signals = self.data.copy()
            signals["state"] = states

            # 映射状态到交易信号
            state_means = (
                signals.groupby("state")["log_ret"].mean().sort_values(ascending=False)
            )
            ranked_states = state_means.index.tolist()
            state_map = {ranked_states[0]: "Bull", ranked_states[-1]: "Bear"}
            for s in set(range(n_states)) - set(state_map.keys()):
                state_map[s] = "Neutral"
            signals["regime"] = signals["state"].map(state_map)

            # 生成交易信号
            signals["position"] = 0
            signals.loc[signals["regime"] == "Bull", "position"] = 1
            signals.loc[signals["regime"] == "Bear", "position"] = -1

            # 计算策略收益率
            signals["log_ret"] = np.log(signals["PX"]).diff().fillna(0.0)
            signals["strat_ret"] = signals["position"] * signals["log_ret"]

            # 计算性能指标
            days = len(signals)
            years = days / 252

            cagr = (
                (np.exp(signals["strat_ret"].sum()) - 1) ** (1 / years) - 1
                if years > 0
                else 0
            )
            sharpe = (
                (signals["strat_ret"].mean() / signals["strat_ret"].std())
                * np.sqrt(252)
                if signals["strat_ret"].std() > 0
                else 0
            )
            mdd = (
                (signals["cum_strat"].cummax() - signals["cum_strat"]).max()
                if "cum_strat" in signals.columns
                else 0
            )

            performance_metrics = {
                "cagr": cagr,
                "sharpe": sharpe,
                "mdd": mdd,
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

        # 验证模型按夏普比率排序
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

    def test_backtest_with_feature_optimization(self):
        """测试使用特征优化进行回测"""
        # 定义特征列表
        feature_columns = ["log_ret", "VOL", "SPREAD", "EBS", "BUFFETT"]

        # 执行特征选择
        feature_result = self.optimizer.optimize_features(
            self.data, feature_columns, n_states=3
        )

        # 使用最佳特征
        best_features = feature_result["best_features"]
        X_best = self.data[best_features].values

        # 优化模型参数
        param_grid = {"n_states": [2, 3], "covariance_type": ["diag"], "n_iter": [100]}

        optimization_result = self.optimizer.optimize_parameters(
            X_best, param_grid, n_splits=2
        )

        # 使用最佳参数训练模型
        best_params = optimization_result["best_params"]
        detector = HMMRegimeDetector(
            n_states=best_params.get("n_states", 3),
            covariance_type=best_params.get("covariance_type", "diag"),
            n_iter=best_params.get("n_iter", 100),
        )
        detector.fit(X_best)

        # 预测状态
        states, proba = detector.predict(X_best)

        # 生成交易信号
        signals = self.data.copy()
        signals["state"] = states

        # 映射状态到交易信号
        state_means = (
            signals.groupby("state")["log_ret"].mean().sort_values(ascending=False)
        )
        ranked_states = state_means.index.tolist()
        state_map = {ranked_states[0]: "Bull", ranked_states[-1]: "Bear"}
        for s in set(range(best_params.get("n_states", 3))) - set(state_map.keys()):
            state_map[s] = "Neutral"
        signals["regime"] = signals["state"].map(state_map)

        # 生成交易信号
        signals["position"] = 0
        signals.loc[signals["regime"] == "Bull", "position"] = 1
        signals.loc[signals["regime"] == "Bear", "position"] = -1

        # 计算策略收益率
        signals["log_ret"] = np.log(signals["PX"]).diff().fillna(0.0)
        signals["strat_ret"] = signals["position"] * signals["log_ret"]

        # 计算累积收益率
        signals["cum_bh"] = np.exp(signals["log_ret"].cumsum())
        signals["cum_strat"] = np.exp(signals["strat_ret"].cumsum())

        # 验证回测结果
        assert "cum_bh" in signals.columns
        assert "cum_strat" in signals.columns
        assert len(signals["cum_bh"]) == len(signals)
        assert len(signals["cum_strat"]) == len(signals)
