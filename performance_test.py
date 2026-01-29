"""
性能测试脚本
测试HMM模型优化模块的性能
"""

import time
import numpy as np
import pandas as pd
from src.models.hmm_regime_detector import HMMRegimeDetector
from src.models.hmm_optimizer import HMMOptimizer
from src.models.bayesian_optimizer import BayesianOptimizer


def generate_test_data(n_samples=1000, n_features=5):
    """生成测试数据"""
    np.random.seed(42)
    X = np.random.normal(0, 1, (n_samples, n_features))

    # 创建模拟价格和收益率数据
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_samples)))
    log_ret = np.log(prices[1:]) - np.log(prices[:-1])
    log_ret = np.insert(log_ret, 0, 0)  # 添加初始值

    data = pd.DataFrame(
        {
            "PX": prices,
            "log_ret": log_ret,
            "VOL": np.abs(np.random.normal(0.1, 0.05, n_samples)),
            "SPREAD": np.random.normal(0.0, 0.02, n_samples),
            "EBS": np.random.normal(3.0, 0.5, n_samples),
            "BUFFETT": np.random.normal(100, 15, n_samples),
        },
        index=dates,
    )

    return X, data


def test_grid_search_performance():
    """测试网格搜索性能"""
    print("\n=== 测试网格搜索性能 ===")
    X, _ = generate_test_data()

    # 定义参数网格
    param_grid = {
        "n_states": [2, 3],
        "covariance_type": ["diag", "full"],
        "n_iter": [100, 200],
    }

    optimizer = HMMOptimizer()

    # 测量执行时间
    start_time = time.time()
    result = optimizer.optimize_parameters(X, param_grid, n_splits=2)
    end_time = time.time()

    print("执行时间: %.2f 秒" % (end_time - start_time))
    print("最佳参数: %s" % result.get("best_params", {}))
    print("最佳得分: %.4f" % result.get("best_score", 0))


def test_bayesian_optimization_performance():
    """测试贝叶斯优化性能"""
    print("\n=== 测试贝叶斯优化性能 ===")
    X, _ = generate_test_data()

    optimizer = BayesianOptimizer()

    # 测量执行时间
    start_time = time.time()
    result = optimizer.optimize_hmm_parameters(X, n_calls=20, n_splits=2)
    end_time = time.time()

    print("执行时间: %.2f 秒" % (end_time - start_time))
    print("最佳参数: %s" % result.get("best_params", {}))
    print("最佳得分: %.4f" % result.get("best_score", 0))


def test_feature_selection_performance():
    """测试特征选择性能"""
    print("\n=== 测试特征选择性能 ===")
    _, data = generate_test_data()

    feature_columns = ["log_ret", "VOL", "SPREAD", "EBS", "BUFFETT"]

    optimizer = HMMOptimizer()

    # 测量执行时间
    start_time = time.time()
    result = optimizer.optimize_features(data, feature_columns, n_states=3)
    end_time = time.time()

    print("执行时间: %.2f 秒" % (end_time - start_time))
    print("最佳特征: %s" % result.get("best_features", []))
    print("最佳得分: %.4f" % result.get("best_score", 0))


def test_model_training_performance():
    """测试模型训练性能"""
    print("\n=== 测试模型训练性能 ===")
    X, _ = generate_test_data()

    detector = HMMRegimeDetector(n_states=3)

    # 测量执行时间
    start_time = time.time()
    detector.fit(X)
    end_time = time.time()

    print("执行时间: %.2f 秒" % (end_time - start_time))

    # 测量预测时间
    start_time = time.time()
    states, proba = detector.predict(X)
    end_time = time.time()

    print("预测时间: %.2f 秒" % (end_time - start_time))


def test_optimized_model_performance():
    """测试优化后模型的性能"""
    print("\n=== 测试优化后模型的性能 ===")
    X, data = generate_test_data()

    # 首先使用贝叶斯优化获取最佳参数
    optimizer = BayesianOptimizer()
    opt_result = optimizer.optimize_hmm_parameters(X, n_calls=20, n_splits=2)
    best_params = opt_result.get("best_params", {})

    print("使用最佳参数: %s" % best_params)

    # 使用最佳参数训练模型
    detector = HMMRegimeDetector(
        n_states=best_params.get("n_states", 3),
        covariance_type=best_params.get("covariance_type", "diag"),
        n_iter=best_params.get("n_iter", 100),
    )

    start_time = time.time()
    detector.fit(X)
    end_time = time.time()

    print("训练时间: %.2f 秒" % (end_time - start_time))

    # 预测并评估
    states, proba = detector.predict(
        X,
        min_len=best_params.get("min_duration", 10),
        sticky_strength=best_params.get("sticky_strength", 5.0),
    )

    # 生成交易信号
    from src.models.hmm_regime_detector import MarketStateAnalyzer

    analyzer = MarketStateAnalyzer()

    # 获取状态标签
    returns = data["log_ret"]
    state_labels = detector.get_state_labels(returns, states)

    # 生成交易信号
    signals = analyzer.generate_trading_signals(data, states, state_labels)

    # 评估策略性能
    performance = analyzer.evaluate_strategy_performance(signals)
    metrics = performance.get("metrics", {})

    print("\n策略性能:")
    print("年化收益率: %.2f%%" % (metrics.get("cagr", 0) * 100))
    print("夏普比率: %.2f" % metrics.get("sharpe_ratio", 0))
    print("最大回撤: %.2f%%" % (metrics.get("max_drawdown", 0) * 100))


def main():
    """主函数"""
    print("开始性能测试...")

    # 测试各功能的性能
    test_grid_search_performance()
    test_bayesian_optimization_performance()
    test_feature_selection_performance()
    test_model_training_performance()
    test_optimized_model_performance()

    print("\n性能测试完成!")


if __name__ == "__main__":
    main()
