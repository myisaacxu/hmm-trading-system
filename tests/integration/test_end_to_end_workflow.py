"""
端到端工作流集成测试
测试整个系统从数据获取到策略生成的完整流程
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# 导入项目模块
from src.data.data_fetcher import DataFetcher
from src.features.feature_engineer import FeatureEngineer
from src.models.hmm_regime_detector import HMMRegimeDetector, MarketStateAnalyzer
from src.strategies.trading_strategy import TradingStrategy


class TestEndToEndWorkflow:
    """端到端工作流集成测试类"""

    def test_complete_workflow_with_mock_data(
        self, sample_stock_data, sample_macro_data
    ):
        """测试使用模拟数据的完整工作流"""
        # 1. 创建各模块实例
        data_fetcher = DataFetcher()
        feature_engineer = FeatureEngineer()
        hmm_detector = HMMRegimeDetector(n_states=3, n_iter=50, random_state=42)

        # 2. 模拟数据获取（使用fixture数据）
        stock_data = sample_stock_data
        macro_data = sample_macro_data

        # 3. 特征工程 - 使用正确的函数名
        price_series = stock_data["close"]
        ebs_data = (
            macro_data["ebs_indicator"]
            if "ebs_indicator" in macro_data.columns
            else None
        )
        buffett_data = (
            macro_data["buffett_index"]
            if "buffett_index" in macro_data.columns
            else None
        )

        features_result = feature_engineer.engineer_features(
            price_series, ebs_data, buffett_data
        )

        # 验证特征工程结果
        assert "features_df" in features_result, "特征工程应该返回特征数据框"
        features_df = features_result["features_df"]
        assert not features_df.empty, "特征工程应该生成非空特征"

        # 4. HMM模型训练和状态识别
        # 使用标准化后的特征矩阵进行训练
        X_standardized = features_result["X_standardized"]
        hmm_detector.fit(X_standardized)
        regimes, proba = hmm_detector.predict(X_standardized)

        assert len(regimes) == len(features_df), "状态序列长度应与特征数据一致"
        assert set(regimes).issubset({0, 1, 2}), "状态应该为0,1,2"

        # 5. 状态分析
        state_analyzer = MarketStateAnalyzer()
        # 使用正确的状态标签映射
        state_labels = hmm_detector.get_state_labels(features_df["log_ret"], regimes)
        performance = state_analyzer.analyze_regime_performance(
            features_df, regimes, state_labels
        )
        signals = state_analyzer.generate_trading_signals(
            features_df, regimes, state_labels
        )

        assert len(performance) > 0, "状态分析应该生成结果"
        assert len(signals) == len(features_df), "信号数据长度应与特征数据一致"

        # 6. 交易策略生成
        trading_strategy = TradingStrategy()
        strategy_results = trading_strategy.generate_signals(stock_data, regimes)

        assert "signals_df" in strategy_results, "策略结果应包含信号数据"
        assert "performance" in strategy_results, "策略结果应包含性能指标"

        # 验证整个流程的成功执行
        assert True  # 所有步骤都成功执行

    def test_data_alignment_in_workflow(self, sample_stock_data, sample_macro_data):
        """测试工作流中数据对齐功能"""
        # 创建不同时间范围的数据来测试对齐功能
        stock_dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        macro_dates = pd.date_range(
            start="2020-06-01", end="2023-06-30", freq="D"
        )  # 不同时间范围

        # 生成模拟数据
        stock_data = pd.DataFrame(
            {"close": np.random.normal(10, 1, len(stock_dates))}, index=stock_dates
        )

        macro_data = pd.DataFrame(
            {"ebs_indicator": np.random.normal(3, 0.5, len(macro_dates))},
            index=macro_dates,
        )

        # 测试特征工程的数据对齐
        feature_engineer = FeatureEngineer()
        price_series = stock_data["close"]
        ebs_data = macro_data["ebs_indicator"]

        features_result = feature_engineer.engineer_features(price_series, ebs_data)
        features_df = features_result["features_df"]

        # 验证数据对齐 - 使用实际特征工程对齐逻辑
        if not features_df.empty:
            # 特征工程使用前向填充，对齐后的数据应该基于股票数据的索引
            aligned_dates = price_series.index

            # 验证特征数据索引与价格数据索引一致
            assert features_df.index.equals(
                aligned_dates
            ), "特征数据索引应与价格数据索引对齐"

            # 验证数据长度一致
            assert len(features_df) == len(price_series), "特征数据长度应与价格数据一致"

            # 验证数据完整性
            assert not features_df.isnull().any().any(), "特征数据不应包含NaN值"
        else:
            # 如果特征数据为空，可能是数据量过少，这是合理的
            assert len(price_series) < 10, "只有数据量过少时特征数据才可能为空"

    def test_error_handling_in_workflow(self):
        """测试工作流中的错误处理机制"""
        # 测试空数据处理
        feature_engineer = FeatureEngineer()
        hmm_detector = HMMRegimeDetector()

        # 空数据测试
        empty_price_series = pd.Series([], dtype=float)

        # 特征工程应该能够处理空数据
        features_result = feature_engineer.engineer_features(empty_price_series)
        features_df = features_result["features_df"]
        assert features_df.empty, "空数据输入应该返回空DataFrame"

        # HMM模型应该能够处理空特征
        X_standardized = features_result["X_standardized"]
        with pytest.raises(ValueError):
            hmm_detector.fit(X_standardized)


class TestModuleIntegration:
    """模块间集成测试类"""

    def test_data_fetcher_to_feature_engineer_integration(self, sample_stock_data):
        """测试数据获取模块到特征工程模块的集成"""
        # 模拟数据获取（使用fixture数据）
        stock_data = sample_stock_data

        # 特征工程处理
        feature_engineer = FeatureEngineer()
        price_series = stock_data["close"]

        # 使用正确的函数名
        tech_features = feature_engineer.calculate_technical_indicators(price_series)

        # 验证数据传递正确性
        assert not tech_features.empty, "特征工程应该生成特征数据"
        assert "log_return" in tech_features.columns, "应该包含对数收益率特征"
        assert "volatility" in tech_features.columns, "应该包含波动率特征"

        # 验证数据完整性
        assert len(tech_features) == len(stock_data), "特征数据长度应与输入数据一致"
        assert tech_features.index.equals(
            stock_data.index
        ), "特征数据索引应与输入数据一致"

    def test_feature_engineer_to_hmm_integration(self, sample_features, hmm_config):
        """测试特征工程到HMM模型的集成"""
        # 准备特征数据
        features_df = sample_features.dropna()

        # 创建并训练HMM模型
        hmm_detector = HMMRegimeDetector(**hmm_config)

        # 需要将DataFrame转换为特征矩阵格式
        # 使用特征工程模块创建特征矩阵
        feature_engineer = FeatureEngineer()

        # 假设sample_features已经包含技术指标，我们需要将其转换为适合HMM的格式
        # 这里我们简化处理，直接使用数值列
        X = features_df.select_dtypes(include=[np.number]).values

        # 标准化特征
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X)

        hmm_detector.fit(X_standardized)

        # 预测市场状态
        regimes, proba = hmm_detector.predict(X_standardized)

        # 验证集成结果
        assert len(regimes) == len(features_df), "状态序列长度应与特征数据一致"
        assert set(regimes).issubset({0, 1, 2}), "状态应该为0,1,2"

        # 验证模型收敛性
        assert hmm_detector._is_fitted, "HMM模型应该被正确训练"

        # 验证状态平滑
        smoothed_regimes = hmm_detector.smooth_regimes(regimes)
        assert len(smoothed_regimes) == len(regimes), "平滑后的状态序列长度应保持不变"

    def test_hmm_to_strategy_integration(self, sample_regimes, sample_stock_data):
        """测试HMM模型到交易策略的集成"""
        # 准备输入数据
        regimes = sample_regimes
        stock_data = sample_stock_data

        # 创建交易策略
        trading_strategy = TradingStrategy()
        strategy_results = trading_strategy.generate_signals(stock_data, regimes)

        # 验证集成结果
        assert "signals_df" in strategy_results, "策略结果应包含信号数据"
        assert "performance" in strategy_results, "策略结果应包含性能指标"

        signals_df = strategy_results["signals_df"]
        performance = strategy_results["performance"]

        # 验证信号数据
        assert not signals_df.empty, "信号数据不应为空"

        # 验证信号数据包含预期的列
        expected_columns = ["regime", "position", "price"]
        actual_columns = signals_df.columns.tolist()

        # 检查是否包含预期的列
        for col in expected_columns:
            assert (
                col in actual_columns
            ), f"信号数据应包含{col}列，实际列: {actual_columns}"

        # 验证性能指标
        assert hasattr(performance, "total_return"), "性能指标应包含总收益率"
        assert hasattr(performance, "sharpe"), "性能指标应包含夏普比率"
