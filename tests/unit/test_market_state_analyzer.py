#!/usr/bin/env python3
"""
market_state_analyzer.py 单元测试
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.models.market_state_analyzer import MarketStateAnalyzer, StateAnalysis


class TestMarketStateAnalyzer(unittest.TestCase):
    """测试MarketStateAnalyzer类的功能"""

    def setUp(self):
        """设置测试环境"""
        # 创建MarketStateAnalyzer实例
        self.analyzer = MarketStateAnalyzer()

        # 创建测试数据
        dates = pd.date_range("2023-01-01", periods=100)
        self.price_data = pd.DataFrame(
            {
                "close": np.cumsum(np.random.randn(100)) + 100,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

        self.regime_series = pd.Series(np.random.randint(0, 3, 100), index=dates)

        self.feature_data = pd.DataFrame(
            {
                "rsi": np.random.randn(100) * 20 + 50,
                "macd": np.random.randn(100),
                "bollinger_upper": np.random.randn(100) + 110,
                "bollinger_lower": np.random.randn(100) + 90,
                "atr": np.random.randn(100) + 2,
                "adx": np.random.randn(100) + 20,
                "cci": np.random.randn(100) * 50,
            },
            index=dates,
        )

    def test_analyze_states(self):
        """测试分析市场状态特征"""
        # 运行状态分析
        analysis_results = self.analyzer.analyze_states(
            self.price_data, self.regime_series, self.feature_data
        )

        # 检查返回值
        self.assertIsInstance(analysis_results, dict)
        for state_id, analysis in analysis_results.items():
            self.assertIsInstance(state_id, (int, np.integer))
            self.assertIsInstance(analysis, StateAnalysis)

    def test_analyze_states_empty_data(self):
        """测试使用空数据进行状态分析"""
        # 创建空数据
        empty_price_data = pd.DataFrame()
        empty_regime_series = pd.Series()
        empty_feature_data = pd.DataFrame()

        # 运行状态分析
        analysis_results = self.analyzer.analyze_states(
            empty_price_data, empty_regime_series, empty_feature_data
        )

        # 检查返回值
        self.assertIsInstance(analysis_results, dict)
        self.assertEqual(len(analysis_results), 0)

    def test_analyze_states_missing_close(self):
        """测试缺少close列的价格数据进行状态分析"""
        # 创建缺少close列的价格数据
        price_data_no_close = self.price_data.drop("close", axis=1)

        # 检查是否抛出ValueError
        with self.assertRaises(ValueError):
            self.analyzer.analyze_states(
                price_data_no_close, self.regime_series, self.feature_data
            )

    def test_calculate_transition_probabilities(self):
        """测试计算状态转移概率"""
        # 创建测试状态序列
        regime_series = pd.Series([0, 0, 1, 1, 2, 2, 0, 1, 2])
        unique_states = [0, 1, 2]

        # 计算转移概率
        transition_probs = self.analyzer._calculate_transition_probabilities(
            regime_series, 0, unique_states
        )

        # 检查返回值
        self.assertIsInstance(transition_probs, list)
        self.assertEqual(len(transition_probs), 3)
        self.assertAlmostEqual(sum(transition_probs), 1.0, delta=0.01)

    def test_calculate_state_features(self):
        """测试计算状态特征统计"""
        # 创建测试状态数据
        state_data = self.price_data.copy()
        state_data = state_data.join(self.feature_data)

        # 计算状态特征
        features = self.analyzer._calculate_state_features(state_data)

        # 检查返回值
        self.assertIsInstance(features, dict)
        self.assertIn("price_mean", features)
        self.assertIn("price_std", features)
        self.assertIn("volume_mean", features)
        self.assertIn("rsi_mean", features)
        self.assertIn("macd_mean", features)

    def test_get_state_summary(self):
        """测试生成状态分析摘要"""
        # 先运行状态分析
        analysis_results = self.analyzer.analyze_states(
            self.price_data, self.regime_series, self.feature_data
        )

        # 生成状态摘要
        summary = self.analyzer.get_state_summary(analysis_results)

        # 检查返回值
        self.assertIsInstance(summary, pd.DataFrame)
        expected_columns = [
            "状态ID",
            "状态名称",
            "持续时间比例",
            "平均收益率",
            "波动率",
            "样本数量",
        ]
        for col in expected_columns:
            self.assertIn(col, summary.columns)

    def test_get_state_summary_empty(self):
        """测试使用空分析结果生成状态摘要"""
        # 生成状态摘要
        summary = self.analyzer.get_state_summary({})

        # 检查返回值
        self.assertIsInstance(summary, pd.DataFrame)
        expected_columns = [
            "状态ID",
            "状态名称",
            "持续时间比例",
            "平均收益率",
            "波动率",
            "样本数量",
        ]
        for col in expected_columns:
            self.assertIn(col, summary.columns)
        self.assertEqual(len(summary), 0)

    def test_identify_dominant_state(self):
        """测试识别主导状态"""
        # 先运行状态分析
        analysis_results = self.analyzer.analyze_states(
            self.price_data, self.regime_series, self.feature_data
        )

        # 识别主导状态
        dominant_state = self.analyzer.identify_dominant_state(analysis_results)

        # 检查返回值
        if analysis_results:
            self.assertIsInstance(dominant_state, (int, np.integer))
        else:
            self.assertIsNone(dominant_state)

    def test_identify_dominant_state_empty(self):
        """测试使用空分析结果识别主导状态"""
        # 识别主导状态
        dominant_state = self.analyzer.identify_dominant_state({})

        # 检查返回值
        self.assertIsNone(dominant_state)

    def test_detect_state_transitions(self):
        """测试检测状态转换点"""
        # 创建有明确状态转换的状态序列
        regime_series = pd.Series(
            [0, 0, 1, 1, 2, 2, 0], index=self.price_data.index[:7]
        )

        # 检测状态转换
        transitions = self.analyzer.detect_state_transitions(regime_series)

        # 检查返回值
        self.assertIsInstance(transitions, list)
        # 应该有3个状态转换点
        self.assertEqual(len(transitions), 3)

    def test_detect_state_transitions_short(self):
        """测试使用短状态序列检测状态转换点"""
        # 创建短状态序列
        regime_series = pd.Series([0], index=self.price_data.index[:1])

        # 检测状态转换
        transitions = self.analyzer.detect_state_transitions(regime_series)

        # 检查返回值
        self.assertIsInstance(transitions, list)
        self.assertEqual(len(transitions), 0)

    def test_get_state_colors(self):
        """测试获取状态颜色映射"""
        # 获取状态颜色映射
        state_colors = self.analyzer.get_state_colors()

        # 检查返回值
        self.assertIsInstance(state_colors, dict)
        for state_id, color in state_colors.items():
            self.assertIsInstance(state_id, int)
            self.assertIsInstance(color, str)

    def test_get_state_names(self):
        """测试获取状态名称映射"""
        # 获取状态名称映射
        state_names = self.analyzer.get_state_names()

        # 检查返回值
        self.assertIsInstance(state_names, dict)
        for state_id, name in state_names.items():
            self.assertIsInstance(state_id, int)
            self.assertIsInstance(name, str)


if __name__ == "__main__":
    unittest.main()
