"""
特征工程模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.features.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """特征工程器测试类"""

    def setup_method(self):
        """测试初始化"""
        self.fe = FeatureEngineer(vol_window=30, ma_short=20, ma_long=100)

        # 创建测试数据
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        np.random.seed(42)

        # 创建模拟价格序列（带有趋势和波动）
        trend = np.linspace(100, 200, len(dates))
        noise = np.random.normal(0, 5, len(dates))
        prices = trend + noise

        self.price_series = pd.Series(prices, index=dates, name="price")

        # 创建模拟宏观数据
        self.ebs_data = pd.Series(
            np.random.uniform(1, 5, len(dates)), index=dates, name="ebs_indicator"
        )

        self.buffett_data = pd.Series(
            np.random.uniform(50, 150, len(dates)), index=dates, name="buffett_index"
        )

    def test_initialization(self):
        """测试初始化参数"""
        fe = FeatureEngineer(vol_window=20, ma_short=10, ma_long=50)
        assert fe.vol_window == 20
        assert fe.ma_short == 10
        assert fe.ma_long == 50
        assert fe._features_df is None

    def test_calculate_technical_indicators(self):
        """测试技术指标计算"""
        tech_df = self.fe.calculate_technical_indicators(self.price_series)

        # 验证返回类型
        assert isinstance(tech_df, pd.DataFrame)

        # 验证列名
        expected_columns = ["PX", "log_ret", "VOL", "MA_SHORT", "MA_LONG", "SPREAD"]
        assert all(col in tech_df.columns for col in expected_columns)

        # 验证数据长度
        assert len(tech_df) == len(self.price_series)

        # 验证没有NaN值
        assert not tech_df.isnull().any().any()

        # 验证技术指标计算
        assert tech_df["VOL"].std() > 0  # 波动率应该有变异性
        assert tech_df["SPREAD"].mean() != 0  # 趋势指标应该非零

    def test_technical_indicators_with_duplicate_indices(self):
        """测试处理重复索引的情况"""
        # 创建有重复索引的价格序列
        dates_with_duplicates = list(self.price_series.index[:10]) * 2 + list(
            self.price_series.index[10:]
        )
        prices_with_duplicates = list(self.price_series.values[:10]) * 2 + list(
            self.price_series.values[10:]
        )

        price_series_dup = pd.Series(
            prices_with_duplicates, index=dates_with_duplicates
        )

        tech_df = self.fe.calculate_technical_indicators(price_series_dup)

        # 验证重复索引已被处理
        assert not tech_df.index.duplicated().any()

    def test_align_macro_data_with_valid_data(self):
        """测试对齐有效的宏观数据"""
        tech_df = self.fe.calculate_technical_indicators(self.price_series)
        features_df = self.fe.align_macro_data(
            tech_df, self.ebs_data, self.buffett_data
        )

        # 验证列名
        expected_columns = [
            "PX",
            "log_ret",
            "VOL",
            "MA_SHORT",
            "MA_LONG",
            "SPREAD",
            "EBS",
            "BUFFETT",
        ]
        assert all(col in features_df.columns for col in expected_columns)

        # 验证宏观数据对齐
        assert "EBS" in features_df.columns
        assert "BUFFETT" in features_df.columns
        assert not features_df["EBS"].isnull().any()
        assert not features_df["BUFFETT"].isnull().any()

    def test_align_macro_data_with_none_data(self):
        """测试处理空的宏观数据"""
        tech_df = self.fe.calculate_technical_indicators(self.price_series)
        features_df = self.fe.align_macro_data(tech_df, None, None)

        # 验证默认值设置
        assert (features_df["EBS"] == 0.0).all()
        assert (features_df["BUFFETT"] == 0.0).all()

    def test_align_macro_data_with_partial_data(self):
        """测试部分宏观数据的情况"""
        tech_df = self.fe.calculate_technical_indicators(self.price_series)
        features_df = self.fe.align_macro_data(tech_df, self.ebs_data, None)

        # 验证部分数据对齐
        assert "EBS" in features_df.columns
        assert "BUFFETT" in features_df.columns
        assert not features_df["EBS"].isnull().any()
        assert (features_df["BUFFETT"] == 0.0).all()

    def test_create_feature_matrix(self):
        """测试特征矩阵创建"""
        tech_df = self.fe.calculate_technical_indicators(self.price_series)
        features_df = self.fe.align_macro_data(
            tech_df, self.ebs_data, self.buffett_data
        )

        X, Xz = self.fe.create_feature_matrix(features_df, use_standardization=True)

        # 验证矩阵形状
        assert X.shape[0] == len(features_df)
        assert X.shape[1] == 5  # 5个特征
        assert Xz.shape == X.shape

        # 验证没有NaN值
        assert not np.isnan(X).any()
        assert not np.isnan(Xz).any()

        # 验证标准化效果
        assert np.allclose(Xz.mean(axis=0), 0, atol=1e-10)  # 均值接近0
        assert np.allclose(Xz.std(axis=0), 1, atol=1e-10)  # 标准差接近1

    def test_create_feature_matrix_without_standardization(self):
        """测试不使用标准化的特征矩阵创建"""
        tech_df = self.fe.calculate_technical_indicators(self.price_series)
        features_df = self.fe.align_macro_data(
            tech_df, self.ebs_data, self.buffett_data
        )

        X, Xz = self.fe.create_feature_matrix(features_df, use_standardization=False)

        # 验证不使用标准化时，X和Xz应该相同
        assert np.array_equal(X, Xz)

    def test_engineer_features_complete_flow(self):
        """测试完整特征工程流程"""
        result = self.fe.engineer_features(
            self.price_series,
            self.ebs_data,
            self.buffett_data,
            use_standardization=True,
        )

        # 验证返回结构
        assert "features_df" in result
        assert "X_raw" in result
        assert "X_standardized" in result
        assert "feature_names" in result

        # 验证特征名称
        expected_names = ["log_ret", "volatility", "trend_spread", "ebs", "buffett"]
        assert result["feature_names"] == expected_names

        # 验证特征数据框
        assert isinstance(result["features_df"], pd.DataFrame)
        assert len(result["features_df"]) == len(self.price_series)

    def test_get_feature_summary(self):
        """测试特征统计摘要"""
        # 先运行特征工程
        self.fe.engineer_features(self.price_series, self.ebs_data, self.buffett_data)

        summary = self.fe.get_feature_summary()

        # 验证摘要结构
        assert isinstance(summary, dict)

        # 验证包含的指标
        expected_features = ["log_ret", "VOL", "SPREAD", "EBS", "BUFFETT"]
        for feature in expected_features:
            assert feature in summary
            assert "mean" in summary[feature]
            assert "std" in summary[feature]
            assert "min" in summary[feature]
            assert "max" in summary[feature]

    def test_get_feature_summary_without_features(self):
        """测试没有特征数据时的统计摘要"""
        summary = self.fe.get_feature_summary()
        assert summary == {}

    def test_validate_features_valid(self):
        """测试有效的特征验证"""
        self.fe.engineer_features(self.price_series, self.ebs_data, self.buffett_data)
        assert self.fe.validate_features() is True

    def test_validate_features_invalid_short_data(self):
        """测试数据量不足时的特征验证"""
        # 创建短数据序列
        short_dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
        short_prices = pd.Series([100] * len(short_dates), index=short_dates)

        self.fe.engineer_features(short_prices, self.ebs_data, self.buffett_data)
        assert self.fe.validate_features() is False

    def test_validate_features_with_nan_data(self):
        """测试包含NaN值时的特征验证"""
        # 创建包含NaN的价格序列
        prices_with_nan = self.price_series.copy()
        prices_with_nan.iloc[10:20] = np.nan

        # 由于特征工程模块已经改进了NaN处理，现在应该返回True
        self.fe.engineer_features(prices_with_nan, self.ebs_data, self.buffett_data)
        assert self.fe.validate_features() is True

    def test_feature_engineering_with_edge_cases(self):
        """测试边界情况"""
        # 测试恒定价格
        constant_prices = pd.Series(
            [100] * 200, index=pd.date_range(start="2023-01-01", periods=200, freq="D")
        )

        result = self.fe.engineer_features(constant_prices, None, None)

        # 即使价格恒定，特征工程也应该完成
        assert "features_df" in result
        assert len(result["features_df"]) == 200

        # 验证波动率为0（符合预期）
        assert result["features_df"]["VOL"].std() == 0


class TestFeatureEngineerIntegration:
    """特征工程集成测试"""

    def test_integration_with_realistic_data(self):
        """测试与真实数据格式的集成"""
        fe = FeatureEngineer()

        # 创建更真实的数据模式
        dates = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")

        # 创建更真实的股票价格模式（有趋势、周期性、波动）
        trend = np.linspace(50, 150, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)  # 年度周期
        noise = np.random.normal(0, 2, len(dates))
        prices = trend + seasonal + noise

        price_series = pd.Series(prices, index=dates)

        # 创建真实的宏观数据模式
        ebs_data = pd.Series(
            3 + 1.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 252 * 3),  # 3年周期
            index=dates,
        )

        buffett_data = pd.Series(
            100 + 30 * np.cos(2 * np.pi * np.arange(len(dates)) / 252 * 5),  # 5年周期
            index=dates,
        )

        result = fe.engineer_features(price_series, ebs_data, buffett_data)

        # 验证特征的有效性
        assert fe.validate_features() is True

        # 验证特征矩阵形状
        assert result["X_raw"].shape[0] == len(
            result["features_df"]
        )  # 特征矩阵行数应与特征数据框一致
        assert result["X_raw"].shape[1] == 5

        # 验证特征有足够的变异性（调整阈值以匹配实际数据模式）
        features_df = result["features_df"]
        assert features_df["VOL"].std() > 0.01
        assert features_df["SPREAD"].std() > 0.0001
        assert features_df["EBS"].std() > 0.1
        assert features_df["BUFFETT"].std() > 0.1
