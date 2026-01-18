"""
数据处理器模块
负责数据清洗、异常值处理和特征工程
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import warnings


class DataProcessor:
    """
    数据处理器类

    功能：
    1. 数据清洗和异常值处理
    2. 特征工程和指标计算
    3. 数据标准化
    """

    def __init__(
        self,
        volatility_window: int = 30,
        ma_short_window: int = 20,
        ma_long_window: int = 100,
    ):
        """
        初始化数据处理器

        Args:
            volatility_window: 波动率计算窗口
            ma_short_window: 短期均线窗口
            ma_long_window: 长期均线窗口
        """
        self.volatility_window = volatility_window
        self.ma_short_window = ma_short_window
        self.ma_long_window = ma_long_window

    def clean_data(
        self, data: pd.DataFrame, price_col: str = "close", volume_col: str = "volume"
    ) -> pd.DataFrame:
        """
        数据清洗

        Args:
            data: 原始数据
            price_col: 价格列名
            volume_col: 成交量列名

        Returns:
            pd.DataFrame: 清洗后的数据
        """
        cleaned_data = data.copy()

        # 处理缺失值
        cleaned_data = cleaned_data.ffill().bfill()

        # 处理价格异常值（使用IQR方法）
        if price_col in cleaned_data.columns:
            Q1 = cleaned_data[price_col].quantile(0.25)
            Q3 = cleaned_data[price_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 将异常值替换为边界值
            cleaned_data[price_col] = np.clip(
                cleaned_data[price_col], lower_bound, upper_bound
            )

        # 处理成交量异常值
        if volume_col in cleaned_data.columns:
            # 使用移动中位数处理成交量异常值
            volume_median = (
                cleaned_data[volume_col].rolling(window=10, min_periods=1).median()
            )
            volume_std = (
                cleaned_data[volume_col].rolling(window=10, min_periods=1).std()
            )

            # 将超过3倍标准差的成交量视为异常值
            outlier_mask = (
                cleaned_data[volume_col] - volume_median
            ).abs() > 3 * volume_std
            cleaned_data.loc[outlier_mask, volume_col] = volume_median[outlier_mask]

        return cleaned_data

    def detect_outliers(self, data: pd.Series, method: str = "iqr") -> pd.Series:
        """
        检测异常值

        Args:
            data: 数据序列
            method: 检测方法，支持'iqr'和'zscore'

        Returns:
            pd.Series: 异常值序列
        """
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data < lower_bound) | (data > upper_bound)]

        elif method == "zscore":
            z_scores = (data - data.mean()) / data.std()
            outliers = data[z_scores.abs() > 3]

        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")

        return outliers

    def normalize_data(
        self, data: pd.DataFrame, method: str = "zscore"
    ) -> pd.DataFrame:
        """
        数据标准化

        Args:
            data: 原始数据
            method: 标准化方法，支持'zscore'和'minmax'

        Returns:
            pd.DataFrame: 标准化后的数据
        """
        normalized_data = data.copy()

        if method == "zscore":
            # Z-score标准化
            normalized_data = (data - data.mean()) / data.std()

        elif method == "minmax":
            # 最小-最大标准化
            normalized_data = (data - data.min()) / (data.max() - data.min())

        else:
            raise ValueError(f"不支持的标准化方法: {method}")

        # 处理无穷值和NaN值
        normalized_data = normalized_data.replace([np.inf, -np.inf], np.nan)
        normalized_data = normalized_data.fillna(0)

        return normalized_data

    def create_features(
        self, prices: pd.Series, macro_data: Dict[str, pd.Series] = None
    ) -> pd.DataFrame:
        """
        创建特征数据

        Args:
            prices: 价格序列
            macro_data: 宏观数据字典

        Returns:
            pd.DataFrame: 特征数据
        """
        features_df = pd.DataFrame(index=prices.index)

        # 基本价格特征
        features_df["price"] = prices

        # 对数收益率
        log_returns = np.log(prices).diff().fillna(0.0)
        features_df["log_return"] = log_returns

        # 波动率特征
        volatility = (
            log_returns.rolling(self.volatility_window, min_periods=1).std().fillna(0.0)
        )
        features_df["volatility"] = volatility

        # 均线特征
        ma_short = prices.rolling(self.ma_short_window).mean().bfill()
        ma_long = prices.rolling(self.ma_long_window).mean().bfill()
        spread = ((ma_short - ma_long) / ma_long).fillna(0.0)

        features_df["ma_short"] = ma_short
        features_df["ma_long"] = ma_long
        features_df["spread"] = spread

        # 添加宏观特征
        if macro_data:
            for name, series in macro_data.items():
                # 对齐日期索引
                aligned_series = series.reindex(prices.index, method="ffill")
                features_df[name] = aligned_series

        # 处理缺失值
        features_df = features_df.ffill().bfill().fillna(0)

        return features_df

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            pd.DataFrame: 包含技术指标的DataFrame
        """
        indicators_df = pd.DataFrame(index=data.index)

        if "close" in data.columns:
            close = data["close"]

            # RSI指标
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators_df["rsi"] = 100 - (100 / (1 + rs))

            # MACD指标
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            indicators_df["macd"] = ema_12 - ema_26
            indicators_df["macd_signal"] = indicators_df["macd"].ewm(span=9).mean()
            indicators_df["macd_histogram"] = (
                indicators_df["macd"] - indicators_df["macd_signal"]
            )

            # 布林带
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            indicators_df["bb_upper"] = sma_20 + 2 * std_20
            indicators_df["bb_middle"] = sma_20
            indicators_df["bb_lower"] = sma_20 - 2 * std_20

        # 处理缺失值
        indicators_df = indicators_df.ffill().bfill().fillna(0)

        return indicators_df

    def prepare_features_for_model(
        self, features_df: pd.DataFrame, feature_cols: List[str] = None
    ) -> np.ndarray:
        """
        准备模型输入特征

        Args:
            features_df: 特征数据
            feature_cols: 使用的特征列，默认使用所有数值列

        Returns:
            np.ndarray: 模型输入特征矩阵
        """
        if feature_cols is None:
            # 默认使用所有数值列
            feature_cols = features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        # 选择特征列
        X = features_df[feature_cols].values

        # 处理NaN和无穷值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化特征
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-12
        X_normalized = (X - X_mean) / X_std

        return X_normalized
