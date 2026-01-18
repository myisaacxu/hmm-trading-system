"""
特征工程模块 - 按照原有程序逻辑实现
负责从原始数据中提取特征，用于HMM模型训练
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Tuple


class FeatureEngineer:
    """特征工程类，按照原有程序逻辑实现"""

    def __init__(self, vol_window: int = 30, ma_short: int = 20, ma_long: int = 100):
        """
        初始化特征工程器

        Args:
            vol_window: 波动率计算窗口
            ma_short: 短期均线窗口
            ma_long: 长期均线窗口
        """
        self.vol_window = vol_window
        self.ma_short = ma_short
        self.ma_long = ma_long
        self._features_df = None

    def calculate_technical_indicators(self, price_series: pd.Series) -> pd.DataFrame:
        """
        计算技术指标，与参考文件保持一致，针对银行股优化

        Args:
            price_series: 价格序列，索引为日期

        Returns:
            DataFrame包含技术指标
        """
        # 确保价格序列没有重复索引，与参考文件一致
        price_series = price_series[~price_series.index.duplicated(keep="first")]
        price_series = price_series.sort_index()

        # 处理NaN值 - 使用前向填充和后向填充
        price_series = price_series.ffill().bfill()

        # 如果仍有NaN值，则填充为0（边界情况）
        price_series = price_series.fillna(0.0)

        # 安全地计算对数收益率，避免极端值警告
        # 确保价格为正数，避免log(0)或log(负数)
        price_positive = price_series.copy()
        price_positive[price_positive <= 0] = 1e-10  # 将0或负数替换为很小的正数

        # 计算对数收益率和波动率特征（针对银行股优化），与参考文件一致
        lr = np.log(price_positive).diff().fillna(0.0)  # 对数收益率

        # 计算波动率，使用可调窗口，与参考文件一致
        vol = (
            lr.rolling(self.vol_window, min_periods=1).std().fillna(0.0)
        )  # 可调波动率窗口

        # 计算短期和长期均线，与参考文件一致
        ma_short = price_series.rolling(self.ma_short).mean().bfill()
        ma_long = price_series.rolling(self.ma_long).mean().bfill()

        # 计算趋势指标，与参考文件一致
        spread = ((ma_short - ma_long) / ma_long).fillna(0.0)  # 可调趋势指标

        # 创建技术指标数据框，与参考文件保持一致的列名
        tech_df = pd.DataFrame(
            {
                "PX": price_series,
                "log_ret": lr,  # 与参考文件一致
                "log_return": lr,  # 兼容其他调用
                "VOL": vol,  # 与参考文件一致
                "volatility": vol,  # 兼容测试
                "MA_SHORT": ma_short,
                "MA_LONG": ma_long,
                "SPREAD": spread,  # 与参考文件一致
                "trend_spread": spread,  # 兼容测试
            }
        )

        return tech_df

    def align_macro_data(
        self,
        tech_df: pd.DataFrame,
        ebs_data: Optional[pd.Series] = None,
        buffett_data: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        对齐宏观数据，与参考文件保持一致

        Args:
            tech_df: 技术指标数据框
            ebs_data: 股债利差数据
            buffett_data: 巴菲特指数数据

        Returns:
            对齐后的特征数据框
        """
        df = tech_df.copy()

        # 处理股债利差数据，与参考文件保持一致
        if ebs_data is not None and not ebs_data.empty:
            # 检查并处理重复索引，与参考文件一致
            ebs_data = ebs_data[~ebs_data.index.duplicated(keep="first")]
            ebs_data = ebs_data.sort_index()
            # 对齐日期索引，使用前向填充，与参考文件一致
            ebs_aligned = ebs_data.reindex(df.index, method="ffill")
            df["EBS"] = ebs_aligned.fillna(0.0)  # 确保填充NaN值
        else:
            # 如果获取失败，创建空的股债利差列
            df["EBS"] = 0.0

        # 处理巴菲特指数数据，与参考文件保持一致
        if buffett_data is not None and not buffett_data.empty:
            # 检查并处理重复索引，与参考文件一致
            buffett_data = buffett_data[~buffett_data.index.duplicated(keep="first")]
            buffett_data = buffett_data.sort_index()
            # 对齐日期索引，使用前向填充，与参考文件一致
            buffett_aligned = buffett_data.reindex(df.index, method="ffill")
            df["BUFFETT"] = buffett_aligned.fillna(0.0)  # 确保填充NaN值
        else:
            # 如果获取失败，创建空的巴菲特指数列
            df["BUFFETT"] = 0.0

        # 确保日期索引对齐，与参考文件一致
        df = df.sort_index()

        # 移除包含NaN的行，与参考文件一致
        # 但保留技术指标可能产生的NaN行，因为它们是计算过程中的正常现象
        df = df.dropna(subset=["PX", "log_ret", "VOL", "SPREAD"])

        return df

    def create_feature_matrix(
        self, features_df: pd.DataFrame, use_standardization: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建特征矩阵，与参考文件保持一致

        Args:
            features_df: 特征数据框
            use_standardization: 是否使用Z-分数标准化

        Returns:
            特征矩阵和标准化后的特征矩阵
        """
        # 确保特征数据框不为空
        if features_df.empty:
            return np.array([]), np.array([])

        # 使用已有的对数收益率列，与参考文件一致
        log_returns = features_df["log_ret"].values

        # 设计矩阵（行对应 df.index），与参考文件保持一致
        X = np.column_stack(
            [
                log_returns,  # 对数收益率，与参考文件一致
                features_df["VOL"].values,  # 波动率，与参考文件一致
                features_df["SPREAD"].values,  # 趋势指标，与参考文件一致
                features_df["EBS"].values,  # 股债利差，与参考文件一致
                features_df["BUFFETT"].values,  # 巴菲特指数，与参考文件一致
            ]
        )

        # 处理异常值，与参考文件一致
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 可选：Z-分数标准化，提高状态区分度，与参考文件一致
        Xz = X.copy()
        if use_standardization and len(X) > 1:  # 只有在数据量足够时才进行标准化
            X_mean = X.mean(axis=0, keepdims=True)
            X_std = X.std(axis=0, keepdims=True) + 1e-12
            Xz = (X - X_mean) / X_std

        return X, Xz

    def engineer_features(
        self,
        price_series: pd.Series,
        ebs_data: Optional[pd.Series] = None,
        buffett_data: Optional[pd.Series] = None,
        use_standardization: bool = True,
    ) -> Dict:
        """
        完整的特征工程流程，与参考文件保持一致

        Args:
            price_series: 价格序列
            ebs_data: 股债利差数据
            buffett_data: 巴菲特指数数据
            use_standardization: 是否使用标准化

        Returns:
            包含特征和矩阵的字典
        """
        # 检查输入数据是否为空
        if price_series.empty:
            return self._create_empty_features()

        # 检查是否为单点数据
        if len(price_series) == 1:
            return self._handle_single_point_data(price_series)

        # 检查数据量是否过少
        if len(price_series) < 10:
            return self._handle_small_dataset(price_series)

        # 计算技术指标
        tech_df = self.calculate_technical_indicators(price_series)

        # 对齐宏观数据
        features_df = self.align_macro_data(tech_df, ebs_data, buffett_data)

        # 创建特征矩阵
        X, Xz = self.create_feature_matrix(features_df, use_standardization)

        self._features_df = features_df

        return {
            "features_df": features_df,
            "X_raw": X,
            "X_standardized": Xz,
            "feature_names": [
                "log_ret",
                "volatility",
                "trend_spread",
                "ebs",
                "buffett",
            ],
        }

    def _create_empty_features(self) -> Dict:
        """创建空特征数据"""
        empty_df = pd.DataFrame(
            columns=[
                "PX",
                "log_ret",
                "log_return",
                "VOL",
                "volatility",
                "MA_SHORT",
                "MA_LONG",
                "SPREAD",
                "trend_spread",
                "EBS",
                "BUFFETT",
            ]
        )
        empty_X = np.array([]).reshape(0, 5)

        return {
            "features_df": empty_df,
            "X_raw": empty_X,
            "X_standardized": empty_X,
            "feature_names": [
                "log_ret",
                "volatility",
                "trend_spread",
                "ebs",
                "buffett",
            ],
        }

    def _handle_single_point_data(self, price_series: pd.Series) -> Dict:
        """处理单点数据"""
        # 为单点数据创建默认特征
        date = price_series.index[0]
        price = price_series.iloc[0]

        single_df = pd.DataFrame(
            {
                "PX": [price],
                "log_ret": [0.0],
                "log_return": [0.0],
                "VOL": [0.0],
                "volatility": [0.0],
                "MA_SHORT": [price],
                "MA_LONG": [price],
                "SPREAD": [0.0],
                "trend_spread": [0.0],
                "EBS": [0.0],
                "BUFFETT": [0.0],
            },
            index=[date],
        )

        single_X = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])

        return {
            "features_df": single_df,
            "X_raw": single_X,
            "X_standardized": single_X,
            "feature_names": [
                "log_ret",
                "volatility",
                "trend_spread",
                "ebs",
                "buffett",
            ],
        }

    def _handle_small_dataset(self, price_series: pd.Series) -> Dict:
        """处理小数据集"""
        # 对于小数据集，使用简单特征计算
        tech_df = self.calculate_technical_indicators(price_series)

        # 创建简化特征
        small_df = pd.DataFrame(
            {
                "PX": tech_df["PX"],
                "log_ret": tech_df["log_ret"],
                "log_return": tech_df["log_return"],
                "VOL": tech_df["VOL"],
                "volatility": tech_df["volatility"],
                "MA_SHORT": tech_df["MA_SHORT"],
                "MA_LONG": tech_df["MA_LONG"],
                "SPREAD": tech_df["SPREAD"],
                "trend_spread": tech_df["trend_spread"],
                "EBS": [0.0] * len(tech_df),
                "BUFFETT": [0.0] * len(tech_df),
            },
            index=tech_df.index,
        )

        # 创建特征矩阵（不使用标准化）
        X = np.column_stack(
            [
                small_df["log_return"].values,
                small_df["VOL"].values,
                small_df["SPREAD"].values,
                small_df["EBS"].values,
                small_df["BUFFETT"].values,
            ]
        )

        return {
            "features_df": small_df,
            "X_raw": X,
            "X_standardized": X,  # 小数据集不使用标准化
            "feature_names": [
                "log_ret",
                "volatility",
                "trend_spread",
                "ebs",
                "buffett",
            ],
        }

    def get_feature_summary(self) -> Dict:
        """获取特征统计摘要"""
        if self._features_df is None:
            return {}

        summary = {}
        for col in self._features_df.columns:
            if col not in ["PX", "MA_SHORT", "MA_LONG"]:  # 跳过价格和均线列
                summary[col] = {
                    "mean": self._features_df[col].mean(),
                    "std": self._features_df[col].std(),
                    "min": self._features_df[col].min(),
                    "max": self._features_df[col].max(),
                }

        return summary

    def validate_features(self) -> bool:
        """验证特征的有效性"""
        if self._features_df is None:
            return False

        # 检查是否有NaN值
        if self._features_df.isnull().any().any():
            return False

        # 检查数据量是否足够
        if len(self._features_df) < 100:
            return False

        # 检查特征是否有足够的变异性
        for col in ["VOL", "SPREAD", "EBS", "BUFFETT"]:
            if col in self._features_df.columns:
                if self._features_df[col].std() < 1e-6:
                    return False

        return True
