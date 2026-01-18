"""
技术指标计算模块
负责计算各种技术指标和特征
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings


class TechnicalIndicators:
    """
    技术指标计算类

    功能：
    1. 计算价格相关指标（收益率、波动率等）
    2. 计算趋势指标（均线、MACD等）
    3. 计算振荡器指标（RSI、布林带等）
    """

    def __init__(
        self,
        volatility_window: int = 30,
        ma_short_window: int = 20,
        ma_long_window: int = 100,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        """
        初始化技术指标计算器

        Args:
            volatility_window: 波动率计算窗口
            ma_short_window: 短期均线窗口
            ma_long_window: 长期均线窗口
            rsi_window: RSI计算窗口
            macd_fast: MACD快速线窗口
            macd_slow: MACD慢速线窗口
            macd_signal: MACD信号线窗口
        """
        self.volatility_window = volatility_window
        self.ma_short_window = ma_short_window
        self.ma_long_window = ma_long_window
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        计算对数收益率

        Args:
            prices: 价格序列

        Returns:
            pd.Series: 对数收益率序列
        """
        log_returns = np.log(prices).diff().fillna(0.0)
        return log_returns

    def calculate_volatility(
        self, returns: pd.Series, window: Optional[int] = None
    ) -> pd.Series:
        """
        计算滚动波动率

        Args:
            returns: 收益率序列
            window: 计算窗口，默认使用实例变量

        Returns:
            pd.Series: 波动率序列
        """
        if window is None:
            window = self.volatility_window

        volatility = returns.rolling(window, min_periods=1).std().fillna(0.0)
        return volatility

    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        计算移动平均线

        Args:
            prices: 价格序列

        Returns:
            dict: 包含各种移动平均线的字典
        """
        ma_short = prices.rolling(self.ma_short_window).mean().bfill()
        ma_long = prices.rolling(self.ma_long_window).mean().bfill()

        # 计算趋势指标
        spread = ((ma_short - ma_long) / ma_long).fillna(0.0)

        return {"ma_short": ma_short, "ma_long": ma_long, "spread": spread}

    def calculate_rsi(
        self, prices: pd.Series, window: Optional[int] = None
    ) -> pd.Series:
        """
        计算相对强弱指数(RSI)

        Args:
            prices: 价格序列
            window: RSI计算窗口，默认使用实例变量

        Returns:
            pd.Series: RSI序列
        """
        if window is None:
            window = self.rsi_window

        delta = prices.diff()

        # 计算涨跌幅
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 计算平均涨跌幅
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        # 计算RSI
        rs = avg_gain / (avg_loss + 1e-10)  # 避免除零
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)  # 默认中性值

    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        计算MACD指标

        Args:
            prices: 价格序列

        Returns:
            dict: 包含MACD、信号线和柱状图的字典
        """
        # 计算EMA
        ema_fast = prices.ewm(span=self.macd_fast).mean()
        ema_slow = prices.ewm(span=self.macd_slow).mean()

        # 计算MACD线
        macd_line = ema_fast - ema_slow

        # 计算信号线
        signal_line = macd_line.ewm(span=self.macd_signal).mean()

        # 计算柱状图
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def calculate_bollinger_bands(
        self, prices: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        计算布林带

        Args:
            prices: 价格序列
            window: 计算窗口
            num_std: 标准差倍数

        Returns:
            dict: 包含布林带上中下轨的字典
        """
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper_band = sma + num_std * std
        lower_band = sma - num_std * std

        # 计算布林带宽度和位置
        bandwidth = (upper_band - lower_band) / sma
        position = (prices - lower_band) / (upper_band - lower_band)

        return {
            "upper": upper_band,
            "middle": sma,
            "lower": lower_band,
            "bandwidth": bandwidth,
            "position": position,
        }

    def calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> pd.Series:
        """
        计算平均真实波幅(ATR)

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            window: 计算窗口

        Returns:
            pd.Series: ATR序列
        """
        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算平均真实波幅
        atr = tr.rolling(window=window).mean()

        return atr.bfill()

    def calculate_volume_indicators(
        self, volume: pd.Series, close: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        计算成交量相关指标

        Args:
            volume: 成交量序列
            close: 收盘价序列

        Returns:
            dict: 包含成交量指标的字典
        """
        # 成交量均线
        volume_ma = volume.rolling(window=20).mean()

        # 成交量比率
        volume_ratio = volume / volume_ma

        # 资金流向
        money_flow = volume * close
        money_flow_ma = money_flow.rolling(window=20).mean()

        return {
            "volume_ma": volume_ma,
            "volume_ratio": volume_ratio,
            "money_flow": money_flow,
            "money_flow_ma": money_flow_ma,
        }

    def create_comprehensive_features(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        创建综合技术指标特征

        Args:
            ohlcv_data: 包含OHLCV数据的DataFrame

        Returns:
            pd.DataFrame: 包含技术指标的特征数据
        """
        features_df = pd.DataFrame(index=ohlcv_data.index)

        # 基本价格特征
        if "close" in ohlcv_data.columns:
            close = ohlcv_data["close"]

            # 收益率和波动率
            log_returns = self.calculate_log_returns(close)
            volatility = self.calculate_volatility(log_returns)

            features_df["log_return"] = log_returns
            features_df["volatility"] = volatility

            # 移动平均线特征
            ma_features = self.calculate_moving_averages(close)
            for name, series in ma_features.items():
                features_df[f"ma_{name}"] = series

            # RSI指标
            rsi = self.calculate_rsi(close)
            features_df["rsi"] = rsi

            # MACD指标
            macd_features = self.calculate_macd(close)
            for name, series in macd_features.items():
                features_df[f"macd_{name}"] = series

            # 布林带指标
            if "high" in ohlcv_data.columns and "low" in ohlcv_data.columns:
                bb_features = self.calculate_bollinger_bands(close)
                for name, series in bb_features.items():
                    features_df[f"bb_{name}"] = series

        # 成交量特征
        if "volume" in ohlcv_data.columns and "close" in ohlcv_data.columns:
            volume_features = self.calculate_volume_indicators(
                ohlcv_data["volume"], ohlcv_data["close"]
            )
            for name, series in volume_features.items():
                features_df[f"volume_{name}"] = series

        # 处理缺失值
        features_df = features_df.ffill().bfill().fillna(0)

        return features_df
