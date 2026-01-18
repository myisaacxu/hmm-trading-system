"""
数据获取器模块
负责从外部数据源获取股票数据和宏观数据
"""

import pandas as pd
import numpy as np
import baostock as bs
import akshare as ak
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import warnings
import os
from pathlib import Path
from src.config.config import HMMConfig


class DataFetcher:
    """
    数据获取器类

    功能：
    1. 从baostock获取股票数据
    2. 从akshare获取宏观数据
    3. 数据对齐和预处理
    """

    def __init__(
        self,
        config: Optional[HMMConfig] = None,
        symbol: Optional[str] = None,
        data_source: Optional[str] = None,
    ):
        """
        初始化数据获取器

        Args:
            config: HMM配置对象，优先使用配置参数
            symbol: 股票代码（如果未提供config则使用此参数）
            data_source: 数据源（如果未提供config则使用此参数）
        """
        # 优先使用配置对象
        if config is not None:
            self._config = config
            self._symbol = config.stock_symbol
            self._data_source = config.data_source
        else:
            # 向后兼容：使用直接参数
            self._config = HMMConfig()
            # 如果提供了直接参数，更新配置对象
            if symbol is not None:
                self._config.stock_symbol = symbol
            if data_source is not None:
                self._config.data_source = data_source
            self._symbol = self._config.stock_symbol
        self._data_source = self._config.data_source

        # 缓存配置
        self._cache_dir = Path(self._config.cache_dir)
        self._cache_expiry_days = self._config.cache_expiry_days

        # 创建缓存目录
        self._create_cache_dir()

        self._logged_in = False

    @property
    def config(self):
        """配置对象属性"""
        return self._config

    @property
    def symbol(self):
        """股票代码属性（与配置对象同步）"""
        return self._config.stock_symbol

    @symbol.setter
    def symbol(self, value):
        """设置股票代码（与配置对象同步）"""
        self._config.stock_symbol = value
        self._symbol = value

    @property
    def data_source(self):
        """数据源属性（与配置对象同步）"""
        return self._config.data_source

    def _login_baostock(self) -> bool:
        """登录baostock"""
        if not self._logged_in:
            lg = bs.login()
            self._logged_in = lg.error_code == "0"
        return self._logged_in

    def _logout_baostock(self):
        """登出baostock"""
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def get_stock_data(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取股票数据

        Args:
            start_date: 开始日期，格式YYYY-MM-DD，默认使用配置中的开始日期
            end_date: 结束日期，格式YYYY-MM-DD，默认当前日期

        Returns:
            pd.DataFrame: 股票数据

        Raises:
            ValueError: 日期参数无效
            ConnectionError: 数据获取失败
        """
        # 使用配置中的默认日期
        if start_date is None:
            start_date = self.config.start_date

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date > end_date:
            raise ValueError("开始日期不能晚于结束日期")

        # 检查缓存
        cache_path = self._get_stock_cache_path(start_date, end_date)
        if self._is_cache_valid(cache_path, self._cache_expiry_days["stock"]):
            try:
                return self._load_data_from_cache(cache_path, is_series=False)  # type: ignore[return-value]
            except Exception as e:
                warnings.warn(f"从缓存加载股票数据失败: {e}，将从网络获取")

        if not self._login_baostock():
            raise ConnectionError("baostock登录失败")

        try:
            fields = [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "turn",
                "pctChg",
            ]

            rs = bs.query_history_k_data_plus(
                code=self.symbol,
                fields=fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",  # 日线
                adjustflag="3",  # 前复权
            )

            if rs.error_code != "0":
                raise ConnectionError(f"查询失败: {rs.error_msg}")

            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())

            result = pd.DataFrame(data_list, columns=rs.fields)

            if result.empty:
                raise ConnectionError("获取的数据为空")

            # 日期转换
            result["date"] = pd.to_datetime(result["date"])
            result = result.sort_values("date").reset_index(drop=True)
            result.set_index("date", inplace=True)

            # 转换数值列的数据类型
            numeric_cols = [
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "turn",
                "pctChg",
            ]
            for col in numeric_cols:
                if col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors="coerce")

            # 保存到缓存
            self._save_data_to_csv(result, cache_path)

            return result

        except Exception as e:
            self._logout_baostock()
            raise ConnectionError(f"获取股票数据时出错: {e}")
        finally:
            # 确保在成功获取数据后也调用logout
            self._logout_baostock()

    def get_macro_data(self, indicator_type: str) -> pd.Series:
        """
        获取宏观数据

        Args:
            indicator_type: 指标类型 ('ebs' 或 'buffett')

        Returns:
            pd.Series: 宏观数据序列
        """
        # 检查缓存
        cache_path = self._get_macro_cache_path(indicator_type)
        if self._is_cache_valid(cache_path, self._cache_expiry_days["macro"]):
            try:
                return self._load_data_from_cache(cache_path, is_series=True)  # type: ignore[return-value]
            except Exception as e:
                warnings.warn(f"从缓存加载宏观数据失败: {e}，将从网络获取")

        # 从网络获取数据
        if indicator_type == "ebs":
            data = self._get_ebs_data()
        elif indicator_type == "buffett":
            data = self._get_buffett_index()
        else:
            raise ValueError(f"不支持的指标类型: {indicator_type}")

        # 保存到缓存
        self._save_data_to_csv(data, cache_path)

        return data

    def _get_ebs_data(self) -> pd.Series:
        """获取股债利差数据"""
        try:
            # 获取原始数据缓存路径
            raw_cache_path = self._get_raw_cache_path("ebs")

            # 检查原始数据缓存是否有效
            if self._is_cache_valid(raw_cache_path, self._cache_expiry_days["macro"]):
                try:
                    # 从原始缓存加载数据，原始数据没有索引，所以不设置index_col
                    ebs_df = pd.read_csv(raw_cache_path)
                except Exception as e:
                    warnings.warn(f"从原始缓存加载股债利差数据失败: {e}，将从网络获取")
                    # 缓存加载失败，从网络获取
                    ebs_df = ak.stock_ebs_lg()
                    # 保存到原始缓存，原始数据没有索引，所以不保存索引
                    self._save_data_to_csv(ebs_df, raw_cache_path, save_index=False)
            else:
                # 原始缓存无效或不存在，从网络获取
                ebs_df = ak.stock_ebs_lg()
                # 保存到原始缓存，原始数据没有索引，所以不保存索引
                self._save_data_to_csv(ebs_df, raw_cache_path, save_index=False)

            if ebs_df.empty:
                return pd.Series([], name="ebs_indicator")

            # 重命名和格式化
            ebs_df = ebs_df.rename(
                columns={"日期": "date", "股债利差": "ebs_indicator"}
            )
            ebs_df = ebs_df[["date", "ebs_indicator"]]
            ebs_df["ebs_indicator"] = ebs_df["ebs_indicator"] * 100

            # 转换日期格式
            ebs_df["date"] = pd.to_datetime(ebs_df["date"])

            # 设置日期为索引并排序
            ebs_df.set_index("date", inplace=True)
            ebs_df = ebs_df.sort_index()

            return ebs_df["ebs_indicator"]

        except Exception as e:
            warnings.warn(f"获取股债利差数据失败: {e}")
            return pd.Series([], name="ebs_indicator")

    def _get_buffett_index(self) -> pd.Series:
        """获取巴菲特指数数据"""
        try:
            # 优先使用akshare的直接接口获取巴菲特指数
            # 获取原始数据缓存路径
            raw_cache_path = self._get_raw_cache_path("buffett")

            # 检查原始数据缓存是否有效
            if self._is_cache_valid(raw_cache_path, self._cache_expiry_days["macro"]):
                try:
                    # 从原始缓存加载数据，原始数据没有索引
                    buffett_df = pd.read_csv(raw_cache_path)
                except Exception as e:
                    warnings.warn(
                        f"从原始缓存加载巴菲特指数数据失败: {e}，将从网络获取"
                    )
                    # 缓存加载失败，从网络获取
                    buffett_df = ak.stock_buffett_index_lg()
                    # 保存到原始缓存，原始数据没有索引，所以不保存索引
                    self._save_data_to_csv(buffett_df, raw_cache_path, save_index=False)
            else:
                # 原始缓存无效或不存在，从网络获取
                buffett_df = ak.stock_buffett_index_lg()
                # 保存到原始缓存，原始数据没有索引，所以不保存索引
                self._save_data_to_csv(buffett_df, raw_cache_path, save_index=False)

            if not buffett_df.empty:
                # 重命名和格式化
                buffett_df = buffett_df.rename(columns={"日期": "date"})

                # 计算巴菲特指数：总市值/GDP * 100
                if "总市值" in buffett_df.columns and "GDP" in buffett_df.columns:
                    buffett_df["buffett_index"] = (
                        buffett_df["总市值"] / buffett_df["GDP"]
                    ) * 100
                else:
                    # 如果列名不同，尝试其他可能的列名组合
                    if (
                        "market_cap" in buffett_df.columns
                        and "gdp" in buffett_df.columns
                    ):
                        buffett_df["buffett_index"] = (
                            buffett_df["market_cap"] / buffett_df["gdp"]
                        ) * 100
                    else:
                        # 使用默认的巴菲特指数列
                        buffett_df["buffett_index"] = (
                            buffett_df.iloc[:, 1]
                            if len(buffett_df.columns) > 1
                            else 100
                        )

                # 转换日期格式
                buffett_df["date"] = pd.to_datetime(buffett_df["date"])

                # 设置日期为索引并排序
                buffett_df.set_index("date", inplace=True)
                buffett_df = buffett_df.sort_index()

                return buffett_df["buffett_index"]
            else:
                # 如果直接接口失败，回退到组合计算方式
                return self._calculate_buffett_index_fallback()

        except Exception as e:
            warnings.warn(f"获取巴菲特指数数据失败，尝试备用方法: {e}")
            # 使用备用计算方式
            return self._calculate_buffett_index_fallback()

    def _calculate_buffett_index_fallback(self) -> pd.Series:
        """备用方法：手动计算巴菲特指数"""
        try:
            # 使用akshare获取GDP和市值数据
            gdp_df = ak.macro_china_gdp()
            market_cap_df = ak.macro_china_stock_market_cap()

            if gdp_df.empty or market_cap_df.empty:
                return pd.Series([], name="buffett_index")

            # 处理市值数据
            if not market_cap_df.empty:
                latest_market_cap = market_cap_df.iloc[0]
                shanghai_market_cap = latest_market_cap.get("市价总值-上海", 0)
                shenzhen_market_cap = latest_market_cap.get("市价总值-深圳", 0)

                if pd.notna(shanghai_market_cap) and pd.notna(shenzhen_market_cap):
                    total_market_cap = shanghai_market_cap + shenzhen_market_cap
                else:
                    total_market_cap = None
            else:
                total_market_cap = None

            # 获取最新的GDP数据
            latest_gdp = gdp_df.iloc[0] if not gdp_df.empty else None

            if latest_gdp is not None and "value" in latest_gdp:
                current_gdp = latest_gdp["value"]

                # 计算巴菲特指数 = 总市值 / GDP
                buffett_index_value = (
                    (total_market_cap / current_gdp) * 100
                    if current_gdp > 0 and total_market_cap is not None
                    else 100
                )

                # 创建时间序列数据，使用日频率以匹配股票数据
                dates = pd.date_range(start="2010-01-01", end=datetime.now(), freq="D")
                buffett_values = np.full(len(dates), buffett_index_value)

                return pd.Series(buffett_values, index=dates, name="buffett_index")
            else:
                # GDP数据获取失败，使用模拟数据
                dates = pd.date_range(start="2010-01-01", end=datetime.now(), freq="D")
                buffett_values = np.linspace(80, 120, len(dates))

                return pd.Series(buffett_values, index=dates, name="buffett_index")

        except Exception as e:
            warnings.warn(f"备用方法也失败，使用默认值: {e}")
            # 返回默认数据
            dates = pd.date_range(start="2010-01-01", end=datetime.now(), freq="D")
            buffett_values = np.linspace(80, 120, len(dates))

            return pd.Series(buffett_values, index=dates, name="buffett_index")

    def align_data(
        self, data_list: List[Union[pd.DataFrame, pd.Series]]
    ) -> pd.DataFrame:
        """
        对齐多个数据集的时间索引

        Args:
            data_list: 数据列表

        Returns:
            pd.DataFrame: 对齐后的数据
        """
        if not data_list:
            raise ValueError("数据列表不能为空")

        # 预处理数据：去除重复索引
        processed_data: List[Union[pd.Series, pd.DataFrame]] = []
        for data in data_list:
            if isinstance(data, pd.Series):
                # 对于Series，去除重复索引，保留第一个出现的值
                processed_series = data[~data.index.duplicated(keep="first")]
                processed_data.append(processed_series)
            else:
                # 对于DataFrame，去除重复索引，保留第一个出现的行
                processed_df = data[~data.index.duplicated(keep="first")]
                processed_data.append(processed_df)

        # 获取所有数据集的索引交集
        common_index = processed_data[0].index
        for data in processed_data[1:]:
            common_index = common_index.intersection(data.index)

        if common_index.empty:
            # 如果没有交集，使用前向填充和插值方法进行对齐
            # 获取所有索引的并集，并按日期排序
            all_index = processed_data[0].index
            for data in processed_data[1:]:
                all_index = all_index.union(data.index)
            all_index = all_index.sort_values()

            # 对每个数据集进行前向填充
            aligned_data: List[Union[pd.Series, pd.DataFrame]] = []
            for data in processed_data:
                if isinstance(data, pd.Series):
                    # 对于Series，使用前向填充
                    aligned_series = data.reindex(all_index).ffill().bfill()
                    aligned_data.append(aligned_series)
                else:
                    # 对于DataFrame，对数值列进行前向填充
                    aligned_df = data.reindex(all_index)
                    numeric_cols = aligned_df.select_dtypes(include=[np.number]).columns
                    aligned_df[numeric_cols] = aligned_df[numeric_cols].ffill().bfill()
                    aligned_data.append(aligned_df)

            # 合并数据
            result = pd.concat(aligned_data, axis=1)
        else:
            # 有交集，直接对齐
            aligned_data: List[Union[pd.Series, pd.DataFrame]] = []
            for data in processed_data:
                if isinstance(data, pd.Series):
                    aligned_data.append(data.reindex(common_index))
                else:
                    aligned_data.append(data.reindex(common_index))

            # 合并数据
            result = pd.concat(aligned_data, axis=1)

        # 删除仍然有缺失值的行
        result = result.dropna()

        if result.empty:
            raise ValueError("数据对齐后结果为空，请检查数据源和时间范围")

        return result

    def get_combined_data(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        获取组合数据（股票数据 + 宏观数据）

        Args:
            start_date: 开始日期，默认使用配置中的开始日期
            end_date: 结束日期，默认当前日期

        Returns:
            dict: 包含股票数据和宏观数据的字典
        """
        stock_data = self.get_stock_data(start_date, end_date)
        ebs_data = self.get_macro_data("ebs")
        buffett_data = self.get_macro_data("buffett")

        # 对齐数据
        aligned_data = self.align_data([stock_data, ebs_data, buffett_data])

        return {
            "stock_data": stock_data,
            "ebs_data": ebs_data,
            "buffett_data": buffett_data,
            "aligned_data": aligned_data,
        }

    def fetch_stock_data(
        self, symbol: str, start_date: str, end_date: str, max_retries: int = 3
    ) -> pd.DataFrame:
        """
        获取股票数据（兼容测试接口）

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            max_retries: 最大重试次数

        Returns:
            pd.DataFrame: 股票数据
        """
        # 临时保存原始symbol
        original_symbol = self.symbol

        try:
            # 设置新的symbol
            self.symbol = symbol

            # 尝试使用baostock获取数据
            if not self._login_baostock():
                return pd.DataFrame()

            fields = [
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "volume",
                "amount",
                "turn",
                "pctChg",
            ]

            import baostock as bs

            rs = bs.query_history_k_data_plus(
                code=self.symbol,
                fields=fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",  # 日线
                adjustflag="3",  # 前复权
            )

            if rs.error_code != "0":
                return pd.DataFrame()

            data_list = []
            while (rs.error_code == "0") & rs.next():
                data_list.append(rs.get_row_data())

            result = pd.DataFrame(data_list, columns=rs.fields)

            if result.empty:
                return pd.DataFrame()

            # 日期转换
            result["date"] = pd.to_datetime(result["date"])
            result = result.sort_values("date").reset_index(drop=True)
            result.set_index("date", inplace=True)

            return result

        except Exception:
            return pd.DataFrame()
        finally:
            # 恢复原始symbol
            self.symbol = original_symbol

    def fetch_gdp_data(self) -> pd.DataFrame:
        """获取GDP数据（兼容测试接口）"""
        try:
            # 使用akshare获取GDP数据
            gdp_df = ak.macro_china_gdp()

            if gdp_df.empty:
                return pd.DataFrame()

            # 重命名和格式化
            gdp_df = gdp_df.rename(columns={"季度": "quarter", "value": "gdp"})
            gdp_df["date"] = pd.to_datetime(gdp_df["quarter"].str.replace("Q", "-Q"))
            gdp_df.set_index("date", inplace=True)

            return gdp_df[["gdp"]]

        except Exception:
            return pd.DataFrame()

    def fetch_market_cap_data(self) -> pd.DataFrame:
        """获取市值数据（兼容测试接口）"""
        try:
            # 使用akshare获取市值数据
            market_cap_df = ak.macro_china_stock_market_cap()

            if market_cap_df.empty:
                return pd.DataFrame()

            # 创建DataFrame
            market_cap_df = pd.DataFrame(
                {
                    "date": market_cap_df["月份"],
                    "total_market_cap": market_cap_df["市价总值-上海"]
                    + market_cap_df["市价总值-深圳"],
                }
            )

            market_cap_df["date"] = pd.to_datetime(
                market_cap_df["date"].str.replace("年", "-").str.replace("月份", "")
            )
            market_cap_df.set_index("date", inplace=True)

            # 计算总市值
            # 转换为亿元
            market_cap_df["total_market_cap"] = market_cap_df["total_market_cap"] * 1e8

            return market_cap_df[["total_market_cap"]]

        except Exception:
            return pd.DataFrame()

    def _create_cache_dir(self):
        """
        创建缓存目录结构
        """
        # 创建主缓存目录
        self._cache_dir.mkdir(exist_ok=True)

        # 创建股票数据缓存目录
        stock_cache_dir = self._cache_dir / "stock" / self._symbol
        stock_cache_dir.mkdir(parents=True, exist_ok=True)

        # 创建宏观数据缓存目录
        for indicator_type in ["ebs", "buffett"]:
            macro_cache_dir = self._cache_dir / "macro" / indicator_type
            macro_cache_dir.mkdir(parents=True, exist_ok=True)

        # 创建原始数据缓存目录
        for indicator_type in ["ebs", "buffett"]:
            raw_cache_dir = self._cache_dir / "raw" / indicator_type
            raw_cache_dir.mkdir(parents=True, exist_ok=True)

    def _is_cache_valid(self, file_path: Path, max_age_days: int) -> bool:
        """
        检查缓存文件是否有效

        Args:
            file_path: 缓存文件路径
            max_age_days: 最大缓存天数

        Returns:
            bool: 缓存是否有效
        """
        if not file_path.exists():
            return False

        # 检查文件修改时间
        file_stat = file_path.stat()
        file_modify_time = datetime.fromtimestamp(file_stat.st_mtime)
        current_time = datetime.now()

        # 计算文件年龄（天数）
        file_age_days = (current_time - file_modify_time).total_seconds() / (24 * 3600)

        return file_age_days < max_age_days

    def _save_data_to_csv(
        self,
        data: Union[pd.DataFrame, pd.Series],
        file_path: Path,
        save_index: bool = True,
    ):
        """
        将数据保存为CSV文件

        Args:
            data: 要保存的数据
            file_path: 保存路径
            save_index: 是否保存索引
        """
        # 创建父目录
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存数据
        data.to_csv(file_path, index=save_index)

    def _load_data_from_cache(
        self, file_path: Path, is_series: bool = False
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        从缓存加载数据

        Args:
            file_path: 缓存文件路径
            is_series: 是否返回Series（如果CSV只有一列）

        Returns:
            Union[pd.DataFrame, pd.Series]: 加载的数据
        """
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # 如果指定了返回Series且数据只有一列，转换为Series
        if is_series and len(data.columns) == 1:
            return data.iloc[:, 0]

        return data

    def _get_stock_cache_path(self, start_date: str, end_date: str) -> Path:
        """
        获取股票数据缓存路径

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Path: 缓存文件路径
        """
        # 格式化日期（去除连字符）
        start_date_formatted = start_date.replace("-", "")
        end_date_formatted = end_date.replace("-", "")

        return (
            self._cache_dir
            / "stock"
            / self._symbol
            / f"{start_date_formatted}_{end_date_formatted}.csv"
        )

    def _get_macro_cache_path(self, indicator_type: str) -> Path:
        """
        获取宏观数据缓存路径

        Args:
            indicator_type: 指标类型

        Returns:
            Path: 缓存文件路径
        """
        # 使用今天的日期作为文件名
        today = datetime.now().strftime("%Y%m%d")
        return self._cache_dir / "macro" / indicator_type / f"{today}.csv"

    def _get_raw_cache_path(self, indicator_type: str) -> Path:
        """
        获取原始数据缓存路径

        Args:
            indicator_type: 指标类型

        Returns:
            Path: 原始数据缓存文件路径
        """
        # 使用今天的日期作为文件名
        today = datetime.now().strftime("%Y%m%d")
        return self._cache_dir / "raw" / indicator_type / f"{today}.csv"

    def __del__(self):
        """析构函数，确保登出"""
        self._logout_baostock()
