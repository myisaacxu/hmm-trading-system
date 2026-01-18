"""
工具函数模块
提供通用的工具函数和日志设置
"""

import logging
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple, Union


# 日志相关函数
def setup_logging(
    log_level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """
    设置日志配置

    Args:
        log_level: 日志级别
        log_file: 日志文件路径（可选）

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger("cebbank_hmm")
    logger.setLevel(log_level)

    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 添加文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 防止日志传播到根记录器
    logger.propagate = False

    return logger


# 目录和文件相关函数
def create_directories(directories: Optional[List[str]] = None) -> None:
    """
    创建必要的目录结构

    Args:
        directories: 要创建的目录列表，如果为None则使用默认目录
    """
    default_dirs = ["models", "logs", "data/cache"]
    dirs_to_create = directories or default_dirs

    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)


# 包和依赖相关函数
def check_required_packages() -> List[str]:
    """
    检查必要的Python包是否已安装

    Returns:
        List[str]: 缺失的包列表
    """
    required_packages = [
        "pandas",
        "numpy",
        "hmmlearn",
        "plotly",
        "streamlit",
        "baostock",
        "akshare",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    return missing_packages


# 数据格式化函数
def format_percentage(value: float, decimals: int = 2) -> str:
    """
    格式化百分比数值

    Args:
        value: 要格式化的数值（小数形式，如0.05表示5%）
        decimals: 保留的小数位数

    Returns:
        str: 格式化后的百分比字符串
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, currency_symbol: str = "¥") -> str:
    """
    格式化货币数值

    Args:
        value: 要格式化的货币数值
        currency_symbol: 货币符号，默认为人民币符号¥

    Returns:
        str: 格式化后的货币字符串
    """
    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1e9:
        return f"{sign}{currency_symbol}{abs_value/1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"{sign}{currency_symbol}{abs_value/1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"{sign}{currency_symbol}{abs_value/1e3:.2f}K"
    else:
        return f"{sign}{currency_symbol}{abs_value:.2f}"


def calculate_time_period(
    start_date: Union[datetime, str], end_date: Union[datetime, str]
) -> str:
    """
    计算时间期间描述

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        str: 时间期间描述
    """
    # 转换字符串日期为datetime对象
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # 计算天数差
    days = (end_date - start_date).days

    if days >= 365:
        years = days / 365
        return f"{years:.1f}年"
    elif days >= 30:
        months = days / 30
        return f"{months:.1f}个月"
    elif days >= 1:
        return f"{days}天"
    else:
        return "不足1天"


# 数学计算函数
def safe_divide(
    numerator: Union[int, float],
    denominator: Union[int, float],
    default: Union[int, float] = 0.0,
) -> Union[int, float]:
    """
    安全的除法运算，避免除零错误

    Args:
        numerator: 分子
        denominator: 分母
        default: 当分母为0或无效值时返回的默认值

    Returns:
        Union[int, float]: 除法结果或默认值
    """
    if (
        denominator == 0
        or denominator is None
        or np.isnan(denominator)
        or np.isinf(denominator)
    ):
        return default
    return numerator / denominator


# 数据验证函数
def validate_dataframe(
    df: Optional[pd.DataFrame],
    required_columns: Optional[List[str]] = None,
    max_null_ratio: float = 0.1,
) -> Tuple[bool, str]:
    """
    验证DataFrame的完整性

    Args:
        df: 要验证的DataFrame
        required_columns: 必须包含的列列表
        max_null_ratio: 允许的最大缺失值比例（0-1之间）

    Returns:
        Tuple[bool, str]: (验证结果, 验证消息)
    """
    if df is None:
        return False, "DataFrame为None"

    if df.empty:
        return False, "DataFrame为空"

    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"缺少列: {', '.join(missing_columns)}"

    # 检查是否有过多的缺失值
    total_cells = len(df) * len(df.columns)
    null_count = df.isnull().sum().sum()
    null_ratio = null_count / total_cells

    if null_ratio > max_null_ratio:
        return False, f"缺失值比例过高: {format_percentage(null_ratio)}"

    return True, "数据验证通过"
