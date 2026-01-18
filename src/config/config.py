"""
HMM配置模块
集中管理所有应用参数，提供参数验证和默认值设置
支持从环境变量、配置文件、命令行参数等多种方式加载配置
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
import yaml
import re


class HMMConfig:
    """HMM系统配置类"""

    # 默认配置值
    DEFAULT_CONFIG = {
        "stock_symbol": "sh.601818",
        "start_date": "2010-01-01",
        "n_states": 3,
        "covariance_type": "diag",
        "n_iter": 300,
        "tol": 1e-4,
        "random_state": None,
        "min_duration": 10,
        "stickiness": 10.0,
        "data_source": "baostock",
        "cache_dir": "./cache",
        "cache_expiry_days": {"stock": 7, "macro": 30},
    }

    def __init__(self, **kwargs):
        """
        初始化配置参数

        Args:
            **kwargs: 配置参数，支持覆盖默认值
        """
        # 股票相关参数
        self.stock_symbol = kwargs.get(
            "stock_symbol", self.DEFAULT_CONFIG["stock_symbol"]
        )
        self.start_date = kwargs.get("start_date", self.DEFAULT_CONFIG["start_date"])

        # HMM模型参数
        self.n_states = kwargs.get("n_states", self.DEFAULT_CONFIG["n_states"])
        self.covariance_type = kwargs.get(
            "covariance_type", self.DEFAULT_CONFIG["covariance_type"]
        )
        self.n_iter = kwargs.get("n_iter", self.DEFAULT_CONFIG["n_iter"])
        self.tol = kwargs.get("tol", self.DEFAULT_CONFIG["tol"])
        self.random_state = kwargs.get(
            "random_state", self.DEFAULT_CONFIG["random_state"]
        )

        # 状态平滑参数
        self.min_duration = kwargs.get(
            "min_duration", self.DEFAULT_CONFIG["min_duration"]
        )
        self.stickiness = kwargs.get("stickiness", self.DEFAULT_CONFIG["stickiness"])

        # 数据源配置
        self.data_source = kwargs.get("data_source", self.DEFAULT_CONFIG["data_source"])

        # 缓存配置
        self.cache_dir = kwargs.get("cache_dir", self.DEFAULT_CONFIG["cache_dir"])
        self.cache_expiry_days = kwargs.get(
            "cache_expiry_days", self.DEFAULT_CONFIG["cache_expiry_days"]
        )

        # 验证错误信息
        self.validation_errors: List[str] = []

    def validate(self) -> bool:
        """
        验证配置参数的有效性

        Returns:
            bool: 验证是否通过
        """
        self.validation_errors = []

        # 验证股票代码
        if not self._validate_stock_symbol(self.stock_symbol):
            self.validation_errors.append(f"无效的股票代码格式: {self.stock_symbol}")

        # 验证日期格式
        if not self._validate_date_format(self.start_date):
            self.validation_errors.append(f"无效的日期格式: {self.start_date}")

        # 验证HMM参数
        if not self._validate_n_states(self.n_states):
            self.validation_errors.append(f"无效的状态数量: {self.n_states}")

        if not self._validate_covariance_type(self.covariance_type):
            self.validation_errors.append(f"无效的协方差类型: {self.covariance_type}")

        if not self._validate_positive_int(self.n_iter, "迭代次数"):
            self.validation_errors.append(f"无效的迭代次数: {self.n_iter}")

        if not self._validate_positive_float(self.tol, "收敛容忍度"):
            self.validation_errors.append(f"无效的收敛容忍度: {self.tol}")

        if not self._validate_positive_int(self.min_duration, "最小持续时间"):
            self.validation_errors.append(f"无效的最小持续时间: {self.min_duration}")

        if not self._validate_positive_float(self.stickiness, "粘性强度"):
            self.validation_errors.append(f"无效的粘性强度: {self.stickiness}")

        return len(self.validation_errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典

        Returns:
            Dict[str, Any]: 配置参数字典
        """
        return {
            "stock_symbol": self.stock_symbol,
            "start_date": self.start_date,
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "min_duration": self.min_duration,
            "stickiness": self.stickiness,
            "data_source": self.data_source,
            "cache_dir": self.cache_dir,
            "cache_expiry_days": self.cache_expiry_days,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HMMConfig":
        """
        从字典创建配置实例

        Args:
            config_dict: 配置参数字典

        Returns:
            HMMConfig: 配置实例
        """
        return cls(**config_dict)

    @classmethod
    def get_default(cls) -> "HMMConfig":
        """
        获取默认配置

        Returns:
            HMMConfig: 默认配置实例
        """
        return cls()

    @classmethod
    def from_env(cls) -> "HMMConfig":
        """
        从环境变量创建配置

        Returns:
            HMMConfig: 配置实例
        """
        config_dict = {}

        # 从环境变量读取配置
        env_mapping = {
            "HMM_STOCK_SYMBOL": "stock_symbol",
            "HMM_START_DATE": "start_date",
            "HMM_N_STATES": "n_states",
            "HMM_COVARIANCE_TYPE": "covariance_type",
            "HMM_N_ITER": "n_iter",
            "HMM_TOL": "tol",
            "HMM_RANDOM_STATE": "random_state",
            "HMM_MIN_DURATION": "min_duration",
            "HMM_STICKINESS": "stickiness",
            "HMM_DATA_SOURCE": "data_source",
            "HMM_CACHE_DIR": "cache_dir",
        }

        for env_var, config_key in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # 类型转换
                if config_key in ["n_states", "n_iter", "random_state", "min_duration"]:
                    try:
                        config_dict[config_key] = int(env_value)
                    except ValueError:
                        pass
                elif config_key in ["tol", "stickiness"]:
                    try:
                        config_dict[config_key] = float(env_value)
                    except ValueError:
                        pass
                else:
                    config_dict[config_key] = env_value

        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, filepath: str) -> "HMMConfig":
        """
        从JSON文件加载配置

        Args:
            filepath: JSON文件路径

        Returns:
            HMMConfig: 配置实例
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"无法加载配置文件 {filepath}: {e}")

    @classmethod
    def from_yaml_file(cls, filepath: str) -> "HMMConfig":
        """
        从YAML文件加载配置

        Args:
            filepath: YAML文件路径

        Returns:
            HMMConfig: 配置实例
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except (FileNotFoundError, yaml.YAMLError) as e:
            raise ValueError(f"无法加载配置文件 {filepath}: {e}")

    def save_to_json(self, filepath: str) -> None:
        """
        保存配置到JSON文件

        Args:
            filepath: 保存路径
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_to_yaml(self, filepath: str) -> None:
        """
        保存配置到YAML文件

        Args:
            filepath: 保存路径
        """
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def __str__(self) -> str:
        """字符串表示"""
        return f"HMMConfig(stock_symbol={self.stock_symbol}, n_states={self.n_states})"

    def __repr__(self) -> str:
        """repr表示"""
        return f"HMMConfig({self.to_dict()})"

    # 静态验证方法
    @staticmethod
    def _validate_stock_symbol(symbol: str) -> bool:
        """验证股票代码格式"""
        pattern = r"^(sh|sz)\.\d{6}$"
        return bool(re.match(pattern, symbol))

    @staticmethod
    def _validate_date_format(date_str: str) -> bool:
        """验证日期格式"""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    @staticmethod
    def _validate_n_states(n_states: int) -> bool:
        """验证状态数量"""
        return isinstance(n_states, int) and n_states >= 2

    @staticmethod
    def _validate_covariance_type(cov_type: str) -> bool:
        """验证协方差类型"""
        valid_types = ["diag", "full", "tied", "spherical"]
        return cov_type in valid_types

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> bool:
        """验证正整数"""
        return isinstance(value, int) and value > 0

    @staticmethod
    def _validate_positive_float(value: float, field_name: str) -> bool:
        """验证正浮点数"""
        return isinstance(value, (int, float)) and value > 0


# 配置工厂函数
def create_config_from_dict(config_dict: Dict[str, Any]) -> HMMConfig:
    """从字典创建配置"""
    return HMMConfig.from_dict(config_dict)


def get_default_config() -> HMMConfig:
    """获取默认配置"""
    return HMMConfig.get_default()
