"""
配置模块单元测试
使用测试驱动开发方法，先写测试再实现配置模块
"""

import pytest
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# 导入要测试的配置模块
from src.config.config import HMMConfig


class TestHMMConfig:
    """HMM配置类单元测试"""

    def test_config_initialization(self):
        """测试配置类初始化"""
        # 当配置类被创建时
        config = HMMConfig()

        # 应该具有默认的股票代码
        assert hasattr(config, "stock_symbol")
        assert config.stock_symbol == "sh.601818"

        # 应该具有默认的开始日期
        assert hasattr(config, "start_date")
        assert config.start_date == "2010-01-01"

        # 应该具有默认的HMM参数
        assert hasattr(config, "n_states")
        assert config.n_states == 3

        assert hasattr(config, "covariance_type")
        assert config.covariance_type == "diag"

        assert hasattr(config, "n_iter")
        assert config.n_iter == 300

        assert hasattr(config, "tol")
        assert config.tol == 1e-4

        assert hasattr(config, "random_state")
        assert config.random_state is None

    def test_config_custom_initialization(self):
        """测试自定义参数初始化"""
        # 当使用自定义参数创建配置时
        custom_params = {
            "stock_symbol": "sh.000001",
            "start_date": "2015-01-01",
            "n_states": 4,
            "covariance_type": "full",
            "n_iter": 500,
            "tol": 1e-6,
            "random_state": 42,
        }

        config = HMMConfig(**custom_params)

        # 应该使用自定义参数而不是默认值
        assert config.stock_symbol == "sh.000001"
        assert config.start_date == "2015-01-01"
        assert config.n_states == 4
        assert config.covariance_type == "full"
        assert config.n_iter == 500
        assert config.tol == 1e-6
        assert config.random_state == 42

    def test_config_validation_valid(self):
        """测试配置参数验证（有效情况）"""
        # 当配置参数有效时
        config = HMMConfig()

        # 验证应该通过
        assert config.validate() is True
        assert config.validation_errors == []

    def test_config_validation_invalid_stock_symbol(self):
        """测试无效股票代码验证"""
        # 当股票代码格式无效时
        config = HMMConfig(stock_symbol="invalid_symbol")

        # 验证应该失败
        assert config.validate() is False
        assert len(config.validation_errors) > 0
        assert "股票代码" in str(config.validation_errors)

    def test_config_validation_invalid_date_format(self):
        """测试无效日期格式验证"""
        # 当日期格式无效时
        config = HMMConfig(start_date="2020-13-45")

        # 验证应该失败
        assert config.validate() is False
        assert len(config.validation_errors) > 0
        assert "日期格式" in str(config.validation_errors)

    def test_config_validation_invalid_n_states(self):
        """测试无效状态数量验证"""
        # 当状态数量无效时
        config = HMMConfig(n_states=1)  # 最少2个状态

        # 验证应该失败
        assert config.validate() is False
        assert len(config.validation_errors) > 0
        assert "状态数量" in str(config.validation_errors)

    def test_config_validation_invalid_covariance_type(self):
        """测试无效协方差类型验证"""
        # 当协方差类型无效时
        config = HMMConfig(covariance_type="invalid_type")

        # 验证应该失败
        assert config.validate() is False
        assert len(config.validation_errors) > 0
        assert "协方差类型" in str(config.validation_errors)

    def test_config_to_dict(self):
        """测试配置转换为字典"""
        # 当调用to_dict方法时
        config = HMMConfig()
        config_dict = config.to_dict()

        # 应该返回包含所有配置参数的字典
        assert isinstance(config_dict, dict)
        assert "stock_symbol" in config_dict
        assert "start_date" in config_dict
        assert "n_states" in config_dict
        assert "covariance_type" in config_dict
        assert "n_iter" in config_dict
        assert "tol" in config_dict
        assert "random_state" in config_dict

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        # 当使用from_dict方法时
        config_dict = {
            "stock_symbol": "sh.601398",
            "start_date": "2018-01-01",
            "n_states": 3,
            "covariance_type": "diag",
            "n_iter": 400,
            "tol": 1e-5,
            "random_state": 123,
        }

        config = HMMConfig.from_dict(config_dict)

        # 应该使用字典中的参数创建配置
        assert config.stock_symbol == "sh.601398"
        assert config.start_date == "2018-01-01"
        assert config.n_states == 3
        assert config.covariance_type == "diag"
        assert config.n_iter == 400
        assert config.tol == 1e-5
        assert config.random_state == 123

    def test_config_get_default(self):
        """测试获取默认配置"""
        # 当获取默认配置时
        default_config = HMMConfig.get_default()

        # 应该返回默认参数配置
        assert isinstance(default_config, HMMConfig)
        assert default_config.stock_symbol == "sh.601818"
        assert default_config.start_date == "2010-01-01"
        assert default_config.n_states == 3

    def test_config_str_representation(self):
        """测试配置的字符串表示"""
        # 当转换为字符串时
        config = HMMConfig()
        config_str = str(config)

        # 应该包含关键配置信息
        assert "HMMConfig" in config_str
        assert "sh.601818" in config_str
        assert "3" in config_str  # n_states

    def test_config_repr_representation(self):
        """测试配置的repr表示"""
        # 当调用repr时
        config = HMMConfig()
        config_repr = repr(config)

        # 应该包含类名和关键参数
        assert "HMMConfig" in config_repr
        assert "stock_symbol" in config_repr


class TestConfigFactory:
    """配置工厂类测试"""

    def test_create_config_from_environment(self):
        """测试从环境变量创建配置"""
        # 当环境变量设置时
        # 注意：这个测试需要环境变量支持，暂时标记为跳过
        pass

    def test_create_config_from_file(self):
        """测试从配置文件创建配置"""
        # 当配置文件存在时
        # 注意：这个测试需要配置文件支持，暂时标记为跳过
        pass


class TestConfigValidation:
    """配置验证功能测试"""

    def test_validate_stock_symbol(self):
        """测试股票代码验证"""
        # 有效的股票代码
        assert HMMConfig._validate_stock_symbol("sh.601818") is True
        assert HMMConfig._validate_stock_symbol("sz.000001") is True

        # 无效的股票代码
        assert HMMConfig._validate_stock_symbol("invalid") is False
        assert HMMConfig._validate_stock_symbol("601818") is False

    def test_validate_date_format(self):
        """测试日期格式验证"""
        # 有效的日期格式
        assert HMMConfig._validate_date_format("2020-01-01") is True
        assert HMMConfig._validate_date_format("2023-12-31") is True

        # 无效的日期格式
        assert HMMConfig._validate_date_format("2020-13-01") is False
        assert HMMConfig._validate_date_format("2020-01-32") is False
        assert HMMConfig._validate_date_format("01-01-2020") is False

    def test_validate_n_states(self):
        """测试状态数量验证"""
        # 有效的状态数量
        assert HMMConfig._validate_n_states(2) is True
        assert HMMConfig._validate_n_states(3) is True
        assert HMMConfig._validate_n_states(6) is True

        # 无效的状态数量
        assert HMMConfig._validate_n_states(1) is False
        assert HMMConfig._validate_n_states(0) is False
        assert HMMConfig._validate_n_states(-1) is False

    def test_validate_covariance_type(self):
        """测试协方差类型验证"""
        # 有效的协方差类型
        assert HMMConfig._validate_covariance_type("diag") is True
        assert HMMConfig._validate_covariance_type("full") is True
        assert HMMConfig._validate_covariance_type("tied") is True
        assert HMMConfig._validate_covariance_type("spherical") is True

        # 无效的协方差类型
        assert HMMConfig._validate_covariance_type("invalid") is False
        assert HMMConfig._validate_covariance_type("") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
