"""
日志系统测试用例
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

# 导入日志系统（先使用mock，待实现后替换）
from src.utils.logger import Logger, LogLevel
from src.strategies.trading_strategy import TradingSignal, StrategyPerformance


class TestLogger:
    """日志系统测试类"""
    
    def setup_method(self):
        """测试前置方法"""
        # 创建临时日志文件
        self.temp_log_file = "/tmp/test_hmm_log.log"
        if os.path.exists(self.temp_log_file):
            os.remove(self.temp_log_file)
        
    def teardown_method(self):
        """测试后置方法"""
        # 清理临时日志文件
        if os.path.exists(self.temp_log_file):
            os.remove(self.temp_log_file)
    
    def test_logger_initialization(self):
        """测试日志记录器初始化"""
        # 测试默认初始化
        logger = Logger()
        assert logger is not None
        
        # 测试自定义配置初始化
        logger = Logger(
            log_file=self.temp_log_file,
            log_level=LogLevel.DEBUG,
            file_log_level=LogLevel.INFO,
            console_log_level=LogLevel.WARNING
        )
        assert logger is not None
    
    def test_log_market_state(self):
        """测试市场状态日志记录"""
        logger = Logger(log_file=self.temp_log_file, log_level=LogLevel.INFO)
        
        # 记录市场状态
        timestamp = pd.Timestamp(datetime.now())
        logger.log_market_state(timestamp, "Bull", confidence=0.85)
        
        # 验证日志文件存在
        assert os.path.exists(self.temp_log_file)
        
        # 验证日志内容
        with open(self.temp_log_file, 'r') as f:
            logs = f.read()
            assert "MARKET_STATE" in logs
            assert "Bull" in logs
            assert "confidence" in logs
            assert "0.85" in logs
    
    def test_log_trading_signal(self):
        """测试交易信号日志记录"""
        logger = Logger(log_file=self.temp_log_file, log_level=LogLevel.INFO)
        
        # 创建交易信号
        signal = TradingSignal(
            date=pd.Timestamp(datetime.now()),
            regime="Bear",
            position=-1,
            price=100.5
        )
        
        # 记录交易信号
        logger.log_trading_signal(signal)
        
        # 验证日志内容
        with open(self.temp_log_file, 'r') as f:
            logs = f.read()
            assert "TRADING_SIGNAL" in logs
            assert "Bear" in logs
            assert "position" in logs
            assert "-1" in logs
            assert "price" in logs
            assert "100.5" in logs
    
    def test_log_trade_execution(self):
        """测试交易执行日志记录"""
        logger = Logger(log_file=self.temp_log_file, log_level=LogLevel.INFO)
        
        # 记录交易执行
        timestamp = pd.Timestamp(datetime.now())
        logger.log_trade_execution(timestamp, "BUY", 100.5, 100)
        
        # 验证日志内容
        with open(self.temp_log_file, 'r') as f:
            logs = f.read()
            assert "TRADE_EXECUTION" in logs
            assert "BUY" in logs
            assert "price" in logs
            assert "100.5" in logs
            assert "quantity" in logs
            assert "100" in logs
    
    def test_log_performance(self):
        """测试策略表现日志记录"""
        logger = Logger(log_file=self.temp_log_file, log_level=LogLevel.INFO)
        
        # 创建策略表现对象
        performance = StrategyPerformance(
            total_return=0.15,
            cagr=0.08,
            sharpe=1.5,
            max_drawdown=-0.12,
            win_rate=0.6,
            profit_factor=1.8,
            num_trades=50
        )
        
        # 记录策略表现
        logger.log_performance(performance)
        
        # 验证日志内容
        with open(self.temp_log_file, 'r') as f:
            logs = f.read()
            assert "PERFORMANCE" in logs
            assert "total_return" in logs
            assert "0.15" in logs
            assert "cagr" in logs
            assert "0.08" in logs
            assert "sharpe" in logs
            assert "1.5" in logs
    
    def test_log_level_filtering(self):
        """测试日志级别过滤"""
        logger = Logger(
            log_file=self.temp_log_file,
            log_level=LogLevel.WARNING,  # 只记录WARNING及以上级别
            console_log_level=LogLevel.OFF  # 关闭控制台输出
        )
        
        # 记录不同级别的日志
        timestamp = pd.Timestamp(datetime.now())
        logger.log_market_state(timestamp, "Neutral", confidence=0.5)  # INFO级别
        logger._log(LogLevel.DEBUG, "This is a debug message")  # DEBUG级别
        logger._log(LogLevel.WARNING, "This is a warning message")  # WARNING级别
        
        # 验证只有WARNING级别日志被记录
        with open(self.temp_log_file, 'r') as f:
            logs = f.read()
            assert "WARNING" in logs
            assert "debug message" not in logs
            assert "MARKET_STATE" not in logs  # 因为INFO级别被过滤
    
    def test_log_format(self):
        """测试日志输出格式"""
        # 使用临时文件测试日志格式
        temp_file = "/tmp/test_log_format.log"
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        logger = Logger(log_file=temp_file, log_level=LogLevel.INFO, console_log_level=LogLevel.OFF)
        
        # 记录日志
        timestamp = pd.Timestamp(datetime.now())
        logger.log_market_state(timestamp, "Bull", confidence=0.85)
        
        # 验证日志内容和格式
        with open(temp_file, 'r') as f:
            log_line = f.read().strip()
            
        # 检查日志格式包含预期字段
        assert "MARKET_STATE" in log_line
        assert "Bull" in log_line
        assert "confidence" in log_line
        assert "0.85" in log_line
        assert "event_type" in log_line
        assert "timestamp" in log_line
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    def test_multiple_loggers(self):
        """测试多个日志记录器实例"""
        # 创建两个不同配置的日志记录器
        logger1 = Logger(log_file="/tmp/test_log1.log", log_level=LogLevel.INFO)
        logger2 = Logger(log_file="/tmp/test_log2.log", log_level=LogLevel.DEBUG)
        
        # 分别记录日志
        timestamp = pd.Timestamp(datetime.now())
        logger1.log_market_state(timestamp, "Bull")
        logger2.log_trade_execution(timestamp, "SELL", 99.5, 50)
        
        # 验证两个日志文件都存在且内容不同
        assert os.path.exists("/tmp/test_log1.log")
        assert os.path.exists("/tmp/test_log2.log")
        
        with open("/tmp/test_log1.log", 'r') as f1, open("/tmp/test_log2.log", 'r') as f2:
            logs1 = f1.read()
            logs2 = f2.read()
            assert logs1 != logs2
            assert "MARKET_STATE" in logs1
            assert "TRADE_EXECUTION" in logs2
        
        # 清理临时文件
        os.remove("/tmp/test_log1.log")
        os.remove("/tmp/test_log2.log")
    
    def test_invalid_log_level(self):
        """测试无效日志级别处理"""
        with pytest.raises(ValueError):
            logger = Logger(log_level="INVALID_LEVEL")
    
    def test_log_rotation(self):
        """测试日志滚动功能"""
        # 使用小文件大小测试滚动
        logger = Logger(
            log_file=self.temp_log_file,
            log_level=LogLevel.INFO,
            max_bytes=1024,  # 1KB
            backup_count=2
        )
        
        # 生成大量日志，触发滚动
        timestamp = pd.Timestamp(datetime.now())
        for i in range(100):
            logger.log_market_state(timestamp, f"State_{i % 3}", confidence=0.7 + i % 30 / 100)
        
        # 验证主日志文件存在
        assert os.path.exists(self.temp_log_file)
        
        # 验证至少生成了一个备份文件
        backup_files = [f for f in os.listdir("/tmp") if f.startswith("test_hmm_log.log.")]
        assert len(backup_files) >= 1
