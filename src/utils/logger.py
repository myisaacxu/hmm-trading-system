"""
日志系统模块
负责记录市场状态、交易信号和交易过程
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from enum import Enum
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime

# 使用类型提示导入，避免循环导入
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.trading_strategy import TradingSignal, StrategyPerformance


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    OFF = 100  # 高于CRITICAL，用于关闭日志


class Logger:
    """日志记录器类
    
    提供专门的方法记录市场状态、交易信号、交易执行和策略表现
    """
    
    def __init__(
        self,
        name: str = "hmm_strategy",
        log_file: Optional[str] = None,
        log_level: LogLevel = LogLevel.INFO,
        file_log_level: Optional[LogLevel] = None,
        console_log_level: Optional[LogLevel] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_file: 日志文件路径，None表示不输出到文件
            log_level: 全局日志级别
            file_log_level: 文件日志级别，None表示使用全局级别
            console_log_level: 控制台日志级别，None表示使用全局级别
            max_bytes: 日志文件最大字节数，超过则滚动
            backup_count: 保留的备份日志文件数量
            log_format: 日志格式字符串
        """
        # 验证日志级别
        if not isinstance(log_level, LogLevel):
            raise ValueError(f"Invalid log_level: {log_level}")
        
        # 创建独立的日志记录器，确保每个实例有自己的处理器
        self.logger = logging.getLogger(f"{name}_{id(self)}")
        self.logger.setLevel(log_level.value)
        self.logger.handlers.clear()  # 清除可能存在的旧处理器
        self.logger.propagate = False  # 防止日志传播到父记录器
        
        # 日志格式
        formatter = logging.Formatter(log_format)
        
        # 添加文件处理器
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # 创建滚动文件处理器
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            # 设置文件日志级别
            file_level = file_log_level or log_level
            file_handler.setLevel(file_level.value)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # 添加控制台处理器
        if console_log_level != LogLevel.OFF:
            console_handler = logging.StreamHandler()
            
            # 设置控制台日志级别
            console_level = console_log_level or log_level
            console_handler.setLevel(console_level.value)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_market_state(
        self,
        timestamp: pd.Timestamp,
        state: str,
        confidence: Optional[float] = None,
        **kwargs: Any
    ):
        """
        记录市场状态
        
        Args:
            timestamp: 时间戳
            state: 市场状态名称
            confidence: 状态置信度
            **kwargs: 其他相关信息
        """
        log_data = {
            "event_type": "MARKET_STATE",
            "timestamp": timestamp.isoformat(),
            "state": state
        }
        
        if confidence is not None:
            log_data["confidence"] = confidence
        
        log_data.update(kwargs)
        
        self._log(LogLevel.INFO, log_data)
    
    def log_trading_signal(self, signal: 'TradingSignal', **kwargs: Any):
        """
        记录交易信号
        
        Args:
            signal: 交易信号对象
            **kwargs: 其他相关信息
        """
        log_data = {
            "event_type": "TRADING_SIGNAL",
            "timestamp": signal.date.isoformat(),
            "regime": signal.regime,
            "position": signal.position,
            "price": signal.price
        }
        
        log_data.update(kwargs)
        
        self._log(LogLevel.INFO, log_data)
    
    def log_trade_execution(
        self,
        date: pd.Timestamp,
        action: str,  # BUY, SELL, HOLD
        price: float,
        quantity: float,
        **kwargs: Any
    ):
        """
        记录交易执行
        
        Args:
            date: 交易日期
            action: 交易动作（BUY, SELL, HOLD）
            price: 交易价格
            quantity: 交易数量
            **kwargs: 其他相关信息
        """
        log_data = {
            "event_type": "TRADE_EXECUTION",
            "timestamp": date.isoformat(),
            "action": action,
            "price": price,
            "quantity": quantity
        }
        
        log_data.update(kwargs)
        
        self._log(LogLevel.INFO, log_data)
    
    def log_performance(self, performance: 'StrategyPerformance', **kwargs: Any):
        """
        记录策略表现
        
        Args:
            performance: 策略表现对象
            **kwargs: 其他相关信息
        """
        log_data = {
            "event_type": "PERFORMANCE",
            "timestamp": datetime.now().isoformat(),
            "total_return": performance.total_return,
            "cagr": performance.cagr,
            "sharpe": performance.sharpe,
            "max_drawdown": performance.max_drawdown,
            "win_rate": performance.win_rate,
            "profit_factor": performance.profit_factor,
            "num_trades": performance.num_trades
        }
        
        log_data.update(kwargs)
        
        self._log(LogLevel.INFO, log_data)
    
    def _log(self, level: LogLevel, message: Any):
        """
        内部日志记录方法
        
        Args:
            level: 日志级别
            message: 日志消息，可以是字符串或字典
        """
        if isinstance(message, dict):
            # 格式化字典为字符串
            log_msg = " ".join([f"{k}={v}" for k, v in message.items()])
        else:
            log_msg = str(message)
        
        # 根据级别记录日志
        if level == LogLevel.DEBUG:
            self.logger.debug(log_msg)
        elif level == LogLevel.INFO:
            self.logger.info(log_msg)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_msg)
        elif level == LogLevel.ERROR:
            self.logger.error(log_msg)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_msg)
    
    def debug(self, message: Any, **kwargs: Any):
        """记录DEBUG级别日志"""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: Any, **kwargs: Any):
        """记录INFO级别日志"""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: Any, **kwargs: Any):
        """记录WARNING级别日志"""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: Any, **kwargs: Any):
        """记录ERROR级别日志"""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: Any, **kwargs: Any):
        """记录CRITICAL级别日志"""
        self._log(LogLevel.CRITICAL, message, **kwargs)


# 创建默认日志记录器实例
# 可以通过修改默认配置来调整全局日志行为
def get_default_logger() -> Logger:
    """获取默认日志记录器"""
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    log_file = os.path.join(logs_dir, f"hmm_strategy_{datetime.now().strftime('%Y%m%d')}.log")
    
    return Logger(
        log_file=log_file,
        log_level=LogLevel.INFO,
        file_log_level=LogLevel.INFO,
        console_log_level=LogLevel.INFO
    )


# 全局日志记录器实例
global_logger = get_default_logger()
