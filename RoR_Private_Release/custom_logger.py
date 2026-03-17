# custom_logger.py
import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional
import os
import json
"""
级别常量	数值	用途说明
DEBUG	10	最详细的调试信息，用于开发阶段的诊断（如变量值、执行路径）。
INFO	20	记录程序正常运行的关键事件（如服务启动、配置加载）。
WARNING	30	潜在问题警告，程序仍能正常运行（如磁盘空间不足、API调用重试）。
ERROR	40	严重错误，导致部分功能失效（如数据库连接失败、文件解析错误）。
CRITICAL	50	致命错误，程序即将终止（如系统资源耗尽、不可恢复的崩溃）。
"""
class StructuredLogger:
    def __init__(self, log_dir: str = "logs", max_size: int = 100, backup_count: int = 5):
        self.log_dir = log_dir
        self.max_size = max_size * 1024 * 1024  # 转换为MB
        self.backup_count = backup_count
        self._configure_logger()

    def _generate_filename(self) -> str:
        """动态生成带时间戳的日志文件名"""
        import time_config
        timestamp = time_config.APP_START_TIME # 统一时间
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        return f"{self.log_dir}/app_{timestamp}.log"

    def _configure_logger(self):
        """配置日志记录器：文件仅保存INFO及以上，控制台显示所有级别"""
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)

        # 主记录器配置
        self.logger = logging.getLogger("MainApp")
        self.logger.setLevel(logging.DEBUG)

        # 文件处理器（仅保存INFO及以上）
        file_handler = RotatingFileHandler(
            filename=self._generate_filename(),
            maxBytes=self.max_size,
            backupCount=self.backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)  # 设置保存log文件级别
        file_handler.addFilter(self._file_filter)
        file_handler.setFormatter(self._json_formatter())

        # 控制台处理器（显示所有级别）
        console_handler = logging.StreamHandler()  # 设置print终端内容级别
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _file_filter(self, record) -> bool:
        """文件日志过滤：仅允许非调试日志且包含特定关键字"""
        return (
                record.levelno >= logging.INFO
                and "sensitive" not in record.getMessage()
        )

    def _json_formatter(self) -> logging.Formatter:
        """JSON结构化日志格式"""

        class JsonFormatter(logging.Formatter):
            # def format(self, record):
            #     return json.dumps({
            #         "timestamp": self.formatTime(record),
            #         "level": record.levelname,
            #         "module": record.module,
            #         "message": record.getMessage()
            #     }, ensure_ascii=False)
            def format(self, record):
                """直接输出纯文本消息（无JSON格式）"""
                return record.getMessage()
        return JsonFormatter()

# 全局实例化
logger = StructuredLogger().logger