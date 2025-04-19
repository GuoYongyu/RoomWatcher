# coding: utf-8
import os
import logging
import logging.handlers as handlers
from datetime import datetime

import colorlog


class CustomTimedRotatingFileHandler(handlers.TimedRotatingFileHandler):
    def __init__(
            self, 
            filename_format: str, 
            when: str = 'h', 
            interval: int = 1, 
            backupCount: int = 0, 
            encoding: str | None = None, 
            delay: bool = False, 
            utc: bool = False, 
            atTime: datetime | None = None
        ):
        # filename_format 是一个包含日期格式化占位符的字符串，例如 "my-log-%Y-%m-%d.log"
        self.filename_format = filename_format
        self.when = when
        self.interval = interval
        self.backupCount = backupCount
        self.encoding = encoding
        self.delay = delay
        self.utc = utc
        self.atTime = atTime
        self.extMatch = r"^\d{4}-\d{2}-\d{2}$"  # 用于匹配文件名后缀的正则表达式

        # 初始化时使用当前日期生成文件名
        base_filename = self._compute_filename()
        handlers.TimedRotatingFileHandler.__init__(
            self, base_filename, when, interval, backupCount, encoding, delay, utc, atTime
        )


    def _compute_filename(self):
        # 使用当前日期生成文件名
        return datetime.now().strftime(self.filename_format)
    

    def rotation_filename(self, default_name):
        # 重写 rotation_filename 方法以使用自定义文件名
        return self._compute_filename()


class Logger(object):
    def __init__(self, log_dir: str = None):
        self.logger = logging.getLogger('KitchenEnermies')

        # 输出到控制台
        console_handler = logging.StreamHandler()
        # 输出到文件
        log_dir = log_dir if log_dir else './logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # date = datetime.now().strftime("%Y-%m-%d")
        file_handler = CustomTimedRotatingFileHandler(
            filename_format=f'{log_dir}/%Y-%m-%d.log',
            encoding='utf-8',
            when="D",
            backupCount=7,
        )

        # 日志级别，logger 和 handler 以最高级别为准
        self.logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.INFO)

        # 日志颜色
        log_colors_config = {
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }

        # 日志输出格式
        file_formatter = logging.Formatter(
            fmt='[%(threadName)s] [%(asctime)s.%(msecs)03d] %(filename)s -> %(module)s.%(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(threadName)s] [%(asctime)s.%(msecs)03d] %(filename)s -> %(module)s.%(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
            # fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s : %(message)s\n',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors=log_colors_config
        )
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)


        # 防止重复添加 handler
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

        console_handler.close()
        file_handler.close()

    @staticmethod
    def clear_all_history_logs(log_dir: str = None):
        log_dir = log_dir if log_dir else './logs'
        if not os.path.exists(log_dir):
            return
        LOGGER.warning("Clearing all history logs ...")
        for root, _, files in os.walk(log_dir):
            for name in files:
                LOGGER.info(f"Log {os.path.join(root, name)} is cleared ...")
                os.remove(os.path.join(root, name))


LOGGER = Logger().logger
