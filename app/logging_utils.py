import logging
import os
from logging.handlers import RotatingFileHandler
from typing import List

from app.utils import get_project_logs_path


class EarlyBufferHandler(logging.Handler):
    """Early log buffer used before full logging configuration.

    Stores LogRecord objects (not just text) so they can be formatted
    correctly after real handlers/formatters are installed.
    """

    def __init__(self, level=logging.DEBUG, max_records: int = 1000):
        super().__init__(level=level)
        self.records: List[logging.LogRecord] = []
        self.max_records = max_records

    def emit(self, record: logging.LogRecord):
        self.records.append(record)
        if len(self.records) > self.max_records:
            # Keep only the last max_records
            self.records = self.records[-self.max_records:]

    def replay_to(self, target_logger: logging.Logger):
        """Replay buffered records through target_logger (usually root)."""
        for rec in self.records:
            # Use logger.handle to preserve level/format/filters
            target_logger.handle(rec)

        # Clear after replay to avoid duplication
        self.records.clear()

def setup_logging(config_manager):
    """Configure root logging based on config manager.

    Mirrors previous implementation from main.py but lives in logging_utils
    to decouple UI from main (CLI removed).
    """
    log_config = config_manager.get_logging_config()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Lowest level; handlers will filter

    root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_config['file']['enabled']:
        logs_dir = get_project_logs_path()
        log_file_path = os.path.join(logs_dir, log_config['file']['filename'])

        rotation_cfg = log_config['file'].get('rotation', {})
        use_rotation = rotation_cfg.get('enabled', False)

        if use_rotation:
            max_bytes = int(rotation_cfg.get('max_bytes', 1_048_576))
            backup_count = int(rotation_cfg.get('backup_count', 5))
            encoding = rotation_cfg.get('encoding', 'utf-8')
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding=encoding
            )
        else:
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')

        file_handler.setLevel(getattr(logging, log_config['level']))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {log_file_path} (rotation={'on' if use_rotation else 'off'})")

    if log_config['console']['enabled']:
        console_handler = logging.StreamHandler()
        console_level = log_config['console'].get('level', 'WARNING')
        console_handler.setLevel(getattr(logging, console_level))
        console_handler.setFormatter(formatter)

        user_cfg = log_config.get('user_messages', {})
        include_user = user_cfg.get('console_include', False)
        if not include_user:
            class ExcludeUserMessages(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    return not getattr(record, 'user_message', False)
            console_handler.addFilter(ExcludeUserMessages())

        root_logger.addHandler(console_handler)

def setup_exception_handler():
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            import sys
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger().error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    import sys
    sys.excepthook = exception_handler

__all__ = ["EarlyBufferHandler", "setup_logging", "setup_exception_handler"]
