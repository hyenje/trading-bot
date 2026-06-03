"""
로깅 설정 유틸리티
"""
import os
import logging
from logging.handlers import RotatingFileHandler

from config import LOG_LEVEL, LOG_FILE, mask_sensitive

_HANDLER_MARK = "_crypto_trading_bot_logger"
_FILTER_MARK = "_crypto_trading_bot_filter"


class SensitiveLogFilter(logging.Filter):
    """마지막 로그 출력 직전에 민감한 query/token 값을 제거한다."""

    def filter(self, record):
        record.msg = mask_sensitive(record.getMessage())
        record.args = ()
        return True


def _add_sensitive_filter(handler):
    if any(getattr(item, _FILTER_MARK, False) for item in handler.filters):
        return
    filt = SensitiveLogFilter()
    setattr(filt, _FILTER_MARK, True)
    handler.addFilter(filt)


def setup_logger():
    """전역 로거 설정"""
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if any(getattr(handler, _HANDLER_MARK, False) for handler in root.handlers):
        for handler in root.handlers:
            if getattr(handler, _HANDLER_MARK, False):
                _add_sensitive_filter(handler)
        return root

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    _add_sensitive_filter(ch)
    setattr(ch, _HANDLER_MARK, True)
    root.addHandler(ch)

    # 파일 핸들러 (5MB 로테이션, 최대 5개)
    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    _add_sensitive_filter(fh)
    setattr(fh, _HANDLER_MARK, True)
    root.addHandler(fh)

    return root
