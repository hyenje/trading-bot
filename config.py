"""
Crypto Trading Bot Configuration
바이낸스 자동 트레이딩 봇 설정 파일
"""
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

load_dotenv()

_TRUE_VALUES = {"1", "true", "yes", "on"}
_PLACEHOLDER_VALUES = {
    "",
    "YOUR_API_KEY_HERE",
    "YOUR_API_SECRET_HERE",
    "YOUR_FUTURES_API_KEY_HERE",
    "YOUR_FUTURES_API_SECRET_HERE",
    "YOUR_TELEGRAM_BOT_TOKEN",
    "YOUR_CHAT_ID",
    "your_api_key_here",
    "your_api_secret_here",
    "your_futures_api_key_here",
    "your_futures_api_secret_here",
    "your_telegram_bot_token",
    "your_chat_id",
}
_SENSITIVE_FIELD_PATTERN = re.compile(
    r"(?i)(^|[?&\s\"'])"
    r"((?:api[_-]?key|api[_-]?secret|secret|signature|timestamp|recvwindow|"
    r"token|access[_-]?token)[\"']?\s*[:=]\s*)"
    r"([^&\s,\"'}]+)"
)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_list(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name)
    if value is None:
        return default
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item] or default


def is_configured(value: Optional[str]) -> bool:
    """환경변수 값이 실제 설정값인지 확인"""
    return bool(value and value.strip() not in _PLACEHOLDER_VALUES)


# ============================================================
# API 설정 - 환경변수에서 읽거나 직접 입력
# ============================================================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_API_KEY_HERE").strip()
BINANCE_API_SECRET = os.getenv(
    "BINANCE_API_SECRET", "YOUR_API_SECRET_HERE"
).strip()
BINANCE_FUTURES_API_KEY = os.getenv("BINANCE_FUTURES_API_KEY", BINANCE_API_KEY).strip()
BINANCE_FUTURES_API_SECRET = os.getenv(
    "BINANCE_FUTURES_API_SECRET", BINANCE_API_SECRET
).strip()

# True = 테스트넷 (실제 돈 X), False = 실거래
USE_TESTNET = _get_env_bool("USE_TESTNET", True)

# True = 주문을 실제로 넣지 않고 체결된 것처럼 기록
DRY_RUN = _get_env_bool("DRY_RUN", True)

# USE_TESTNET=false + DRY_RUN=false 조합은 이 값까지 true여야 허용
ALLOW_LIVE_TRADING = _get_env_bool("ALLOW_LIVE_TRADING", False)

# DRY_RUN일 때 잔고 조회 대신 사용하는 가상 USDT 잔고
DRY_RUN_STARTING_BALANCE = _get_env_float("DRY_RUN_STARTING_BALANCE", 10000.0)


def mask_sensitive(value: object) -> str:
    """로그에 민감정보가 섞이지 않도록 마스킹"""
    text = str(value)
    for secret in (
        BINANCE_API_KEY,
        BINANCE_API_SECRET,
        BINANCE_FUTURES_API_KEY,
        BINANCE_FUTURES_API_SECRET,
    ):
        if is_configured(secret):
            text = text.replace(secret, "<masked>")
    return _SENSITIVE_FIELD_PATTERN.sub(r"\1<masked>=<masked>", text)


def validate_live_trading_allowed() -> None:
    """실거래 주문이 실수로 켜지는 것을 방지"""
    if not USE_TESTNET and not DRY_RUN and not ALLOW_LIVE_TRADING:
        raise RuntimeError(
            "실거래 주문 모드가 차단되었습니다. 실거래를 의도한 경우에만 "
            "ALLOW_LIVE_TRADING=true를 추가하세요. 안전 점검은 DRY_RUN=true "
            "또는 USE_TESTNET=true로 실행하세요."
        )


# ============================================================
# BTC 롱/숏 실행 설정 - 테스트넷 전용
# ============================================================
ENABLE_LONG_SHORT_EXECUTION = _get_env_bool("ENABLE_LONG_SHORT_EXECUTION", False)
LONG_SHORT_TIMEFRAME = os.getenv("LONG_SHORT_TIMEFRAME", "10m")
LONG_SHORT_ORDER_USDT = _get_env_float("LONG_SHORT_ORDER_USDT", 25.0)
LONG_SHORT_LEVERAGE = _get_env_int("LONG_SHORT_LEVERAGE", 1)
LONG_SHORT_POLL_INTERVAL = _get_env_int("LONG_SHORT_POLL_INTERVAL", 600)
LONG_SHORT_RISK_POLL_INTERVAL = _get_env_int("LONG_SHORT_RISK_POLL_INTERVAL", 180)
LONG_SHORT_REGIME_TIMEFRAME = os.getenv("BTC_LS_REGIME_TIMEFRAME", "4h")
LONG_SHORT_REQUIRE_REGIME_ALIGNMENT = _get_env_bool(
    "BTC_LS_REQUIRE_REGIME_ALIGNMENT", True
)
LONG_SHORT_REVERSE_ONLY_WHEN_PROFITABLE = _get_env_bool(
    "BTC_LS_REVERSE_ONLY_WHEN_PROFITABLE", True
)
LONG_SHORT_MIN_REVERSE_NET_PNL_USDT = _get_env_float(
    "BTC_LS_MIN_REVERSE_NET_PNL_USDT", 0.0
)
LONG_SHORT_FEE_RATE = _get_env_float("BTC_LS_FEE_RATE", 0.001)
LONG_SHORT_MAX_SIGNAL_AGE_MINUTES = _get_env_int(
    "LONG_SHORT_MAX_SIGNAL_AGE_MINUTES", 30
)
LONG_SHORT_ENABLE_SIGNAL_CATCHUP = _get_env_bool(
    "LONG_SHORT_ENABLE_SIGNAL_CATCHUP", False
)
LONG_SHORT_STOP_LOSS_PCT = _get_env_float("LONG_SHORT_STOP_LOSS_PCT", 2.0)
LONG_SHORT_TAKE_PROFIT_PCT = _get_env_float("LONG_SHORT_TAKE_PROFIT_PCT", 4.0)
LONG_SHORT_MAX_HOLD_BARS = _get_env_int("LONG_SHORT_MAX_HOLD_BARS", 0)
LONG_SHORT_BREAK_EVEN_AFTER_PCT = _get_env_float(
    "LONG_SHORT_BREAK_EVEN_AFTER_PCT", 0.0
)
LONG_SHORT_MAX_DAILY_LOSS_PCT = _get_env_float("LONG_SHORT_MAX_DAILY_LOSS_PCT", 3.0)
LONG_SHORT_MAX_DAILY_LOSS_USDT = _get_env_float("LONG_SHORT_MAX_DAILY_LOSS_USDT", 50.0)
LONG_SHORT_MAX_DAILY_TRADES = _get_env_int("LONG_SHORT_MAX_DAILY_TRADES", 10)
LONG_SHORT_MAX_CONSECUTIVE_LOSSES = _get_env_int(
    "LONG_SHORT_MAX_CONSECUTIVE_LOSSES", 3
)
LONG_SHORT_COOLDOWN_AFTER_LOSSES_MINUTES = _get_env_int(
    "LONG_SHORT_COOLDOWN_AFTER_LOSSES_MINUTES", 60
)


def validate_long_short_execution_allowed() -> None:
    """BTC 롱/숏 실제 주문은 테스트넷에서 명시적으로만 허용"""
    if not ENABLE_LONG_SHORT_EXECUTION:
        raise RuntimeError("ENABLE_LONG_SHORT_EXECUTION=true가 필요합니다.")
    if not USE_TESTNET:
        raise RuntimeError("롱/숏 실제 주문은 USE_TESTNET=true에서만 허용됩니다.")
    if DRY_RUN:
        raise RuntimeError("실제 테스트넷 주문을 위해 DRY_RUN=false가 필요합니다.")


# ============================================================
# 텔레그램 알림 설정
# ============================================================
TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN"
).strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID").strip()

# ============================================================
# 거래 기본 설정
# ============================================================
@dataclass
class TradingConfig:
    # 거래 대상 심볼 목록
    symbols: List[str] = field(
        default_factory=lambda: _get_env_list(
            "TRADING_SYMBOLS", ["BTC/USDT", "ETH/USDT"]
        )
    )

    # 기본 타임프레임
    timeframe: str = os.getenv("TRADING_TIMEFRAME", "1h")

    # 주문 당 투자 금액 (USDT)
    order_amount: float = _get_env_float("TRADING_ORDER_AMOUNT", 100.0)

    # 최대 동시 포지션 수
    max_positions: int = _get_env_int("TRADING_MAX_POSITIONS", 5)

    # 리스크 관리
    stop_loss_pct: float = _get_env_float("TRADING_STOP_LOSS_PCT", 2.0)
    take_profit_pct: float = _get_env_float("TRADING_TAKE_PROFIT_PCT", 4.0)
    max_daily_loss_pct: float = _get_env_float("TRADING_MAX_DAILY_LOSS_PCT", 5.0)

    # 봇 실행 간격 (초)
    check_interval: int = _get_env_int("TRADING_CHECK_INTERVAL", 60)


# ============================================================
# 전략별 설정
# ============================================================
@dataclass
class MAConfig:
    """이동평균 크로스 전략"""
    short_period: int = _get_env_int("MA_SHORT_PERIOD", 7)
    long_period: int = _get_env_int("MA_LONG_PERIOD", 25)
    signal_period: int = _get_env_int("MA_SIGNAL_PERIOD", 9)


@dataclass
class RSIConfig:
    """RSI 전략"""
    period: int = _get_env_int("RSI_PERIOD", 14)
    overbought: float = _get_env_float("RSI_OVERBOUGHT", 70.0)
    oversold: float = _get_env_float("RSI_OVERSOLD", 30.0)


@dataclass
class BollingerConfig:
    """볼린저 밴드 전략"""
    period: int = _get_env_int("BOLLINGER_PERIOD", 20)
    std_dev: float = _get_env_float("BOLLINGER_STD_DEV", 2.0)


@dataclass
class GridConfig:
    """그리드 트레이딩 전략"""
    grid_levels: int = _get_env_int("GRID_LEVELS", 10)
    grid_range_pct: float = _get_env_float("GRID_RANGE_PCT", 5.0)
    amount_per_grid: float = _get_env_float("GRID_AMOUNT_PER_LEVEL", 50.0)


@dataclass
class BTCTrendLongShortConfig:
    """BTC 롱/숏 추세 전략 설정"""
    fast_ema: int = _get_env_int("BTC_LS_FAST_EMA", 12)
    slow_ema: int = _get_env_int("BTC_LS_SLOW_EMA", 26)
    slope_period: int = _get_env_int("BTC_LS_SLOPE_PERIOD", 6)
    rsi_period: int = _get_env_int("BTC_LS_RSI_PERIOD", 14)
    long_rsi_min: float = _get_env_float("BTC_LS_LONG_RSI_MIN", 52.0)
    short_rsi_max: float = _get_env_float("BTC_LS_SHORT_RSI_MAX", 48.0)
    min_confidence: float = _get_env_float("BTC_LS_MIN_CONFIDENCE", 0.6)


@dataclass
class BTCRegimePullbackConfig:
    """BTC 4h regime + entry timeframe pullback/mean-reversion 전략 설정"""
    mode: str = os.getenv("BTC_RP_MODE", "combined")
    entry_timeframe: str = os.getenv("BTC_RP_ENTRY_TIMEFRAME", "15m")
    regime_timeframe: str = os.getenv("BTC_RP_REGIME_TIMEFRAME", "4h")
    fast_ema: int = _get_env_int("BTC_RP_FAST_EMA", 12)
    slow_ema: int = _get_env_int("BTC_RP_SLOW_EMA", 26)
    slope_period: int = _get_env_int("BTC_RP_SLOPE_PERIOD", 6)
    rsi_period: int = _get_env_int("BTC_RP_RSI_PERIOD", 14)
    bb_period: int = _get_env_int("BTC_RP_BB_PERIOD", 20)
    bb_std: float = _get_env_float("BTC_RP_BB_STD", 2.0)
    regime_min_gap_pct: float = _get_env_float("BTC_RP_REGIME_MIN_GAP_PCT", 0.003)
    pullback_rsi_long: float = _get_env_float("BTC_RP_PULLBACK_RSI_LONG", 35.0)
    pullback_rsi_short: float = _get_env_float("BTC_RP_PULLBACK_RSI_SHORT", 65.0)
    range_rsi_long: float = _get_env_float("BTC_RP_RANGE_RSI_LONG", 35.0)
    range_rsi_short: float = _get_env_float("BTC_RP_RANGE_RSI_SHORT", 65.0)
    trend_stop_loss_pct: float = _get_env_float("BTC_RP_TREND_STOP_LOSS_PCT", 1.0)
    trend_take_profit_pct: float = _get_env_float("BTC_RP_TREND_TAKE_PROFIT_PCT", 1.5)
    range_stop_loss_pct: float = _get_env_float("BTC_RP_RANGE_STOP_LOSS_PCT", 0.8)
    range_take_profit_pct: float = _get_env_float("BTC_RP_RANGE_TAKE_PROFIT_PCT", 1.0)
    range_max_hold_bars: int = _get_env_int("BTC_RP_RANGE_MAX_HOLD_BARS", 12)
    min_confidence: float = _get_env_float("BTC_RP_MIN_CONFIDENCE", 0.6)


# ============================================================
# 백테스팅 설정
# ============================================================
@dataclass
class BacktestConfig:
    initial_capital: float = _get_env_float("BACKTEST_INITIAL_CAPITAL", 10000.0)
    commission_rate: float = _get_env_float("BACKTEST_COMMISSION_RATE", 0.001)
    slippage_rate: float = _get_env_float("BACKTEST_SLIPPAGE_RATE", 0.0)
    start_date: str = os.getenv("BACKTEST_START_DATE", "2025-01-01")
    end_date: str = os.getenv("BACKTEST_END_DATE", "2026-03-31")


# ============================================================
# 대시보드 설정
# ============================================================
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = _get_env_int("DASHBOARD_PORT", 5000)
DASHBOARD_DEBUG = _get_env_bool("DASHBOARD_DEBUG", False)


# ============================================================
# 로깅 설정
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "logs/trading_bot.log")
