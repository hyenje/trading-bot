#!/usr/bin/env python3
"""
Crypto Trading Bot - 메인 실행 스크립트

사용법:
    python main.py              # 봇 + 대시보드 실행
    python main.py --bot-only   # 봇만 실행 (대시보드 X)
    python main.py --backtest   # 백테스팅 실행
    python main.py --dashboard  # 대시보드만 실행
"""
import argparse
import threading
import logging
from datetime import datetime, timedelta
from typing import Optional

from utils.logger import setup_logger
from config import (
    BacktestConfig,
    BTCTrendLongShortConfig,
    BollingerConfig,
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    DRY_RUN,
    LONG_SHORT_ORDER_USDT,
    LONG_SHORT_STOP_LOSS_PCT,
    LONG_SHORT_TAKE_PROFIT_PCT,
    LONG_SHORT_TIMEFRAME,
    MAConfig,
    RSIConfig,
    TradingConfig,
    USE_TESTNET,
    is_configured,
    mask_sensitive,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--bot-only", action="store_true", help="봇만 실행")
    mode_group.add_argument("--dashboard", action="store_true", help="대시보드만 실행")
    mode_group.add_argument("--backtest", action="store_true", help="백테스팅 실행")
    mode_group.add_argument(
        "--backtest-long-short",
        action="store_true",
        help="BTC 롱/숏 전략 백테스팅 실행",
    )
    mode_group.add_argument(
        "--observe-long-short",
        action="store_true",
        help="BTC 롱/숏 시그널 관찰 대시보드 실행",
    )
    mode_group.add_argument(
        "--trade-long-short",
        action="store_true",
        help="BTC 롱/숏 Futures 테스트넷 주문 실행",
    )
    mode_group.add_argument(
        "--check-futures",
        action="store_true",
        help="Binance Futures 테스트넷 설정/API 점검",
    )
    mode_group.add_argument("--check", action="store_true", help="설정/API 점검")
    return parser.parse_args()


def build_bot(config: Optional[TradingConfig] = None):
    from bot import TradingBot

    return TradingBot(config or TradingConfig())


def build_backtest_strategies():
    from strategies import (
        BollingerStrategy,
        EnsembleStrategy,
        MACrossStrategy,
        RSIStrategy,
    )

    return {
        "MA Cross": MACrossStrategy(MAConfig()),
        "RSI": RSIStrategy(RSIConfig()),
        "Bollinger": BollingerStrategy(BollingerConfig()),
        "Ensemble": EnsembleStrategy(),
    }


def run_bot(bot):
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()


def run_bot_and_dashboard():
    """봇 + 대시보드 동시 실행"""
    from dashboard.app import run_dashboard, set_bot

    bot = build_bot()
    # 대시보드에 봇 인스턴스 주입
    set_bot(bot)

    # 대시보드를 별도 스레드에서 실행
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    logging.getLogger(__name__).info(
        f"대시보드 시작 → http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
    )

    # 봇 시작 (메인 스레드)
    run_bot(bot)


def run_bot_only():
    """봇만 실행"""
    run_bot(build_bot())


def run_dashboard_only():
    """대시보드만 실행"""
    from dashboard.app import run_dashboard

    logging.getLogger(__name__).info(
        f"대시보드만 실행 → http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
    )
    run_dashboard()


def run_long_short_observer():
    """BTC 롱/숏 시그널 관찰 대시보드 실행"""
    from dashboard.app import run_dashboard, set_bot
    from signal_observer import BTCSignalObserver

    observer = BTCSignalObserver()
    set_bot(observer)
    logging.getLogger(__name__).info(
        f"롱/숏 관찰 대시보드 시작 → http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
    )
    run_dashboard()


def run_long_short_trader():
    """BTC 롱/숏 Futures 테스트넷 주문 실행"""
    from dashboard.app import run_dashboard, set_bot
    from long_short_executor import BTCLongShortExecutor

    try:
        executor = BTCLongShortExecutor()
    except RuntimeError as e:
        logging.getLogger(__name__).error(str(e))
        raise SystemExit(1)

    executor.start()
    set_bot(executor)
    logging.getLogger(__name__).info(
        f"롱/숏 테스트넷 주문 대시보드 시작 → http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
    )
    run_dashboard()


def run_check() -> bool:
    """주문 없이 설정, 시세 조회, 잔고 인증을 점검"""
    from exchange import BinanceExchange

    logger = logging.getLogger(__name__)
    config = TradingConfig()
    keys_configured = is_configured(BINANCE_API_KEY) and is_configured(
        BINANCE_API_SECRET
    )

    logger.info("설정/API 점검 시작")
    logger.info(f"모드: {'테스트넷' if USE_TESTNET else '메인넷'}")
    logger.info(f"드라이런: {'ON' if DRY_RUN else 'OFF'}")
    logger.info(f"API 키 설정: {'예' if keys_configured else '아니오'}")
    logger.info(f"심볼: {config.symbols}")

    try:
        exchange = BinanceExchange()
    except Exception as e:
        logger.error(f"거래소 초기화 실패: {mask_sensitive(e)}")
        return False

    ok = True
    logger.info(f"마켓 로드: {len(exchange.exchange.markets)}개")

    for symbol in config.symbols:
        ticker = exchange.get_ticker(symbol)
        if ticker:
            logger.info(f"{symbol} 시세 조회 성공 | last={ticker.get('last')}")
        else:
            ok = False
            logger.error(f"{symbol} 시세 조회 실패")

    if not keys_configured:
        logger.warning("API 키가 없어 잔고 인증 점검은 건너뜀")
    else:
        try:
            balance = exchange.exchange.fetch_balance()
            free = balance.get("free", {}) if isinstance(balance, dict) else {}
            logger.info(
                "잔고 인증 성공 | USDT free=%s",
                free.get("USDT", 0.0),
            )
        except Exception as e:
            ok = False
            logger.error(f"잔고 인증 실패: {mask_sensitive(e)}")
            logger.error("테스트넷/메인넷 키 종류, IP 제한, Spot 읽기 권한을 확인하세요")

    logger.info(f"점검 결과: {'성공' if ok else '확인 필요'}")
    return ok


def run_futures_check() -> bool:
    """주문 없이 Futures 테스트넷 설정, 시세, 잔고 인증을 점검"""
    from exchange import BinanceFuturesExchange

    logger = logging.getLogger(__name__)
    symbol = "BTC/USDT"

    logger.info("Futures 테스트넷 점검 시작")
    logger.info(f"모드: {'테스트넷' if USE_TESTNET else '메인넷'}")
    logger.info(f"드라이런: {'ON' if DRY_RUN else 'OFF'}")

    try:
        exchange = BinanceFuturesExchange()
    except Exception as e:
        logger.error(f"Futures 초기화 실패: {mask_sensitive(e)}")
        return False

    ok = True
    logger.info(f"Futures 마켓 로드: {len(exchange.exchange.markets)}개")
    ticker = exchange.get_ticker(symbol)
    if ticker:
        logger.info(f"{symbol} Futures 시세 조회 성공 | last={ticker.get('last')}")
    else:
        logger.error(f"{symbol} Futures 시세 조회 실패")
        ok = False

    if exchange.check_private_access():
        logger.info(f"Futures 잔고 인증 성공 | USDT free={exchange.get_balance('USDT')}")
    else:
        logger.error("Futures 테스트넷 API Key/Secret 또는 권한을 확인하세요")
        ok = False

    logger.info(f"Futures 점검 결과: {'성공' if ok else '확인 필요'}")
    return ok


def run_backtest():
    """백테스팅 실행"""
    from exchange import BinanceExchange
    from backtesting.engine import BacktestEngine

    logger = logging.getLogger(__name__)
    logger.info("백테스팅 시작...")

    try:
        exchange = BinanceExchange()
    except Exception as e:
        logger.error(f"백테스팅 거래소 초기화 실패: {mask_sensitive(e)}")
        return
    bt_config = BacktestConfig()
    engine = BacktestEngine(bt_config)

    # 테스트할 전략 목록
    strategies_to_test = build_backtest_strategies()

    symbols = ["BTC/USDT", "ETH/USDT"]

    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"심볼: {symbol}")
        logger.info(f"{'='*50}")

        # 데이터 수집 (1시간 봉, 최대)
        df = exchange.fetch_ohlcv(symbol, "1h", limit=1000)
        if df.empty:
            logger.warning(f"{symbol} 데이터 수집 실패")
            continue

        logger.info(f"데이터: {df.index[0]} ~ {df.index[-1]} ({len(df)}봉)")

        for name, strategy in strategies_to_test.items():
            result = engine.run(
                df,
                strategy,
                symbol=symbol,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
            )
            logger.info(f"\n--- {name} ---")
            print(BacktestEngine.format_result(result))


def run_long_short_backtest():
    """BTC 롱/숏 전략 전용 백테스팅 실행"""
    from exchange import BinanceExchange
    from backtesting.engine import BacktestEngine
    from strategies import BTCTrendLongShortStrategy

    logger = logging.getLogger(__name__)
    symbol = "BTC/USDT"
    timeframe = LONG_SHORT_TIMEFRAME

    logger.info("BTC 롱/숏 백테스팅 시작...")
    try:
        exchange = BinanceExchange()
    except Exception as e:
        logger.error(f"BTC 롱/숏 백테스팅 거래소 초기화 실패: {mask_sensitive(e)}")
        return
    df = exchange.fetch_ohlcv_history(symbol, timeframe, total_limit=5000)
    if not df.empty and timeframe.endswith("m"):
        latest = df.index[-1]
        close_time = latest.to_pydatetime() + timedelta(minutes=int(timeframe[:-1]))
        if datetime.utcnow() < close_time:
            df = df.iloc[:-1]
    if df.empty:
        logger.warning(f"{symbol} 데이터 수집 실패")
        return

    buy_hold_pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    logger.info(f"데이터: {df.index[0]} ~ {df.index[-1]} ({len(df)}봉)")
    logger.info(f"Buy & Hold 수익률: {buy_hold_pct:.2f}%")
    logger.info(f"거래당 명목금액: ${LONG_SHORT_ORDER_USDT:.2f}")

    strategy = BTCTrendLongShortStrategy(BTCTrendLongShortConfig())
    result = BacktestEngine(BacktestConfig()).run(
        df,
        strategy,
        symbol=symbol,
        stop_loss_pct=LONG_SHORT_STOP_LOSS_PCT,
        take_profit_pct=LONG_SHORT_TAKE_PROFIT_PCT,
        allow_short=True,
        position_size_usdt=LONG_SHORT_ORDER_USDT,
    )

    long_count = sum(1 for trade in result.trades if trade.side == "long")
    short_count = sum(1 for trade in result.trades if trade.side == "short")
    logger.info(f"롱 거래: {long_count} | 숏 거래: {short_count}")
    print(BacktestEngine.format_result(result))


def main():
    args = parse_args()

    setup_logger()
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("  Crypto Trading Bot v1.0")
    logger.info("=" * 50)

    if args.check:
        raise SystemExit(0 if run_check() else 1)
    elif args.check_futures:
        raise SystemExit(0 if run_futures_check() else 1)
    elif args.backtest_long_short:
        run_long_short_backtest()
    elif args.trade_long_short:
        run_long_short_trader()
    elif args.observe_long_short:
        run_long_short_observer()
    elif args.backtest:
        run_backtest()
    elif args.bot_only:
        run_bot_only()
    elif args.dashboard:
        run_dashboard_only()
    else:
        run_bot_and_dashboard()


if __name__ == "__main__":
    main()
