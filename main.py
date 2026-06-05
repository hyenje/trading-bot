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
    BTCRegimePullbackConfig,
    BTCTrendLongShortConfig,
    BollingerConfig,
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    DRY_RUN,
    LONG_SHORT_BREAK_EVEN_AFTER_PCT,
    LONG_SHORT_MAX_HOLD_BARS,
    LONG_SHORT_MIN_REVERSE_NET_PNL_USDT,
    LONG_SHORT_ORDER_USDT,
    LONG_SHORT_REGIME_TIMEFRAME,
    LONG_SHORT_REQUIRE_REGIME_ALIGNMENT,
    LONG_SHORT_REVERSE_ONLY_WHEN_PROFITABLE,
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
        "--backtest-regime-pullback",
        action="store_true",
        help="BTC 4h regime + RSI/BB pullback 전략 백테스팅 실행",
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
    import ccxt

    from exchange import _maybe_resample_ohlcv, _ohlcv_frame, _source_ohlcv_request
    from backtesting.engine import BacktestEngine
    from strategies import BTCTrendLongShortStrategy

    logger = logging.getLogger(__name__)
    symbol = "BTC/USDT"
    timeframe = LONG_SHORT_TIMEFRAME

    logger.info("BTC 롱/숏 백테스팅 시작...")
    try:
        df = _fetch_usdm_ohlcv_history(
            ccxt.binanceusdm,
            symbol,
            timeframe,
            total_limit=5000,
            helpers=(_source_ohlcv_request, _ohlcv_frame, _maybe_resample_ohlcv),
        )
    except Exception as e:
        logger.error(f"BTC 롱/숏 백테스팅 데이터 수집 실패: {mask_sensitive(e)}")
        return
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
    logger.info("백테스트 데이터: Binance USD-M Futures public OHLCV")

    base_config = BTCTrendLongShortConfig()
    mtf_options = {
        "regime_timeframe": LONG_SHORT_REGIME_TIMEFRAME,
        "require_regime_alignment": LONG_SHORT_REQUIRE_REGIME_ALIGNMENT,
        "reverse_only_when_profitable": LONG_SHORT_REVERSE_ONLY_WHEN_PROFITABLE,
        "min_reverse_net_pnl_usdt": LONG_SHORT_MIN_REVERSE_NET_PNL_USDT,
        "max_hold_bars": LONG_SHORT_MAX_HOLD_BARS,
        "break_even_after_pct": LONG_SHORT_BREAK_EVEN_AFTER_PCT,
    }
    scenarios = [
        ("current: 10m/4h fee reverse", base_config, mtf_options),
        ("baseline: raw executor-like", base_config, {"allowed_sides": "both"}),
        ("long-only", base_config, {"allowed_sides": "long"}),
        ("short-only", base_config, {"allowed_sides": "short"}),
    ]
    experiments = [
        ("exp: long RSI >= 58", _long_short_config(long_rsi_min=58.0), {}),
        ("exp: max hold 6 bars", base_config, {"max_hold_bars": 6}),
        ("exp: max hold 12 bars", base_config, {"max_hold_bars": 12}),
        ("exp: 1h trend filter", base_config, {"higher_timeframe": "1h"}),
        ("exp: min gap 0.05%", base_config, {"min_trend_gap": 0.0005}),
        ("exp: min slope 25", base_config, {"min_ema_slope": 25.0}),
    ]

    rows = []
    for name, config, options in scenarios + experiments:
        result = _run_long_short_backtest_case(
            df,
            symbol,
            config,
            **options,
        )
        rows.append((name, result))

    default_result = rows[0][1]
    logger.info(
        "기본 MTF 결과 | 총 거래 %s | 롱 %s | 숏 %s | Net PnL $%.2f",
        default_result.total_trades,
        default_result.long_trades,
        default_result.short_trades,
        default_result.total_pnl,
    )
    print(BacktestEngine.format_result(default_result))
    print(_format_execution_assumptions())
    print(_format_long_short_backtest_table("기본/비교 결과", rows[:4]))
    print(_format_long_short_backtest_table("실험 결과", rows[4:]))
    print(_format_cost_scenarios(df, symbol, base_config, mtf_options))
    print(_format_holdout_table(df, symbol, base_config, mtf_options))
    print(_format_walk_forward_table(df, symbol, base_config, mtf_options))
    print(_format_parameter_robustness_table(df, symbol, base_config, mtf_options))
    print(_format_timeframe_filter_table(df, symbol, base_config, mtf_options))
    print(_format_period_trade_breakdown(default_result))
    print(_format_regime_trade_breakdown(df, default_result, base_config))
    print(_format_trade_distribution(default_result))
    if default_result.total_trades < 30:
        print(
            "\n주의: 기본 MTF 결과는 거래 수가 30건 미만입니다. "
            "testnet 관찰과 더 긴 구간 검증 전에는 생산 전략으로 보지 마세요."
        )


def run_regime_pullback_backtest():
    """BTC 4h regime + 15m/1h RSI/BB pullback 전략 백테스팅 실행"""
    import ccxt

    from exchange import _maybe_resample_ohlcv, _ohlcv_frame, _source_ohlcv_request
    from backtesting.engine import BacktestEngine
    from strategies import BTCRegimePullbackStrategy

    logger = logging.getLogger(__name__)
    symbol = "BTC/USDT"
    days = 365
    helpers = (_source_ohlcv_request, _ohlcv_frame, _maybe_resample_ohlcv)

    logger.info("BTC regime pullback 백테스팅 시작...")
    baseline_df = _fetch_closed_usdm_ohlcv(
        ccxt.binanceusdm,
        symbol,
        LONG_SHORT_TIMEFRAME,
        days,
        helpers,
    )
    if baseline_df.empty:
        logger.warning(f"{symbol} baseline 데이터 수집 실패")
        return

    baseline_config = BTCTrendLongShortConfig()
    baseline_options = {
        "regime_timeframe": LONG_SHORT_REGIME_TIMEFRAME,
        "require_regime_alignment": LONG_SHORT_REQUIRE_REGIME_ALIGNMENT,
        "reverse_only_when_profitable": LONG_SHORT_REVERSE_ONLY_WHEN_PROFITABLE,
        "min_reverse_net_pnl_usdt": LONG_SHORT_MIN_REVERSE_NET_PNL_USDT,
        "max_hold_bars": LONG_SHORT_MAX_HOLD_BARS,
        "break_even_after_pct": LONG_SHORT_BREAK_EVEN_AFTER_PCT,
    }
    rows = [
        (
            "baseline 10m EMA/4h",
            _run_precomputed_long_short_backtest_case(
                baseline_df,
                symbol,
                baseline_config,
                **baseline_options,
            ),
        )
    ]
    datasets = {"baseline": baseline_df}

    for entry_timeframe in ("15m", "1h"):
        df = _fetch_closed_usdm_ohlcv(
            ccxt.binanceusdm,
            symbol,
            entry_timeframe,
            days,
            helpers,
        )
        if df.empty:
            logger.warning(f"{entry_timeframe} 데이터 수집 실패")
            continue
        datasets[entry_timeframe] = df
        logger.info(
            "%s 데이터: %s ~ %s (%s봉), B&H %.2f%%",
            entry_timeframe,
            df.index[0],
            df.index[-1],
            len(df),
            _buy_hold_pct(df),
        )
        for mode, label in (
            ("trend", f"trend-pullback {entry_timeframe}"),
            ("range", f"range-meanrev {entry_timeframe}"),
            ("combined", f"combined {entry_timeframe}"),
        ):
            config = BTCRegimePullbackConfig(
                mode=mode,
                entry_timeframe=entry_timeframe,
            )
            rows.append(
                (
                    label,
                    _run_regime_pullback_backtest_case(
                        df,
                        symbol,
                        BTCRegimePullbackStrategy(config),
                        config,
                    ),
                )
            )

    print(_format_execution_assumptions())
    print(_format_regime_pullback_assumptions(days))
    print(_format_long_short_backtest_table("Regime pullback 비교", rows))

    candidates = rows[1:]
    best_name, best_result = max(
        candidates or rows,
        key=lambda item: item[1].total_pnl,
    )
    print(f"\nBest candidate by net PnL: {best_name}")
    print(BacktestEngine.format_result(best_result))
    print(_format_period_trade_breakdown(best_result))
    print(_format_trade_distribution(best_result))
    print(_format_regime_pullback_oos_table(datasets, symbol))


def _fetch_usdm_ohlcv_history(
    exchange_class,
    symbol: str,
    timeframe: str,
    total_limit: int,
    helpers,
):
    source_request, frame_builder, resampler = helpers
    source_timeframe, source_total_limit, target_rule = source_request(
        timeframe,
        total_limit,
    )
    exchange = exchange_class(
        {
            "enableRateLimit": True,
            "options": {
                "fetchCurrencies": False,
            },
        }
    )
    timeframe_ms = exchange.parse_timeframe(source_timeframe) * 1000
    since = exchange.milliseconds() - source_total_limit * timeframe_ms
    rows = []

    while len(rows) < source_total_limit:
        batch_limit = min(1000, source_total_limit - len(rows))
        batch = exchange.fetch_ohlcv(
            symbol,
            source_timeframe,
            since=since,
            limit=batch_limit,
        )
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + timeframe_ms
        if len(batch) < batch_limit:
            break

    deduped = []
    seen = set()
    for row in rows:
        if row[0] in seen:
            continue
        seen.add(row[0])
        deduped.append(row)

    df = frame_builder(deduped[-source_total_limit:])
    return resampler(df, target_rule, total_limit)


def _fetch_closed_usdm_ohlcv(
    exchange_class,
    symbol: str,
    timeframe: str,
    days: int,
    helpers,
):
    df = _fetch_usdm_ohlcv_history(
        exchange_class,
        symbol,
        timeframe,
        total_limit=_bars_for_timeframe(timeframe, days),
        helpers=helpers,
    )
    return _drop_open_candle(df, timeframe)


def _long_short_config(**overrides) -> BTCTrendLongShortConfig:
    config = BTCTrendLongShortConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _run_long_short_backtest_case(
    df,
    symbol: str,
    config: BTCTrendLongShortConfig,
    backtest_config=None,
    **options,
):
    from backtesting.engine import BacktestEngine
    from strategies import BTCTrendLongShortStrategy

    run_options = dict(options)
    stop_loss_pct = run_options.pop("stop_loss_pct", LONG_SHORT_STOP_LOSS_PCT)
    take_profit_pct = run_options.pop("take_profit_pct", LONG_SHORT_TAKE_PROFIT_PCT)
    strategy = BTCTrendLongShortStrategy(config)
    return BacktestEngine(backtest_config or BacktestConfig()).run(
        df,
        strategy,
        symbol=symbol,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        allow_short=True,
        position_size_usdt=LONG_SHORT_ORDER_USDT,
        flip_on_reverse=True,
        **run_options,
    )


def _run_regime_pullback_backtest_case(
    df,
    symbol: str,
    strategy,
    config: BTCRegimePullbackConfig,
    backtest_config=None,
):
    from backtesting.engine import BacktestEngine

    precomputed = _precompute_regime_pullback_strategy(df, symbol, strategy)
    return BacktestEngine(backtest_config or BacktestConfig()).run(
        df,
        precomputed,
        symbol=symbol,
        stop_loss_pct=config.trend_stop_loss_pct,
        take_profit_pct=config.trend_take_profit_pct,
        allow_short=True,
        position_size_usdt=LONG_SHORT_ORDER_USDT,
        flip_on_reverse=True,
        max_hold_bars=0,
        reverse_only_when_profitable=LONG_SHORT_REVERSE_ONLY_WHEN_PROFITABLE,
        min_reverse_net_pnl_usdt=LONG_SHORT_MIN_REVERSE_NET_PNL_USDT,
    )


def _run_precomputed_long_short_backtest_case(
    df,
    symbol: str,
    config: BTCTrendLongShortConfig,
    backtest_config=None,
    **options,
):
    from backtesting.engine import BacktestEngine

    strategy = _precompute_btc_trend_strategy(df, symbol, config)
    run_options = dict(options)
    stop_loss_pct = run_options.pop("stop_loss_pct", LONG_SHORT_STOP_LOSS_PCT)
    take_profit_pct = run_options.pop("take_profit_pct", LONG_SHORT_TAKE_PROFIT_PCT)
    return BacktestEngine(backtest_config or BacktestConfig()).run(
        df,
        strategy,
        symbol=symbol,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        allow_short=True,
        position_size_usdt=LONG_SHORT_ORDER_USDT,
        flip_on_reverse=True,
        **run_options,
    )


def _precompute_btc_trend_strategy(
    df,
    symbol: str,
    config: BTCTrendLongShortConfig,
):
    from strategies import BTCTrendLongShortStrategy, Signal, TradeSignal

    strategy = BTCTrendLongShortStrategy(config)
    indicators = strategy.get_indicators(df)
    signals = {}
    min_rows = max(config.slow_ema, config.rsi_period) + config.slope_period + 2
    for index in range(len(indicators)):
        current = indicators.iloc[index]
        price = float(current["close"])
        if symbol != "BTC/USDT" or index + 1 < min_rows or index == 0:
            signals[indicators.index[index]] = TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy_name=strategy.name,
                confidence=0.0,
                price=price,
                reason="데이터 부족",
            )
            continue

        previous = indicators.iloc[index - 1]
        cross_up = (
            previous["ema_fast"] <= previous["ema_slow"]
            and current["ema_fast"] > current["ema_slow"]
        )
        cross_down = (
            previous["ema_fast"] >= previous["ema_slow"]
            and current["ema_fast"] < current["ema_slow"]
        )
        bullish = current["ema_slope"] > 0 and current["rsi"] >= config.long_rsi_min
        bearish = current["ema_slope"] < 0 and current["rsi"] <= config.short_rsi_max
        confidence = strategy._confidence(current)

        if cross_up and bullish and confidence >= config.min_confidence:
            signal = Signal.BUY
            reason = (
                f"BTC 롱: EMA{config.fast_ema}/{config.slow_ema} 상향 전환, "
                f"RSI {current['rsi']:.1f}, slope {current['ema_slope']:.2f}"
            )
            metadata = strategy._metadata(current, "long")
        elif cross_down and bearish and confidence >= config.min_confidence:
            signal = Signal.SELL
            reason = (
                f"BTC 숏: EMA{config.fast_ema}/{config.slow_ema} 하향 전환, "
                f"RSI {current['rsi']:.1f}, slope {current['ema_slope']:.2f}"
            )
            metadata = strategy._metadata(current, "short")
        else:
            signal = Signal.HOLD
            confidence = 0.0
            reason = f"전환 없음: RSI {current['rsi']:.1f}, gap {current['trend_gap']:.3%}"
            metadata = None

        signals[indicators.index[index]] = TradeSignal(
            signal=signal,
            symbol=symbol,
            strategy_name=strategy.name,
            confidence=confidence,
            price=price,
            reason=reason,
            metadata=metadata,
        )
    return _PrecomputedSignalStrategy(strategy.name, indicators, signals, config)


def _precompute_regime_pullback_strategy(df, symbol: str, strategy):
    from strategies import Signal, TradeSignal

    indicators = strategy.get_indicators(df)
    signals = {}
    min_rows = max(strategy.config.bb_period, strategy.config.rsi_period) + 2
    for index in range(len(indicators)):
        current = indicators.iloc[index]
        price = float(current["close"])
        if index + 1 < min_rows or index == 0:
            signal = TradeSignal(
                signal=Signal.HOLD,
                symbol=symbol,
                strategy_name=strategy.name,
                confidence=0.0,
                price=price,
                reason="데이터 부족",
            )
        else:
            signal = strategy.analyze_rows(current, indicators.iloc[index - 1], symbol)
        signals[indicators.index[index]] = signal
    return _PrecomputedSignalStrategy(strategy.name, indicators, signals, strategy.config)


class _PrecomputedSignalStrategy:
    def __init__(self, name: str, indicators, signals, config=None):
        self.name = name
        self.indicators = indicators
        self.signals = signals
        self.config = config

    def analyze(self, df, symbol: str):
        from strategies import Signal, TradeSignal

        timestamp = df.index[-1]
        signal = self.signals.get(timestamp)
        if signal:
            return signal
        return TradeSignal(
            signal=Signal.HOLD,
            symbol=symbol,
            strategy_name=self.name,
            confidence=0.0,
            price=float(df["close"].iloc[-1]),
            reason="precomputed signal 없음",
        )

    def get_indicators(self, df):
        return self.indicators.reindex(df.index)


def _backtest_config(**overrides) -> BacktestConfig:
    config = BacktestConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _format_execution_assumptions() -> str:
    return "\n".join(
        [
            "",
            "--- 실행 등가성 체크 ---",
            f"entry_timeframe={LONG_SHORT_TIMEFRAME} closed candles only",
            f"regime_timeframe={LONG_SHORT_REGIME_TIMEFRAME} closed regime series",
            f"fixed_notional=${LONG_SHORT_ORDER_USDT:.2f}",
            (
                "reverse_guard="
                f"{'fee-aware profitable only' if LONG_SHORT_REVERSE_ONLY_WHEN_PROFITABLE else 'raw reverse'}"
            ),
            f"exit=tp {LONG_SHORT_TAKE_PROFIT_PCT:.2f}% / sl {LONG_SHORT_STOP_LOSS_PCT:.2f}%",
            (
                f"time_exit={LONG_SHORT_MAX_HOLD_BARS} bars "
                f"on {LONG_SHORT_TIMEFRAME}"
            ),
            f"break_even_after={LONG_SHORT_BREAK_EVEN_AFTER_PCT:.2f}%",
            "unmodeled=funding partial_fills latency liquidation websocket_timing",
        ]
    )


def _format_regime_pullback_assumptions(days: int) -> str:
    return "\n".join(
        [
            "",
            "--- Regime pullback 설정 ---",
            f"lookback_days={days}",
            "regime=4h EMA12/26 gap>=0.30% with closed candles only",
            "entries=15m and 1h RSI14 + BB20/2.0",
            "trend_exit=SL 1.0% / TP 1.5% / no time exit",
            "range_exit=SL 0.8% / TP 1.0% / max_hold 12 entry bars",
            "runtime_status=backtest-only; testnet executor unchanged",
        ]
    )


def _format_regime_pullback_oos_table(datasets, symbol: str) -> str:
    rows = []
    from strategies import BTCRegimePullbackStrategy

    for entry_timeframe in ("15m", "1h"):
        df = datasets.get(entry_timeframe)
        if df is None or df.empty:
            continue
        windows = _equal_time_windows(df, 3, min_rows=90)
        for index, window in enumerate(windows, start=1):
            config = BTCRegimePullbackConfig(
                mode="combined",
                entry_timeframe=entry_timeframe,
            )
            result = _run_regime_pullback_backtest_case(
                window,
                symbol,
                BTCRegimePullbackStrategy(config),
                config,
            )
            rows.append((f"combined {entry_timeframe} OOS {index}", window, result))
    if not rows:
        return "\n--- Regime pullback OOS ---\n데이터가 부족해 생략"
    return _format_window_result_table("Regime pullback OOS", rows)


def _format_cost_scenarios(
    df,
    symbol: str,
    config: BTCTrendLongShortConfig,
    options,
) -> str:
    base = BacktestConfig()
    rows = []
    for slippage_bp in (0, 2, 5, 10):
        bt_config = _backtest_config(
            commission_rate=base.commission_rate,
            slippage_rate=slippage_bp / 10000,
        )
        result = _run_long_short_backtest_case(
            df,
            symbol,
            config,
            backtest_config=bt_config,
            **options,
        )
        fee_bp = bt_config.commission_rate * 10000
        rows.append((f"fee {fee_bp:.0f}bp slip {slippage_bp}bp", result))
    return _format_long_short_backtest_table("비용/슬리피지 시나리오", rows)


def _format_holdout_table(
    df,
    symbol: str,
    config: BTCTrendLongShortConfig,
    options,
) -> str:
    windows = _equal_time_windows(df, 3, min_rows=90)
    if not windows:
        return "\n--- 기간별 OOS 결과 ---\n데이터가 부족해 생략"

    rows = []
    for index, window in enumerate(windows, start=1):
        result = _run_long_short_backtest_case(window, symbol, config, **options)
        rows.append((f"OOS {index}/{len(windows)}", window, result))
    return _format_window_result_table("기간별 OOS 결과", rows)


def _format_walk_forward_table(
    df,
    symbol: str,
    config: BTCTrendLongShortConfig,
    options,
) -> str:
    train_rows = max(300, len(df) // 5)
    test_rows = max(120, len(df) // 10)
    rows = []
    start = 0
    while start + train_rows + test_rows <= len(df) and len(rows) < 5:
        train = df.iloc[start : start + train_rows]
        test = df.iloc[start + train_rows : start + train_rows + test_rows]
        if len(test) >= 90:
            result = _run_long_short_backtest_case(test, symbol, config, **options)
            label = f"WF {len(rows) + 1}: {_short_date(train.index[0])}->{_short_date(test.index[-1])}"
            rows.append((label, test, result))
        start += test_rows

    if not rows:
        return "\n--- Walk-forward holdout ---\n데이터가 부족해 생략"
    return _format_window_result_table(
        "Walk-forward holdout (fixed params, test window only)",
        rows,
    )


def _format_parameter_robustness_table(
    df,
    symbol: str,
    config: BTCTrendLongShortConfig,
    options,
) -> str:
    cases = [
        ("default params", config, options),
        ("ema 10/24", _long_short_config(fast_ema=10, slow_ema=24), options),
        ("ema 14/30", _long_short_config(fast_ema=14, slow_ema=30), options),
        ("rsi 50/50", _long_short_config(long_rsi_min=50.0, short_rsi_max=50.0), options),
        ("rsi 54/46", _long_short_config(long_rsi_min=54.0, short_rsi_max=46.0), options),
        ("sl/tp 1.5/3", config, dict(options, stop_loss_pct=1.5, take_profit_pct=3.0)),
        ("sl/tp 2.5/5", config, dict(options, stop_loss_pct=2.5, take_profit_pct=5.0)),
    ]
    rows = [
        (
            name,
            _run_long_short_backtest_case(df, symbol, case_config, **case_options),
        )
        for name, case_config, case_options in cases
    ]
    return _format_long_short_backtest_table("파라미터 민감도", rows)


def _format_timeframe_filter_table(
    df,
    symbol: str,
    config: BTCTrendLongShortConfig,
    options,
) -> str:
    cases = []
    for regime_timeframe in ("1h", "2h", "4h", "6h"):
        cases.append(
            (
                f"10m/{regime_timeframe} regime",
                dict(options, regime_timeframe=regime_timeframe, require_regime_alignment=True),
            )
        )
    cases.append(
        (
            "1h EMA + 4h regime",
            dict(
                options,
                higher_timeframe="1h",
                regime_timeframe="4h",
                require_regime_alignment=True,
            ),
        )
    )
    rows = [
        (
            name,
            _run_long_short_backtest_case(df, symbol, config, **case_options),
        )
        for name, case_options in cases
    ]
    return _format_long_short_backtest_table("시간대 필터 비교", rows)


def _format_period_trade_breakdown(result) -> str:
    groups = {}
    for trade in result.trades:
        time = trade.exit_time or trade.entry_time
        key = f"{time.year}-{time.month:02d}"
        groups.setdefault(key, []).append(trade)
    return _format_trade_group_table("월별 거래 성과", groups)


def _format_regime_trade_breakdown(
    df,
    result,
    config: BTCTrendLongShortConfig,
) -> str:
    from strategies.btc_mtf_regime import build_regime_series

    regime = build_regime_series(df, LONG_SHORT_REGIME_TIMEFRAME, config)
    regime_groups = {}
    vol_groups = {}
    volatility = df["close"].pct_change().rolling(_bars_per_day()).std()
    vol_threshold = volatility.dropna().median()

    for trade in result.trades:
        regime_side = "UNKNOWN"
        if trade.entry_time in regime.index:
            value = regime.loc[trade.entry_time, "regime_side"]
            regime_side = "UNKNOWN" if value != value else str(value)
        regime_groups.setdefault(f"{trade.side}/{regime_side}", []).append(trade)

        vol_value = volatility.get(trade.entry_time)
        vol_key = "vol_unknown"
        if vol_value == vol_value and vol_threshold == vol_threshold:
            vol_key = "high_vol" if vol_value >= vol_threshold else "low_vol"
        vol_groups.setdefault(vol_key, []).append(trade)

    return "\n".join(
        [
            _format_trade_group_table("Regime별 거래 성과", regime_groups),
            _format_trade_group_table("변동성별 거래 성과", vol_groups),
        ]
    )


def _format_trade_distribution(result) -> str:
    trades = result.trades
    if not trades:
        return "\n--- 거래 분포 ---\n거래 없음"

    pnls = [trade.pnl for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl <= 0]
    gross_profit = sum(wins)
    best_share = result.best_trade / gross_profit * 100 if gross_profit > 0 else 0.0
    return "\n".join(
        [
            "",
            "--- 거래 분포 ---",
            f"median_pnl=${_median(pnls):.2f}",
            f"avg_win=${_avg(wins):.2f}",
            f"avg_loss=${_avg(losses):.2f}",
            f"best_trade_profit_share={best_share:.1f}%",
            f"max_consecutive_losses={_max_consecutive_losses(trades)}",
            f"worst_trade=${result.worst_trade:.2f}",
        ]
    )


def _format_window_result_table(title: str, rows) -> str:
    lines = [
        "",
        f"--- {title} ---",
        "window                         bars    b&h% trades win%     net    maxDD     pf",
    ]
    for label, window, result in rows:
        lines.append(
            f"{label[:30]:<30} "
            f"{len(window):>5d} "
            f"{_buy_hold_pct(window):>7.2f} "
            f"{result.total_trades:>6d} "
            f"{result.win_rate:>5.1f} "
            f"{result.total_pnl:>7.2f} "
            f"{result.max_drawdown:>8.2f} "
            f"{result.profit_factor:>6.2f}"
        )
    return "\n".join(lines)


def _format_trade_group_table(title: str, groups) -> str:
    lines = [
        "",
        f"--- {title} ---",
        "group                 trades win%      net      avg    worst",
    ]
    if not groups:
        lines.append("none                      0  0.0     0.00     0.00     0.00")
        return "\n".join(lines)
    for key in sorted(groups):
        trades = groups[key]
        pnls = [trade.pnl for trade in trades]
        wins = sum(1 for pnl in pnls if pnl > 0)
        lines.append(
            f"{key[:20]:<20} "
            f"{len(trades):>6d} "
            f"{(wins / len(trades) * 100 if trades else 0):>5.1f} "
            f"{sum(pnls):>8.2f} "
            f"{_avg(pnls):>8.2f} "
            f"{min(pnls) if pnls else 0:>8.2f}"
        )
    return "\n".join(lines)


def _equal_time_windows(df, count: int, min_rows: int):
    size = len(df) // count if count else 0
    if size < min_rows:
        return []
    windows = []
    for index in range(count):
        start = index * size
        end = len(df) if index == count - 1 else (index + 1) * size
        window = df.iloc[start:end]
        if len(window) >= min_rows:
            windows.append(window)
    return windows


def _buy_hold_pct(df) -> float:
    if df.empty:
        return 0.0
    return (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100


def _short_date(value) -> str:
    return value.strftime("%m-%d")


def _bars_per_day() -> int:
    if LONG_SHORT_TIMEFRAME.endswith("m"):
        minutes = int(LONG_SHORT_TIMEFRAME[:-1])
        return max(1, 24 * 60 // minutes)
    if LONG_SHORT_TIMEFRAME.endswith("h"):
        hours = int(LONG_SHORT_TIMEFRAME[:-1])
        return max(1, 24 // hours)
    return 24


def _bars_for_timeframe(timeframe: str, days: int) -> int:
    if timeframe.endswith("m"):
        minutes = int(timeframe[:-1])
        return max(1, days * 24 * 60 // minutes)
    if timeframe.endswith("h"):
        hours = int(timeframe[:-1])
        return max(1, days * 24 // hours)
    if timeframe.endswith("d"):
        return max(1, days // int(timeframe[:-1]))
    return days * 24


def _drop_open_candle(df, timeframe: str):
    if df.empty:
        return df
    seconds = _timeframe_seconds(timeframe)
    if seconds <= 0:
        return df
    latest = df.index[-1]
    close_time = latest.to_pydatetime() + timedelta(seconds=seconds)
    if datetime.utcnow() < close_time:
        return df.iloc[:-1]
    return df


def _timeframe_seconds(timeframe: str) -> int:
    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except ValueError:
        return 0
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 60 * 60
    if unit == "d":
        return value * 24 * 60 * 60
    return 0


def _avg(values) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _max_consecutive_losses(trades) -> int:
    max_count = 0
    current = 0
    for trade in trades:
        if trade.pnl <= 0:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


def _format_long_short_backtest_table(title: str, rows) -> str:
    lines = [
        "",
        f"--- {title} ---",
        (
            "case                         trades long short "
            "win%   gross    fees     net blocked rev_blk   med_hold exits"
        ),
    ]
    for name, result in rows:
        exits = " ".join(
            f"{key}:{value}" for key, value in result.exit_reason_counts.items()
        ) or "-"
        lines.append(
            f"{name[:28]:<28} "
            f"{result.total_trades:>6d} "
            f"{result.long_trades:>4d} "
            f"{result.short_trades:>5d} "
            f"{result.win_rate:>5.1f} "
            f"{result.gross_pnl:>8.2f} "
            f"{result.total_fees:>7.2f} "
            f"{result.total_pnl:>7.2f} "
            f"{result.blocked_by_regime_count:>7d} "
            f"{result.reverse_block_count:>7d} "
            f"{result.median_hold_minutes:>8.1f} "
            f"{exits}"
        )
    return "\n".join(lines)


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
    elif args.backtest_regime_pullback:
        run_regime_pullback_backtest()
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
