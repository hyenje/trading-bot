"""
메인 트레이딩 봇 엔진
모든 모듈을 통합하여 자동 매매 실행
"""
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List

from exchange import BinanceExchange
from strategies import (
    EnsembleStrategy,
    GridStrategy,
    MACrossStrategy,
    RSIStrategy,
    BollingerStrategy,
    Signal,
)
from risk_manager import RiskManager
from notifier import TelegramNotifier
from config import (
    DRY_RUN,
    TradingConfig,
    MAConfig,
    RSIConfig,
    BollingerConfig,
    GridConfig,
)

logger = logging.getLogger(__name__)


class TradingBot:
    """자동 트레이딩 봇 메인 엔진"""

    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.running = False

        # 모듈 초기화
        logger.info("봇 초기화 시작...")
        self.exchange = BinanceExchange()
        self.risk_manager = RiskManager(self.config)
        self.notifier = TelegramNotifier()

        # 전략 초기화
        self.ensemble = EnsembleStrategy(
            strategies=[
                MACrossStrategy(MAConfig()),
                RSIStrategy(RSIConfig()),
                BollingerStrategy(BollingerConfig()),
            ]
        )
        self.grid_strategy = GridStrategy(GridConfig())

        # 자산 추이 기록
        self.equity_curve: List[float] = []
        self._lock = threading.Lock()

        # 잔고 초기화
        balance = self.exchange.get_balance("USDT")
        self.risk_manager.set_capital(balance)
        logger.info(f"봇 초기화 완료 | USDT 잔고: {balance:.2f}")

    # ----------------------------------------------------------
    # 메인 루프
    # ----------------------------------------------------------
    def start(self):
        """봇 시작"""
        self.running = True
        self.notifier.notify_bot_status("started")
        logger.info("=" * 50)
        logger.info("트레이딩 봇 시작")
        logger.info(f"드라이런: {'ON' if DRY_RUN else 'OFF'}")
        logger.info(f"심볼: {self.config.symbols}")
        logger.info(f"타임프레임: {self.config.timeframe}")
        logger.info(f"주문 금액: ${self.config.order_amount}")
        logger.info("=" * 50)

        while self.running:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("키보드 인터럽트 → 봇 중지")
                self.stop()
                break
            except Exception as e:
                logger.error(f"메인 루프 에러: {e}", exc_info=True)
                self.notifier.notify_error(f"메인 루프 에러: {e}")

            time.sleep(self.config.check_interval)

    def stop(self):
        """봇 중지"""
        self.running = False
        self.notifier.notify_bot_status("stopped")
        logger.info("트레이딩 봇 중지됨")

    def _tick(self):
        """한 사이클 실행"""
        for symbol in self.config.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"심볼 처리 에러 [{symbol}]: {e}")

        # 자산 추이 기록
        balance = self.exchange.get_balance("USDT")
        with self._lock:
            self.equity_curve.append(balance)

    def _process_symbol(self, symbol: str):
        """개별 심볼 처리"""

        # 1. 데이터 수집
        df = self.exchange.fetch_ohlcv(symbol, self.config.timeframe, limit=200)
        if df.empty:
            return

        current_price = df["close"].iloc[-1]

        # 2. 보유 포지션 손절/익절 확인
        sl_tp = self.risk_manager.check_stop_loss_take_profit(symbol, current_price)
        if sl_tp:
            reason = "손절" if sl_tp == "stop_loss" else "익절"
            self._execute_sell(symbol, current_price, reason)
            return

        # 3. 앙상블 전략 분석
        ensemble_signal = self.ensemble.analyze(df, symbol)
        logger.info(
            f"[{symbol}] 앙상블: {ensemble_signal.signal.value} "
            f"(신뢰도: {ensemble_signal.confidence:.2f}) - {ensemble_signal.reason}"
        )

        # 4. 그리드 전략 분석
        grid_signal = self.grid_strategy.analyze(df, symbol)

        # 5. 최종 결정 및 실행
        # 앙상블 시그널 우선, 그리드는 보조
        final_signal = ensemble_signal
        if ensemble_signal.signal == Signal.HOLD and grid_signal.signal != Signal.HOLD:
            final_signal = grid_signal
            logger.info(f"[{symbol}] 그리드 시그널 채택: {grid_signal.reason}")

        if final_signal.signal == Signal.BUY:
            self._execute_buy(symbol, current_price, final_signal)
        elif final_signal.signal == Signal.SELL:
            self._execute_sell(symbol, current_price, final_signal.reason)

    # ----------------------------------------------------------
    # 주문 실행
    # ----------------------------------------------------------
    def _execute_buy(self, symbol: str, price: float, signal):
        """매수 실행"""
        can_open, reason = self.risk_manager.can_open_position(signal)
        if not can_open:
            logger.info(f"[{symbol}] 매수 차단: {reason}")
            return

        amount = self.risk_manager.calculate_order_amount(symbol, price)
        if amount <= 0:
            logger.warning(f"[{symbol}] 주문 수량 부족")
            return

        # 최소 주문 확인
        min_amount = self.exchange.get_min_order_amount(symbol)
        if amount < min_amount:
            logger.warning(f"[{symbol}] 최소 주문 수량 미달 ({amount} < {min_amount})")
            return

        order = self.exchange.create_market_buy(symbol, amount, price=price)
        if order:
            filled_price = order.get("average", price)
            filled_amount = order.get("filled", amount)

            self.risk_manager.open_position(
                symbol=symbol,
                entry_price=filled_price,
                amount=filled_amount,
                strategy=signal.strategy_name,
                order_id=order.get("id"),
            )

            self.notifier.notify_signal({
                "symbol": symbol,
                "signal": "BUY",
                "strategy": signal.strategy_name,
                "confidence": signal.confidence,
                "price": filled_price,
                "reason": signal.reason,
            })
            self.notifier.notify_order({
                "symbol": symbol,
                "side": "buy",
                "price": filled_price,
                "amount": filled_amount,
                "cost": filled_price * filled_amount,
            })

    def _execute_sell(self, symbol: str, price: float, reason: str):
        """매도 실행"""
        if symbol not in self.risk_manager.positions:
            return

        pos = self.risk_manager.positions[symbol]
        order = self.exchange.create_market_sell(symbol, pos.amount, price=price)

        if order:
            filled_price = order.get("average", price)
            trade = self.risk_manager.close_position(symbol, filled_price, reason)

            if trade:
                self.notifier.notify_close(trade)
                self.notifier.notify_order({
                    "symbol": symbol,
                    "side": "sell",
                    "price": filled_price,
                    "amount": pos.amount,
                    "cost": filled_price * pos.amount,
                })

    # ----------------------------------------------------------
    # 상태 조회 (대시보드용)
    # ----------------------------------------------------------
    def get_status(self) -> Dict:
        """현재 상태 반환"""
        summary = self.risk_manager.get_portfolio_summary()
        balance = self.exchange.get_balance("USDT")

        with self._lock:
            equity = list(self.equity_curve[-500:])  # 최근 500개

        return {
            "running": self.running,
            "dry_run": DRY_RUN,
            "balance": balance,
            "daily_pnl": summary["daily_pnl"],
            "open_positions": summary["open_positions"],
            "daily_trades": summary["daily_trades"],
            "daily_win_rate": summary["daily_win_rate"],
            "positions": summary["positions"],
            "recent_trades": self.risk_manager.trade_history[-20:],
            "equity_curve": equity,
            "total_trades": summary["total_trades"],
            "timestamp": datetime.now().isoformat(),
        }
