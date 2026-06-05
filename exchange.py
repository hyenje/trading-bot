"""
바이낸스 거래소 API 연동 모듈
ccxt 라이브러리를 통한 거래소 통신 관리
"""
import ccxt
import pandas as pd
import logging
from typing import Optional, Dict, List
from datetime import datetime

from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_FUTURES_API_KEY,
    BINANCE_FUTURES_API_SECRET,
    DRY_RUN,
    DRY_RUN_STARTING_BALANCE,
    LONG_SHORT_LEVERAGE,
    USE_TESTNET,
    is_configured,
    mask_sensitive,
    validate_live_trading_allowed,
)

logger = logging.getLogger(__name__)


class ExchangeUnavailableError(RuntimeError):
    """거래소 조회 실패를 실행기가 안전장치로 처리할 수 있게 전달한다."""

    def __init__(self, message: str, rate_limited: bool = False):
        super().__init__(message)
        self.rate_limited = rate_limited


def is_rate_limited_error(error: object) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "too many requests",
            "i'm a teapot",
            '"code":-1003',
            "code:-1003",
            " 429 ",
            " 418 ",
            "429 too many",
            "418 i'm",
        )
    )


def _exchange_unavailable(message: str, error: object) -> ExchangeUnavailableError:
    return ExchangeUnavailableError(
        f"{message}: {mask_sensitive(error)}",
        rate_limited=is_rate_limited_error(error),
    )


def _ohlcv_frame(raw: List) -> pd.DataFrame:
    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def _source_ohlcv_request(timeframe: str, limit: int) -> tuple[str, int, Optional[str]]:
    if timeframe == "10m":
        return "5m", limit * 2 + 2, "10min"
    return timeframe, limit, None


def _maybe_resample_ohlcv(
    df: pd.DataFrame, target_rule: Optional[str], limit: int
) -> pd.DataFrame:
    if df.empty or not target_rule:
        return df.tail(limit)

    resampled = (
        df.resample(target_rule, label="left", closed="left", origin="epoch")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(subset=["open", "high", "low", "close"])
    )
    return resampled.tail(limit)


class BinanceExchange:
    """바이낸스 거래소 API 래퍼 클래스"""

    def __init__(self):
        validate_live_trading_allowed()
        self.dry_run = DRY_RUN
        params = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
                "fetchCurrencies": False,
            },
        }
        if is_configured(BINANCE_API_KEY) and is_configured(BINANCE_API_SECRET):
            params["apiKey"] = BINANCE_API_KEY
            params["secret"] = BINANCE_API_SECRET

        self.exchange = ccxt.binance(params)

        if USE_TESTNET:
            self.exchange.set_sandbox_mode(True)
            logger.info("테스트넷 모드로 연결됨")
        else:
            logger.info("실거래 모드로 연결됨")
        if self.dry_run:
            logger.info("DRY_RUN 모드: 실제 주문은 전송하지 않음")

        self.exchange.load_markets()

    def _build_dry_run_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict:
        if price is None:
            ticker = self.get_ticker(symbol)
            price = ticker.get("last", 0.0) if ticker else 0.0

        now = datetime.utcnow()
        order_id = f"dry-run-{side}-{int(now.timestamp() * 1000)}"
        cost = price * amount if price else 0.0
        return {
            "id": order_id,
            "clientOrderId": order_id,
            "timestamp": int(now.timestamp() * 1000),
            "datetime": now.isoformat(),
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "price": price,
            "average": price,
            "amount": amount,
            "filled": amount,
            "remaining": 0.0,
            "cost": cost,
            "status": "closed",
            "fee": None,
            "info": {"dry_run": True},
        }

    # ----------------------------------------------------------
    # 시세 데이터
    # ----------------------------------------------------------
    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 200
    ) -> pd.DataFrame:
        """OHLCV 캔들 데이터를 DataFrame으로 반환"""
        try:
            source_timeframe, source_limit, target_rule = _source_ohlcv_request(
                timeframe, limit
            )
            raw = self.exchange.fetch_ohlcv(
                symbol, source_timeframe, limit=source_limit
            )
            df = _ohlcv_frame(raw)
            return _maybe_resample_ohlcv(df, target_rule, limit)
        except Exception as e:
            logger.error(f"OHLCV 조회 실패 [{symbol}]: {mask_sensitive(e)}")
            return pd.DataFrame()

    def fetch_ohlcv_history(
        self, symbol: str, timeframe: str = "1h", total_limit: int = 5000
    ) -> pd.DataFrame:
        """여러 번 호출해 더 긴 OHLCV 히스토리를 반환"""
        try:
            source_timeframe, source_total_limit, target_rule = _source_ohlcv_request(
                timeframe, total_limit
            )
            timeframe_ms = self.exchange.parse_timeframe(source_timeframe) * 1000
            since = self.exchange.milliseconds() - source_total_limit * timeframe_ms
            rows = []

            while len(rows) < source_total_limit:
                batch_limit = min(1000, source_total_limit - len(rows))
                batch = self.exchange.fetch_ohlcv(
                    symbol, source_timeframe, since=since, limit=batch_limit
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
                if row[0] not in seen:
                    deduped.append(row)
                    seen.add(row[0])
            deduped = deduped[-source_total_limit:]

            df = _ohlcv_frame(deduped)
            return _maybe_resample_ohlcv(df, target_rule, total_limit)
        except Exception as e:
            logger.error(f"OHLCV 히스토리 조회 실패 [{symbol}]: {mask_sensitive(e)}")
            return pd.DataFrame()

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """현재 티커 정보 조회"""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"티커 조회 실패 [{symbol}]: {mask_sensitive(e)}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회"""
        ticker = self.get_ticker(symbol)
        return ticker["last"] if ticker else None

    # ----------------------------------------------------------
    # 잔고
    # ----------------------------------------------------------
    def get_balance(self, currency: str = "USDT") -> float:
        """특정 화폐 잔고 조회"""
        if self.dry_run:
            return DRY_RUN_STARTING_BALANCE if currency.upper() == "USDT" else 0.0
        try:
            balance = self.exchange.fetch_balance()
            return balance.get(currency, {}).get("free", 0.0)
        except Exception as e:
            logger.error(f"잔고 조회 실패: {mask_sensitive(e)}")
            return 0.0

    def get_all_balances(self) -> Dict[str, float]:
        """전체 잔고 조회 (0 이상만)"""
        if self.dry_run:
            return {"USDT": DRY_RUN_STARTING_BALANCE}
        try:
            balance = self.exchange.fetch_balance()
            return {
                k: v
                for k, v in balance.get("free", {}).items()
                if isinstance(v, (int, float)) and v > 0
            }
        except Exception as e:
            logger.error(f"전체 잔고 조회 실패: {mask_sensitive(e)}")
            return {}

    # ----------------------------------------------------------
    # 주문
    # ----------------------------------------------------------
    def create_market_buy(
        self, symbol: str, amount: float, price: Optional[float] = None
    ) -> Optional[Dict]:
        """시장가 매수"""
        if self.dry_run:
            order = self._build_dry_run_order(symbol, "buy", "market", amount, price)
            logger.info(
                f"[DRY_RUN] 시장가 매수 시뮬레이션: {symbol} {amount} | "
                f"ID: {order['id']}"
            )
            return order
        try:
            order = self.exchange.create_market_buy_order(symbol, amount)
            logger.info(f"시장가 매수 완료: {symbol} {amount} | ID: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"시장가 매수 실패 [{symbol}]: {mask_sensitive(e)}")
            return None

    def create_market_sell(
        self, symbol: str, amount: float, price: Optional[float] = None
    ) -> Optional[Dict]:
        """시장가 매도"""
        if self.dry_run:
            order = self._build_dry_run_order(symbol, "sell", "market", amount, price)
            logger.info(
                f"[DRY_RUN] 시장가 매도 시뮬레이션: {symbol} {amount} | "
                f"ID: {order['id']}"
            )
            return order
        try:
            order = self.exchange.create_market_sell_order(symbol, amount)
            logger.info(f"시장가 매도 완료: {symbol} {amount} | ID: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"시장가 매도 실패 [{symbol}]: {mask_sensitive(e)}")
            return None

    def create_limit_buy(
        self, symbol: str, amount: float, price: float
    ) -> Optional[Dict]:
        """지정가 매수"""
        if self.dry_run:
            order = self._build_dry_run_order(symbol, "buy", "limit", amount, price)
            logger.info(
                f"[DRY_RUN] 지정가 매수 시뮬레이션: {symbol} {amount}@{price} | "
                f"ID: {order['id']}"
            )
            return order
        try:
            order = self.exchange.create_limit_buy_order(symbol, amount, price)
            logger.info(
                f"지정가 매수 완료: {symbol} {amount}@{price} | ID: {order['id']}"
            )
            return order
        except Exception as e:
            logger.error(f"지정가 매수 실패 [{symbol}]: {mask_sensitive(e)}")
            return None

    def create_limit_sell(
        self, symbol: str, amount: float, price: float
    ) -> Optional[Dict]:
        """지정가 매도"""
        if self.dry_run:
            order = self._build_dry_run_order(symbol, "sell", "limit", amount, price)
            logger.info(
                f"[DRY_RUN] 지정가 매도 시뮬레이션: {symbol} {amount}@{price} | "
                f"ID: {order['id']}"
            )
            return order
        try:
            order = self.exchange.create_limit_sell_order(symbol, amount, price)
            logger.info(
                f"지정가 매도 완료: {symbol} {amount}@{price} | ID: {order['id']}"
            )
            return order
        except Exception as e:
            logger.error(f"지정가 매도 실패 [{symbol}]: {mask_sensitive(e)}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """주문 취소"""
        if self.dry_run:
            logger.info(f"[DRY_RUN] 주문 취소 시뮬레이션: {order_id}")
            return True
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"주문 취소 완료: {order_id}")
            return True
        except Exception as e:
            logger.error(f"주문 취소 실패 [{order_id}]: {mask_sensitive(e)}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """미체결 주문 조회"""
        if self.dry_run:
            return []
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"미체결 주문 조회 실패: {mask_sensitive(e)}")
            return []

    def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict]:
        """주문 상태 조회"""
        if self.dry_run:
            logger.info(f"[DRY_RUN] 주문 상태 조회 시뮬레이션: {order_id}")
            return None
        try:
            return self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            logger.error(f"주문 상태 조회 실패 [{order_id}]: {mask_sensitive(e)}")
            return None

    # ----------------------------------------------------------
    # 유틸리티
    # ----------------------------------------------------------
    def get_min_order_amount(self, symbol: str) -> float:
        """심볼의 최소 주문 수량 반환"""
        try:
            market = self.exchange.market(symbol)
            return market.get("limits", {}).get("amount", {}).get("min", 0.0)
        except Exception:
            return 0.0

    def calculate_buy_amount(self, symbol: str, usdt_amount: float) -> float:
        """USDT 금액으로 매수 가능한 수량 계산"""
        price = self.get_current_price(symbol)
        if not price or price == 0:
            return 0.0
        amount = usdt_amount / price
        min_amount = self.get_min_order_amount(symbol)
        if amount < min_amount:
            logger.warning(
                f"주문 수량({amount})이 최소 수량({min_amount})보다 작습니다."
            )
            return 0.0
        return amount


class BinanceFuturesExchange:
    """Binance USD-M Futures 테스트넷 래퍼"""

    PROTECTION_CLIENT_ID_PREFIX = "btcls-"
    FALLBACK_MIN_NOTIONAL_USDT = 50.0

    def __init__(self):
        if not USE_TESTNET:
            raise RuntimeError("Futures 실행 래퍼는 테스트넷에서만 사용할 수 있습니다.")

        params = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
                "fetchCurrencies": False,
            },
        }
        if is_configured(BINANCE_FUTURES_API_KEY) and is_configured(
            BINANCE_FUTURES_API_SECRET
        ):
            params["apiKey"] = BINANCE_FUTURES_API_KEY
            params["secret"] = BINANCE_FUTURES_API_SECRET

        self.exchange = ccxt.binanceusdm(params)
        self.exchange.set_sandbox_mode(True)
        logger.info("Futures 테스트넷 모드로 연결됨")
        self.exchange.load_markets()

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 240
    ) -> pd.DataFrame:
        try:
            source_timeframe, source_limit, target_rule = _source_ohlcv_request(
                timeframe, limit
            )
            raw = self.exchange.fetch_ohlcv(
                symbol, source_timeframe, limit=source_limit
            )
            df = _ohlcv_frame(raw)
            return _maybe_resample_ohlcv(df, target_rule, limit)
        except Exception as e:
            error = _exchange_unavailable(f"Futures OHLCV 조회 실패 [{symbol}]", e)
            logger.error(str(error))
            raise error

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Futures 티커 조회 실패 [{symbol}]: {mask_sensitive(e)}")
            return None

    def get_balance(self, currency: str = "USDT") -> float:
        try:
            balance = self.exchange.fetch_balance()
            total = balance.get("total", {}).get(currency)
            if total is not None:
                return total
            return balance.get("free", {}).get(currency, 0.0)
        except Exception as e:
            error = _exchange_unavailable("Futures 잔고 조회 실패", e)
            logger.error(str(error))
            raise error

    def check_private_access(self) -> bool:
        try:
            self.exchange.fetch_balance()
            return True
        except Exception as e:
            logger.error(f"Futures 인증 실패: {mask_sensitive(e)}")
            return False

    def set_leverage(self, symbol: str, leverage: int = LONG_SHORT_LEVERAGE) -> bool:
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Futures 레버리지 설정: {symbol} x{leverage}")
            return True
        except Exception as e:
            logger.warning(f"Futures 레버리지 설정 실패: {mask_sensitive(e)}")
            return False

    def fetch_position(self, symbol: str) -> Dict:
        try:
            positions = self.exchange.fetch_positions([symbol])
        except Exception as e:
            error = _exchange_unavailable("Futures 포지션 조회 실패", e)
            logger.error(str(error))
            raise error

        for position in positions:
            if position.get("symbol") != symbol:
                continue

            amount = self._position_amount(position)
            payload = {
                "amount": abs(amount),
                "entry_price": self._position_number(position, "entryPrice"),
                "mark_price": self._position_number(position, "markPrice"),
                "unrealized_pnl": self._position_number(position, "unrealizedPnl", "unRealizedProfit"),
                "liquidation_price": self._position_number(position, "liquidationPrice"),
                "percentage": self._position_number(position, "percentage"),
                "raw": position,
            }
            if amount > 0:
                return {"side": "long", **payload}
            if amount < 0:
                return {"side": "short", **payload}

        return self._flat_position()

    def amount_from_usdt(self, symbol: str, usdt_amount: float) -> float:
        min_notional = self.get_min_order_notional(symbol)
        if usdt_amount < min_notional:
            logger.warning(
                f"Futures 주문 금액({usdt_amount})이 최소 명목금액({min_notional})보다 작습니다."
            )
            return 0.0

        ticker = self.get_ticker(symbol)
        price = ticker.get("last") if ticker else None
        if not price:
            return 0.0

        raw_amount = usdt_amount / price
        amount = float(self.exchange.amount_to_precision(symbol, raw_amount))
        min_amount = self.get_min_order_amount(symbol)
        if amount < min_amount:
            logger.warning(
                f"Futures 주문 수량({amount})이 최소 수량({min_amount})보다 작습니다."
            )
            return 0.0
        return amount

    def get_min_order_notional(self, symbol: str) -> float:
        try:
            market = self.exchange.market(symbol)
            min_cost = market.get("limits", {}).get("cost", {}).get("min")
            if min_cost:
                return max(float(min_cost), self.FALLBACK_MIN_NOTIONAL_USDT)
        except Exception:
            pass
        return self.FALLBACK_MIN_NOTIONAL_USDT

    def open_long(self, symbol: str, amount: float) -> Optional[Dict]:
        return self._create_market_order(symbol, "buy", amount, reduce_only=False)

    def open_short(self, symbol: str, amount: float) -> Optional[Dict]:
        return self._create_market_order(symbol, "sell", amount, reduce_only=False)

    def close_long(self, symbol: str, amount: float) -> Optional[Dict]:
        return self._create_market_order(symbol, "sell", amount, reduce_only=True)

    def close_short(self, symbol: str, amount: float) -> Optional[Dict]:
        return self._create_market_order(symbol, "buy", amount, reduce_only=True)

    def create_protection_orders(
        self,
        symbol: str,
        position_side: str,
        amount: float,
        stop_loss: float,
        take_profit: float,
    ) -> List[Dict]:
        side = "sell" if position_side == "long" else "buy"
        orders = []

        stop_order = self._create_trigger_order(
            symbol,
            side=side,
            order_type="STOP_MARKET",
            amount=amount,
            stop_price=stop_loss,
            client_id_kind="sl",
        )
        if stop_order:
            orders.append(stop_order)

        take_order = self._create_trigger_order(
            symbol,
            side=side,
            order_type="TAKE_PROFIT_MARKET",
            amount=amount,
            stop_price=take_profit,
            client_id_kind="tp",
        )
        if take_order:
            orders.append(take_order)

        return orders

    def fetch_protection_orders(self, symbol: str) -> List[Dict]:
        try:
            return [
                order
                for order in self.exchange.fetch_open_orders(symbol)
                if self._is_protection_order(order)
            ]
        except Exception as e:
            error = _exchange_unavailable("Futures 보호 주문 조회 실패", e)
            logger.error(str(error))
            raise error

    def cancel_protection_orders(self, symbol: str) -> List[Dict]:
        cancelled = []
        for order in self.fetch_protection_orders(symbol):
            order_id = order.get("id")
            if not order_id:
                continue
            try:
                cancelled.append(self.exchange.cancel_order(order_id, symbol))
                logger.info(f"Futures 보호 주문 취소: {symbol} | ID: {order_id}")
            except Exception as e:
                logger.error(f"Futures 보호 주문 취소 실패: {mask_sensitive(e)}")
        return cancelled

    def _create_market_order(
        self, symbol: str, side: str, amount: float, reduce_only: bool
    ) -> Optional[Dict]:
        try:
            order = self.exchange.create_order(
                symbol,
                "market",
                side,
                amount,
                params={"reduceOnly": reduce_only},
            )
            logger.info(
                f"Futures 시장가 주문 완료: {side} {amount} {symbol} "
                f"reduceOnly={reduce_only} | ID: {order.get('id')}"
            )
            return order
        except Exception as e:
            logger.error(f"Futures 시장가 주문 실패: {mask_sensitive(e)}")
            return None

    def _create_trigger_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        stop_price: float,
        client_id_kind: str,
    ) -> Optional[Dict]:
        try:
            precise_amount = float(self.exchange.amount_to_precision(symbol, amount))
            precise_stop = self.exchange.price_to_precision(symbol, stop_price)
            client_id = self._protection_client_order_id(client_id_kind)
            order = self.exchange.create_order(
                symbol,
                order_type,
                side,
                precise_amount,
                None,
                {
                    "stopPrice": precise_stop,
                    "reduceOnly": True,
                    "workingType": "MARK_PRICE",
                    "newClientOrderId": client_id,
                },
            )
            logger.info(
                f"Futures 보호 주문 생성: {order_type} {side} {precise_amount} "
                f"{symbol} stopPrice={precise_stop} | ID: {order.get('id')}"
            )
            return order
        except Exception as e:
            logger.error(f"Futures 보호 주문 생성 실패: {mask_sensitive(e)}")
            return None

    def _protection_client_order_id(self, kind: str) -> str:
        now = int(datetime.utcnow().timestamp() * 1000)
        return f"{self.PROTECTION_CLIENT_ID_PREFIX}{kind}-{now}"

    def _is_protection_order(self, order: Dict) -> bool:
        client_id = order.get("clientOrderId") or order.get("clientOrderID")
        info = order.get("info", {})
        client_id = client_id or info.get("clientOrderId") or info.get("clientOrderID")
        return bool(
            isinstance(client_id, str)
            and client_id.startswith(self.PROTECTION_CLIENT_ID_PREFIX)
        )

    def get_min_order_amount(self, symbol: str) -> float:
        try:
            market = self.exchange.market(symbol)
            return market.get("limits", {}).get("amount", {}).get("min", 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _position_amount(position: Dict) -> float:
        info = position.get("info", {})
        for key in ("positionAmt", "positionAmount"):
            if key in info:
                try:
                    return float(info[key])
                except (TypeError, ValueError):
                    pass

        side = (position.get("side") or "").lower()
        contracts = position.get("contracts") or position.get("contractSize") or 0
        try:
            amount = float(contracts)
        except (TypeError, ValueError):
            amount = 0.0

        if side == "short":
            return -amount
        if side == "long":
            return amount
        return 0.0

    @staticmethod
    def _position_number(position: Dict, *keys: str) -> float:
        info = position.get("info", {})
        for key in keys:
            for source in (position, info):
                if key in source and source[key] not in (None, ""):
                    try:
                        return float(source[key])
                    except (TypeError, ValueError):
                        pass
        return 0.0

    @staticmethod
    def _flat_position() -> Dict:
        return {
            "side": "flat",
            "amount": 0.0,
            "entry_price": 0.0,
            "mark_price": 0.0,
            "unrealized_pnl": 0.0,
            "liquidation_price": 0.0,
            "percentage": 0.0,
        }
