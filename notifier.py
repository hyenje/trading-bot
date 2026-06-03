"""
텔레그램 알림 모듈
매매 시그널, 체결, 에러 등을 텔레그램으로 전송
"""
import logging
from datetime import datetime
from typing import Dict

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, is_configured

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """텔레그램 봇 알림 전송"""

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.enabled = is_configured(self.token) and is_configured(self.chat_id)

        if not self.enabled:
            logger.warning("텔레그램 토큰 또는 채팅 ID 미설정 → 알림 비활성화")

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """메시지 전송"""
        if not self.enabled:
            logger.debug(f"[알림 비활성] {text}")
            return False

        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"텔레그램 전송 에러: {e}")
            return False

    # ----------------------------------------------------------
    # 포맷된 알림 메서드
    # ----------------------------------------------------------
    def notify_signal(self, signal_data: Dict):
        """매매 시그널 알림"""
        emoji = "🟢" if signal_data.get("signal") == "BUY" else "🔴"
        msg = (
            f"{emoji} <b>매매 시그널</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"심볼: <code>{signal_data.get('symbol', '-')}</code>\n"
            f"방향: <b>{signal_data.get('signal', '-')}</b>\n"
            f"전략: {signal_data.get('strategy', '-')}\n"
            f"신뢰도: {signal_data.get('confidence', 0):.1%}\n"
            f"가격: ${signal_data.get('price', 0):,.2f}\n"
            f"사유: {signal_data.get('reason', '-')}\n"
            f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(msg)

    def notify_order(self, order_data: Dict):
        """주문 체결 알림"""
        side = order_data.get("side", "buy")
        emoji = "✅" if side == "buy" else "💰"
        msg = (
            f"{emoji} <b>주문 체결</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"심볼: <code>{order_data.get('symbol', '-')}</code>\n"
            f"구분: <b>{'매수' if side == 'buy' else '매도'}</b>\n"
            f"가격: ${order_data.get('price', 0):,.2f}\n"
            f"수량: {order_data.get('amount', 0):.6f}\n"
            f"금액: ${order_data.get('cost', 0):,.2f}\n"
            f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(msg)

    def notify_close(self, trade_data: Dict):
        """포지션 청산 알림"""
        pnl = trade_data.get("pnl", 0)
        emoji = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>포지션 청산</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"심볼: <code>{trade_data.get('symbol', '-')}</code>\n"
            f"진입: ${trade_data.get('entry_price', 0):,.2f}\n"
            f"청산: ${trade_data.get('exit_price', 0):,.2f}\n"
            f"수익: <b>${pnl:+,.2f} ({trade_data.get('pnl_pct', 0):+.2f}%)</b>\n"
            f"사유: {trade_data.get('reason', '-')}\n"
            f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(msg)

    def notify_daily_summary(self, summary: Dict):
        """일일 요약 알림"""
        msg = (
            f"📊 <b>일일 요약</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"거래 횟수: {summary.get('trades', 0)}\n"
            f"승률: {summary.get('win_rate', 0):.1f}%\n"
            f"일일 수익: <b>${summary.get('daily_pnl', 0):+,.2f}</b>\n"
            f"보유 포지션: {summary.get('open_positions', 0)}\n"
            f"총 잔고: ${summary.get('balance', 0):,.2f}\n"
            f"날짜: {datetime.now().strftime('%Y-%m-%d')}"
        )
        self.send_message(msg)

    def notify_error(self, error_msg: str):
        """에러 알림"""
        msg = (
            f"⚠️ <b>에러 발생</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{error_msg}\n"
            f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(msg)

    def notify_bot_status(self, status: str):
        """봇 상태 알림"""
        emoji = "🤖" if status == "started" else "🛑"
        action = "시작" if status == "started" else "중지"
        msg = (
            f"{emoji} <b>트레이딩 봇 {action}</b>\n"
            f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(msg)
