"""
Flask 웹 대시보드
실시간 포트폴리오, 거래 내역, 차트 시각화
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DASHBOARD_DEBUG, DASHBOARD_HOST, DASHBOARD_PORT, DRY_RUN

app = Flask(__name__)

# 봇 인스턴스 참조 (메인에서 주입)
bot_instance = None


def set_bot(bot):
    global bot_instance
    bot_instance = bot


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _serialize_trade(trade: dict[str, Any]) -> dict[str, Any]:
    serialized = _serialize_value(trade)
    serialized["side"] = serialized.get("side", "sell")
    serialized["side_label"] = "매수" if serialized["side"] == "buy" else "청산"
    return serialized


def _empty_status() -> dict[str, Any]:
    return {
        "running": False,
        "dry_run": DRY_RUN,
        "balance": 0,
        "daily_pnl": 0,
        "open_positions": 0,
        "daily_trades": 0,
        "daily_win_rate": 0,
        "total_trades": 0,
        "positions": [],
        "recent_trades": [],
        "equity_curve": [],
        "timestamp": datetime.now().isoformat(),
    }


def _build_status_payload() -> dict[str, Any]:
    if not bot_instance:
        return _empty_status()

    summary = _serialize_value(bot_instance.get_status())
    summary["recent_trades"] = [
        _serialize_trade(trade) for trade in summary.get("recent_trades", [])
    ]
    return summary


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/api/status")
def api_status():
    return jsonify(_build_status_payload())


def run_dashboard():
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG)


if __name__ == "__main__":
    run_dashboard()
