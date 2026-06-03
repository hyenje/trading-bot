# Crypto Trading Bot - 설정 가이드

## 프로젝트 구조

```
crypto-trading-bot/
├── main.py              # 메인 실행 스크립트
├── bot.py               # 트레이딩 봇 엔진
├── exchange.py          # 바이낸스 API 연동
├── risk_manager.py      # 리스크 관리 모듈
├── notifier.py          # 텔레그램 알림
├── config.py            # 전체 설정
├── requirements.txt     # 의존성 패키지
├── .env.example         # 환경변수 템플릿
├── strategies/          # 매매 전략
│   ├── base.py          # 전략 베이스 클래스
│   ├── ma_cross.py      # 이동평균 크로스
│   ├── rsi_strategy.py  # RSI 전략
│   ├── bollinger_strategy.py  # 볼린저 밴드
│   ├── grid_strategy.py # 그리드 트레이딩
│   └── ensemble.py      # 앙상블 (종합 판단)
├── backtesting/         # 백테스팅
│   └── engine.py        # 백테스팅 엔진
├── dashboard/           # 웹 대시보드
│   └── app.py           # Flask 앱
├── utils/               # 유틸리티
│   └── logger.py        # 로깅 설정
├── logs/                # 로그 파일
└── data/                # 데이터 저장
```

## 1단계: 환경 설정

```bash
cd crypto-trading-bot
pip install -r requirements.txt
```

## 2단계: API 키 설정

프로젝트에 포함된 `.env.example`을 `.env`로 복사한 뒤 값을 입력합니다.

```bash
cp .env.example .env
```

### 바이낸스 API 키 발급
1. https://www.binance.com 로그인
2. [API Management] → [Create API] 클릭
3. API Key와 Secret Key를 `.env`에 입력
4. IP 제한, 출금 권한 해제 권장

### 안전 모드
기본값은 실제 주문을 넣지 않는 드라이런입니다.

```bash
USE_TESTNET=true
DRY_RUN=true
ALLOW_LIVE_TRADING=false
```

- `DRY_RUN=true`: 매수/매도 주문을 Binance에 전송하지 않고 가짜 체결로 기록
- `USE_TESTNET=true`: Binance 테스트넷 사용
- `ALLOW_LIVE_TRADING=false`: 실거래 주문 차단

실거래 주문은 `USE_TESTNET=false`, `DRY_RUN=false`, `ALLOW_LIVE_TRADING=true`가 모두 설정된 경우에만 가능합니다.

### 텔레그램 봇 설정
1. 텔레그램에서 @BotFather 검색
2. `/newbot` 명령 → 봇 이름/유저네임 입력
3. 받은 토큰을 `.env`에 입력
4. 봇에게 메시지 전송 후 Chat ID 확인:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`

## 3단계: 실행

```bash
# 주문 없이 API/설정 점검
python main.py --check

# 봇 + 대시보드 동시 실행
python main.py

# 봇만 실행 (대시보드 X)
python main.py --bot-only

# 대시보드만 실행
python main.py --dashboard

# 백테스팅 실행
python main.py --backtest

# BTC 롱/숏 전략 백테스팅
python main.py --backtest-long-short

# BTC 롱/숏 시그널 관찰 대시보드
python main.py --observe-long-short

# Futures 테스트넷 API 점검
python main.py --check-futures

# BTC 롱/숏 Futures 테스트넷 실제 주문 실행
python main.py --trade-long-short
```

## 4단계: 설정 커스터마이즈

`config.py`에서 주요 파라미터를 조정합니다.

### 기본 거래 설정
- `symbols`: 거래 대상 (기본: BTC/USDT, ETH/USDT)
- `timeframe`: 캔들 주기 (1m, 5m, 15m, 1h, 4h, 1d)
- `order_amount`: 건당 주문 금액 (USDT)
- `stop_loss_pct`: 손절 비율 (기본 2%)
- `take_profit_pct`: 익절 비율 (기본 4%)
- `max_daily_loss_pct`: 일일 최대 손실 한도 (기본 5%)

### 테스트넷 모드
`config.py`에서 `USE_TESTNET = True`로 설정하면 실제 돈 없이 테스트할 수 있습니다.
**실거래 전 반드시 테스트넷에서 충분히 검증하세요.**

### 드라이런 모드
`DRY_RUN=true`이면 전략, 리스크 관리, 대시보드 흐름은 실행되지만 실제 주문 API는 호출하지 않습니다.
키를 넣은 직후에는 먼저 아래 명령으로 시세 조회와 잔고 인증을 확인하세요.

```bash
python main.py --check
```

## 전략 설명

### 앙상블 전략 (기본)
MA Cross, RSI, 볼린저 밴드 3개 전략을 가중 투표로 종합 판단합니다.
최소 60% 이상 합의가 이루어져야 매매 시그널이 발생합니다.

### 그리드 전략 (보조)
앙상블이 HOLD일 때 그리드 시그널을 참고합니다.
횡보장에서 자동으로 분할 매수/매도합니다.

### BTC 롱/숏 추세 전략
EMA 12/26 크로스, EMA 기울기, RSI 필터를 조합해 BTC/USDT 롱/숏 시그널을 냅니다.
Spot 봇에는 숏 주문을 연결하지 않았고, Futures 테스트넷 실행기에서만 롱/숏 주문을 허용합니다.

```bash
python main.py --backtest-long-short
```

기본 검증값은 10분봉 최대 5000개, 손절 2.0%, 익절 4.0%입니다.
롱/숏 백테스트는 실제 실행기의 `LONG_SHORT_ORDER_USDT`와 같은 고정 명목금액 기준으로 포지션을 계산하고, 반대 신호가 나오면 청산 후 즉시 반대 포지션으로 전환하는 executor-like 결과를 기본으로 출력합니다.
결과에는 both, long-only, short-only 비교와 gross PnL, fee, net PnL, 청산 사유, 보유시간 요약이 함께 표시됩니다.
같은 명령에서 롱 RSI 필터, 시간 청산, 상위봉 추세 필터, EMA gap, EMA slope 실험도 고정 비교표로 확인할 수 있습니다.
파라미터는 `.env`에서 `BTC_LS_FAST_EMA`, `BTC_LS_SLOW_EMA`, `BTC_LS_RSI_PERIOD` 등으로 조정할 수 있습니다.
실제 주문 없이 현재 신호와 최근 롱/숏 신호를 보려면 `--observe-long-short`를 실행하세요.

롱/숏을 실제 테스트넷 주문으로 실행하려면 Binance Futures 테스트넷 키가 필요합니다.
Spot 테스트넷 키(`testnet.binance.vision`)로는 숏 포지션을 열 수 없습니다.
Futures 테스트넷 키를 설정한 뒤 아래 세 값을 명시해야 주문 실행 모드가 켜집니다.

```bash
USE_TESTNET=true
DRY_RUN=false
ENABLE_LONG_SHORT_EXECUTION=true
```

먼저 `python main.py --check-futures`로 Futures 잔고 인증이 성공하는지 확인한 다음,
`python main.py --trade-long-short`를 실행하세요.
기본 봉은 `LONG_SHORT_TIMEFRAME=10m`, 주문 판단 주기는 `LONG_SHORT_POLL_INTERVAL=600`입니다.
리스크 감시는 `LONG_SHORT_RISK_POLL_INTERVAL=180`으로 별도 실행됩니다.
기본 주문 금액은 `LONG_SHORT_ORDER_USDT=25`, 기본 레버리지는 `LONG_SHORT_LEVERAGE=1`입니다.
대시보드의 `WAIT_LONG_BIAS` / `WAIT_SHORT_BIAS`는 현재 추세 편향을 보여주는 대기 상태이며, 실제 포지션 진입 신호와는 구분됩니다.
재시작 직후 최근 신호를 따라잡아 테스트넷 진입까지 허용하려면 `LONG_SHORT_ENABLE_SIGNAL_CATCHUP=true`를 설정하세요. 이 경우에도 flat 상태, 같은 방향 bias, `LONG_SHORT_MAX_SIGNAL_AGE_MINUTES` 이내 신호일 때만 작동합니다.
Futures 실행기는 진입 직후 `STOP_MARKET`/`TAKE_PROFIT_MARKET` reduce-only 보호 주문을 거래소에 생성합니다.
거래소 조회 실패나 rate limit이 감지되면 신규 진입은 차단되고, 마지막으로 확인된 포지션 상태를 `flat`으로 덮어쓰지 않습니다.
포지션 보유 중에는 봇 내부 손절/익절 조건도 보조로 확인한 뒤 새 롱/숏 신호를 처리합니다.
안전장치는 하루 손실 `LONG_SHORT_MAX_DAILY_LOSS_USDT=50` 또는
`LONG_SHORT_MAX_DAILY_LOSS_PCT=3.0`, 하루 신규 진입
`LONG_SHORT_MAX_DAILY_TRADES=10`, 연속 손실
`LONG_SHORT_MAX_CONSECUTIVE_LOSSES=3` 기준으로 신규 진입을 차단합니다.
연속 손실 한도에 닿으면 `LONG_SHORT_COOLDOWN_AFTER_LOSSES_MINUTES=60`분 동안 쉽니다.
일일 손실 한도는 현재 잔고와 미실현 손익을 함께 보고, 포지션 보유 중 한도에 닿으면 시장가로 정리합니다.

## 주의사항

- **투자 원금 손실 가능성**: 자동 매매는 원금 손실 위험이 있습니다
- **테스트넷 먼저**: 실거래 전 반드시 테스트넷에서 검증
- **소액으로 시작**: 처음에는 소액으로 시작하세요
- **API 보안**: API 키에 출금 권한을 부여하지 마세요
- **모니터링**: 봇을 완전 방치하지 말고 주기적으로 확인하세요
