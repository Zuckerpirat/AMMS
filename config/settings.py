import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca — both required; bot will not start without them
ALPACA_API_KEY: str = os.environ["ALPACA_API_KEY"]
ALPACA_API_SECRET: str = os.environ["ALPACA_API_SECRET"]

# Telegram — optional; disabled when either value is missing
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED: bool = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)

# Database
DB_PATH: str = os.getenv("DB_PATH", "data/amms.db")

# Watchlist — tickers the morning scan checks
WATCHLIST: list[str] = [
    t.strip() for t in os.getenv(
        "WATCHLIST", "AAPL,MSFT,NVDA,AMD,TSLA,META,AMZN,GOOGL,SPY,QQQ"
    ).split(",") if t.strip()
]

# Strategy — Momentum
MOMENTUM_LOOKBACK: int = int(os.getenv("MOMENTUM_LOOKBACK", "20"))
VOLUME_MULTIPLIER: float = float(os.getenv("VOLUME_MULTIPLIER", "1.5"))

# Strategy — RSI
RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD: float = float(os.getenv("RSI_OVERSOLD", "30.0"))

# Strategy — Combiner
# How many strategies must agree before a BUY is placed (1 = any single strategy triggers)
STRATEGY_MIN_AGREEMENT: int = int(os.getenv("STRATEGY_MIN_AGREEMENT", "1"))

# Volatility filter — ATR as % of price; outside this band the symbol is skipped
ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", "14"))
ATR_MIN_PCT: float = float(os.getenv("ATR_MIN_PCT", "0.005"))   # 0.5% min movement
ATR_MAX_PCT: float = float(os.getenv("ATR_MAX_PCT", "0.08"))    # 8%  max movement

# Risk
MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "0.05"))
MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.05"))
TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.15"))
MIN_STOCK_PRICE: float = float(os.getenv("MIN_STOCK_PRICE", "1.0"))
MAX_STOCK_PRICE: float = float(os.getenv("MAX_STOCK_PRICE", "500.0"))

# Control API
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
