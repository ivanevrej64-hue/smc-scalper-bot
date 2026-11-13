
import os
import logging
from datetime import timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from twelvedata import TDClient

# ==================================================
# CONFIGURATION
# ==================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "2.1.0"

app = FastAPI(
    title="SMC Signal Bot",
    version=VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY")

ALLOWED_PAIRS = {
    "EUR/USD", "GBP/USD", "USD/JPY",
    "AUD/USD", "USD/CHF", "EUR/JPY"
}

TIMEFRAMES = {
    "1min": timedelta(minutes=1),
    "5min": timedelta(minutes=5),
    "15min": timedelta(minutes=15),
    "1h": timedelta(hours=1),
}

ANALYSIS_TIMEFRAMES = ["1h", "15min", "5min", "1min"]

CANDLE_LIMIT = 150
SWING_LENGTH = 5
CANDLE_SAFETY_SECONDS = 3
MIN_SCORE = 70


# ==================================================
# REQUEST MODEL
# ==================================================

class SignalRequest(BaseModel):
    pair: str = Field(default="EUR/USD")
    timeframe: str = Field(default="1min")


# ==================================================
# MARKET DATA
# ==================================================

def get_market_data(pair: str, timeframe: str) -> pd.DataFrame:

    if not TWELVEDATA_KEY:
        raise HTTPException(
            status_code=503,
            detail="TWELVEDATA_KEY не задан в Render"
        )

    try:
        td = TDClient(apikey=TWELVEDATA_KEY)

        response = td.time_series(
            symbol=pair,
            interval=timeframe,
            outputsize=CANDLE_LIMIT,
            timezone="UTC"
        )

        df = response.as_pandas()

        if df is None or df.empty:
            raise HTTPException(
                status_code=502,
                detail=f"Нет данных: {pair} {timeframe}"
            )

        df.columns = [str(c).lower() for c in df.columns]

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["datetime"],
                utc=True,
                errors="coerce"
            )
            df = df.set_index("datetime")
        else:
            df.index = pd.to_datetime(
                df.index,
                utc=True,
                errors="coerce"
            )

        df = df.sort_index()
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep="last")]

        required = ["open", "high", "low", "close"]

        for col in required:
            if col not in df.columns:
                raise HTTPException(
                    status_code=502,
                    detail=f"Отсутствует колонка {col}"
                )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(subset=required)

        df = df[
            (df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
        ]

        # Используем только закрытые свечи
        cutoff = (
            pd.Timestamp.now(tz="UTC")
            - TIMEFRAMES[timeframe]
            - timedelta(seconds=CANDLE_SAFETY_SECONDS)
        )

        df = df[df.index <= cutoff]

        if len(df) < 60:
            raise HTTPException(
                status_code=502,
                detail=f"Недостаточно закрытых свечей: {pair} {timeframe}"
            )

        return df.tail(CANDLE_LIMIT)

    except HTTPException:
        raise

    except Exception:
        logger.exception("Ошибка получения данных %s %s", pair, timeframe)

        raise HTTPException(
            status_code=502,
            detail="Ошибка получения данных Twelve Data"
        )


# ==================================================
# CONFIRMED SWINGS
# ==================================================

def find_confirmed_swings(df, length=SWING_LENGTH):

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    swing_highs = []
    swing_lows = []

    for i in range(length, len(df) - length):

        high_window = highs[i-length:i+length+1]
        low_window = lows[i-length:i+length+1]

        if highs[i] == max(high_window):
            swing_highs.append((i, float(highs[i])))

        if lows[i] == min(low_window):
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


# ==================================================
# H1: MARKET DIRECTION
# ==================================================

def analyze_h1(df):

    close = df["close"]

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    price = float(close.iloc[-1])
    fast = float(ema20.iloc[-1])
    slow = float(ema50.iloc[-1])

    if price > fast > slow:
        return {
            "signal": "CALL",
            "reason": "Бычье направление: цена выше EMA20, EMA20 выше EMA50",
            "score": 20
        }

    if price < fast < slow:
        return {
            "signal": "PUT",
            "reason": "Медвежье направление: цена ниже EMA20, EMA20 ниже EMA50",
            "score": 20
        }

    return {
        "signal": "WAIT",
        "reason": "EMA не подтверждают единое направление",
        "score": 0
    }


# ==================================================
# M15: BOS / CHoCH
# ==================================================

def analyze_m15(df):

    swing_highs, swing_lows = find_confirmed_swings(df)

    if not swing_highs or not swing_lows:
        return {
            "signal": "WAIT",
            "structure": "NONE",
            "reason": "Недостаточно подтвержденных Swing High / Low",
            "score": 0
        }

    last = df.iloc[-1]
    previous = df.iloc[-2]

    high_level = swing_highs[-1][1]
    low_level = swing_lows[-1][1]

    bullish_break = (
        previous["close"] <= high_level
        and last["close"] > high_level
    )

    bearish_break = (
        previous["close"] >= low_level
        and last["close"] < low_level
    )

    if bullish_break and not bearish_break:
        return {
            "signal": "CALL",
            "structure": "BOS",
            "reason": "Закрытие свечи выше подтвержденного Swing High",
            "score": 25
        }

    if bearish_break and not bullish_break:
        return {
            "signal": "PUT",
            "structure": "BOS",
            "reason": "Закрытие свечи ниже подтвержденного Swing Low",
            "score": 25
        }

    # Предварительная оценка направления структуры
    recent_highs = [x[1] for x in swing_highs[-2:]]
    recent_lows = [x[1] for x in swing_lows[-2:]]

    if len(recent_highs) == 2 and len(recent_lows) == 2:

        bullish_structure = (
            recent_highs[-1] > recent_highs[-2]
            and recent_lows[-1] > recent_lows[-2]
        )

        bearish_structure = (
            recent_highs[-1] < recent_highs[-2]
            and recent_lows[-1] < recent_lows[-2]
        )

        if bearish_break and bullish_structure:
            return {
                "signal": "PUT",
                "structure": "CHoCH",
                "reason": "Медвежий CHoCH: пробой против предыдущей бычьей структуры",
                "score": 25
            }

        if bullish_break and bearish_structure:
            return {
                "signal": "CALL",
                "structure": "CHoCH",
                "reason": "Бычий CHoCH: пробой против предыдущей медвежьей структуры",
                "score": 25
            }

    return {
        "signal": "WAIT",
        "structure": "NONE",
        "reason": "Нет нового BOS или CHoCH",
        "score": 0
    }


# ==================================================
# M5: LIQUIDITY SWEEP
# ==================================================

def detect_liquidity_sweep(df):

    swing_highs, swing_lows = find_confirmed_swings(df)

    last = df.iloc[-1]

    bullish = False
    bearish = False

    if swing_lows:
        level = swing_lows[-1][1]

        bullish = (
            last["low"] < level
            and last["close"] > level
        )

    if swing_highs:
        level = swing_highs[-1][1]

        bearish = (
            last["high"] > level
            and last["close"] < level
        )

    return bullish, bearish


# ==================================================
# M5: FAIR VALUE GAP
# ==================================================

def detect_fvg(df):

    last = df.iloc[-1]
    third = df.iloc[-3]

    bullish = last["low"] > third["high"]
    bearish = last["high"] < third["low"]

    if bullish and not bearish:
        return "CALL"

    if bearish and not bullish:
        return "PUT"

    return "WAIT"


# ==================================================
# M5: SIMPLE ORDER BLOCK
# ==================================================

def detect_order_block(df):

    last = df.iloc[-1]

    # Ищем последнюю противоположную свечу
    # перед подтвержденным движением.
    for i in range(len(df) - 2, max(0, len(df) - 9), -1):

        candle = df.iloc[i]

        bullish_ob = (
            candle["close"] < candle["open"]
            and last["close"] > candle["high"]
        )

        bearish_ob = (
            candle["close"] > candle["open"]
            and last["close"] < candle["low"]
        )

        if bullish_ob:
            return "CALL"

        if bearish_ob:
            return "PUT"

    return "WAIT"


# ==================================================
# M5: COMBINED SETUP
# ==================================================

def analyze_m5(df):

    sweep_call, sweep_put = detect_liquidity_sweep(df)

    fvg_signal = detect_fvg(df)
    ob_signal = detect_order_block(df)

    call_points = 0
    put_points = 0

    if sweep_call:
        call_points += 20

    if sweep_put:
        put_points += 20

    if fvg_signal == "CALL":
        call_points += 15

    if fvg_signal == "PUT":
        put_points += 15

    if ob_signal == "CALL":
        call_points += 10

    if ob_signal == "PUT":
        put_points += 10

    if call_points > put_points and call_points > 0:
        signal = "CALL"
        score = call_points
    elif put_points > call_points and put_points > 0:
        signal = "PUT"
        score = put_points
    else:
        signal = "WAIT"
        score = 0

    return {
        "signal": signal,
        "reason": "Анализ Sweep, FVG и Order Block",
        "liquidity_sweep": {
            "bullish": sweep_call,
            "bearish": sweep_put
        },
        "fvg": fvg_signal,
        "order_block": ob_signal,
        "score": score
    }


# ==================================================
# M1: ENTRY CONFIRMATION
# ==================================================

def analyze_m1(df):

    last = df.iloc[-1]
    recent = df.iloc[-5:-1]

    bullish = (
        last["close"] > last["open"]
        and last["close"] > recent["high"].max()
    )

    bearish = (
        last["close"] < last["open"]
        and last["close"] < recent["low"].min()
    )

    if bullish:
        return {
            "signal": "CALL",
            "reason": "Бычье закрытие выше максимума предыдущих четырех свечей",
            "score": 10
        }

    if bearish:
        return {
            "signal": "PUT",
            "reason": "Медвежье закрытие ниже минимума предыдущих четырех свечей",
            "score": 10
        }

    return {
        "signal": "WAIT",
        "reason": "Нет подтверждения пробоем на M1",
        "score": 0
    }


# ==================================================
# MULTI-TIMEFRAME ENGINE
# ==================================================

def analyze_multi_timeframe(data):

    h1 = analyze_h1(data["1h"])
    m15 = analyze_m15(data["15min"])
    m5 = analyze_m5(data["5min"])
    m1 = analyze_m1(data["1min"])

    components = {
        "H1": h1,
        "M15": m15,
        "M5": m5,
        "M1": m1
    }

    directions = [
        h1["signal"],
        m15["signal"],
        m5["signal"],
        m1["signal"]
    ]

    call_score = 0
    put_score = 0

    if h1["signal"] == "CALL":
        call_score += 20

    if h1["signal"] == "PUT":
        put_score += 20

    if m15["signal"] == "CALL":
        call_score += 25

    if m15["signal"] == "PUT":
        put_score += 25

    if m5["signal"] == "CALL":
        call_score += m5["score"]

    if m5["signal"] == "PUT":
        put_score += m5["score"]

    if m1["signal"] == "CALL":
        call_score += 10

    if m1["signal"] == "PUT":
        put_score += 10

    final_signal = "WAIT"
    reason = "Не выполнены обязательные условия входа"

    # Обязательные условия:
    # H1 + M15 + M5 + M1 должны совпадать.
    if all(s == "CALL" for s in directions) and call_score >= MIN_SCORE:
        final_signal = "CALL"
        reason = "Все таймфреймы подтвердили бычий сетап"

    elif all(s == "PUT" for s in directions) and put_score >= MIN_SCORE:
        final_signal = "PUT"
        reason = "Все таймфреймы подтвердили медвежий сетап"

    else:
        if h1["signal"] == "WAIT":
            reason = "Ожидание: H1 не определил направление"
        elif m15["signal"] == "WAIT":
            reason = "Ожидание: M15 не подтвердил BOS или CHoCH"
        elif m5["signal"] == "WAIT":
            reason = "Ожидание: M5 не подтвердил торговую модель"
        elif m1["signal"] == "WAIT":
            reason = "Ожидание: M1 не подтвердил точку входа"
        else:
            reason = "Ожидание: направления таймфреймов противоречат друг другу"

    return {
        "signal": final_signal,
        "reason": reason,
        "score": {
            "CALL": call_score,
            "PUT": put_score,
            "minimum_required": MIN_SCORE
        },
        "analysis": components
    }


# ==================================================
# API: SIGNAL
# ==================================================

@app.post("/api/signal")
def get_signal(req: SignalRequest):

    pair = req.pair.strip().upper()
    requested_timeframe = req.timeframe.strip()

    if pair not in ALLOWED_PAIRS:
        raise HTTPException(
            status_code=400,
            detail="Эта валютная пара не поддерживается"
        )

    if requested_timeframe not in TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail="Недопустимый таймфрейм"
        )

    market_data = {}

    for tf in ANALYSIS_TIMEFRAMES:
        market_data[tf] = get_market_data(pair, tf)

    result = analyze_multi_timeframe(market_data)

    last_m1 = market_data["1min"].iloc[-1]

    return {
        "signal": result["signal"],
        "reason": result["reason"],
        "pair": pair,
        "timeframe": "1min",
        "requested_timeframe": requested_timeframe,
        "price": float(last_m1["close"]),
        "candle_time": market_data["1min"].index[-1].isoformat(),
        "candles_used": {
            tf: len(market_data[tf])
            for tf in ANALYSIS_TIMEFRAMES
        },
        "strategy": "SMC Multi-Timeframe",
        "strategy_version": VERSION,
        "score": result["score"],
        "analysis": result["analysis"]
    }


# ==================================================
# HEALTH
# ==================================================

@app.get("/")
def health():

    return {
        "status": "ok",
        "service": "SMC Signal Bot",
        "version": VERSION
    }


@app.get("/api/health")
def api_health():

    return {
        "status": "online",
        "twelvedata_key_configured": bool(TWELVEDATA_KEY)
    }


# ==================================================
# LOCAL START
# ==================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
