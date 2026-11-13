
import os
import logging
from datetime import timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from twelvedata import TDClient

# --------------------------------------------------
# НАСТРОЙКИ
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "2.0.0"

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


# --------------------------------------------------
# МОДЕЛЬ ЗАПРОСА
# --------------------------------------------------

class SignalRequest(BaseModel):
    pair: str = Field(default="EUR/USD")
    timeframe: str = Field(default="1min")


# --------------------------------------------------
# ПОЛУЧЕНИЕ РЫНОЧНЫХ ДАННЫХ
# --------------------------------------------------

def get_market_data(pair: str, timeframe: str) -> pd.DataFrame:

    if not TWELVEDATA_KEY:
        raise HTTPException(
            status_code=503,
            detail="Не задан TWELVEDATA_KEY"
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
                detail=f"Нет данных для {pair}, {timeframe}"
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

        for column in required:
            if column not in df.columns:
                raise HTTPException(
                    status_code=502,
                    detail=f"Отсутствует колонка {column}"
                )

            df[column] = pd.to_numeric(
                df[column],
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

        # Исключаем незакрытые свечи
        duration = TIMEFRAMES[timeframe]

        cutoff = (
            pd.Timestamp.now(tz="UTC")
            - duration
            - timedelta(seconds=CANDLE_SAFETY_SECONDS)
        )

        df = df[df.index <= cutoff]

        if len(df) < 60:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Недостаточно закрытых свечей: "
                    f"{pair} {timeframe}"
                )
            )

        return df.tail(CANDLE_LIMIT)

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Ошибка Twelve Data: %s %s",
            pair,
            timeframe
        )

        raise HTTPException(
            status_code=502,
            detail="Ошибка получения данных Twelve Data"
        )


# --------------------------------------------------
# ПОДТВЕРЖДЕННЫЕ SWING HIGH / LOW
# --------------------------------------------------

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


# --------------------------------------------------
# H1: НАПРАВЛЕНИЕ РЫНКА
# --------------------------------------------------

def analyze_h1(df):

    if len(df) < 60:
        return "WAIT", "Недостаточно данных H1"

    close = df["close"]

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    price = close.iloc[-1]

    if price > ema20.iloc[-1] > ema50.iloc[-1]:
        return "CALL", "H1: бычье направление EMA20 выше EMA50"

    if price < ema20.iloc[-1] < ema50.iloc[-1]:
        return "PUT", "H1: медвежье направление EMA20 ниже EMA50"

    return "WAIT", "H1: направление рынка не определено"


# --------------------------------------------------
# M15: BOS — СТРУКТУРА РЫНКА
# --------------------------------------------------

def analyze_m15(df):

    if len(df) < 30:
        return "WAIT", "Недостаточно данных M15"

    swings_high, swings_low = find_confirmed_swings(df)

    last = df.iloc[-1]
    previous = df.iloc[-2]

    bullish = False
    bearish = False

    if swings_high:
        level = swings_high[-1][1]

        bullish = (
            previous["close"] <= level
            and last["close"] > level
        )

    if swings_low:
        level = swings_low[-1][1]

        bearish = (
            previous["close"] >= level
            and last["close"] < level
        )

    if bullish and not bearish:
        return "CALL", "M15: подтвержден бычий BOS"

    if bearish and not bullish:
        return "PUT", "M15: подтвержден медвежий BOS"

    return "WAIT", "M15: нет подтвержденного BOS"


# --------------------------------------------------
# M5: LIQUIDITY SWEEP И FVG
# --------------------------------------------------

def analyze_m5(df):

    if len(df) < 30:
        return "WAIT", "Недостаточно данных M5"

    swings_high, swings_low = find_confirmed_swings(df)

    last = df.iloc[-1]
    third = df.iloc[-3]

    setups = []

    # Снятие ликвидности снизу
    if swings_low:
        low_level = swings_low[-1][1]

        bullish_sweep = (
            last["low"] < low_level
            and last["close"] > low_level
        )

        if bullish_sweep:
            setups.append("CALL")

    # Снятие ликвидности сверху
    if swings_high:
        high_level = swings_high[-1][1]

        bearish_sweep = (
            last["high"] > high_level
            and last["close"] < high_level
        )

        if bearish_sweep:
            setups.append("PUT")

    # Бычий FVG
    bullish_fvg = (
        last["low"] > third["high"]
        and last["close"] > last["open"]
    )

    # Медвежий FVG
    bearish_fvg = (
        last["high"] < third["low"]
        and last["close"] < last["open"]
    )

    if bullish_fvg:
        setups.append("CALL")

    if bearish_fvg:
        setups.append("PUT")

    directions = set(setups)

    if directions == {"CALL"}:
        return "CALL", "M5: обнаружен бычий Sweep или FVG"

    if directions == {"PUT"}:
        return "PUT", "M5: обнаружен медвежий Sweep или FVG"

    if len(directions) > 1:
        return "WAIT", "M5: противоречивые сигналы"

    return "WAIT", "M5: нет подтвержденной модели"


# --------------------------------------------------
# M1: ПОДТВЕРЖДЕНИЕ ВХОДА
# --------------------------------------------------

def analyze_m1(df):

    if len(df) < 10:
        return "WAIT", "Недостаточно данных M1"

    last = df.iloc[-1]
    previous = df.iloc[-2]

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
        return "CALL", "M1: бычье подтверждение пробоем"

    if bearish:
        return "PUT", "M1: медвежье подтверждение пробоем"

    return "WAIT", "M1: нет подтверждения входа"


# --------------------------------------------------
# ОБЩИЙ МУЛЬТИТАЙМФРЕЙМОВЫЙ АНАЛИЗ
# --------------------------------------------------

def analyze_multi_timeframe(data):

    h1_signal, h1_reason = analyze_h1(data["1h"])
    m15_signal, m15_reason = analyze_m15(data["15min"])
    m5_signal, m5_reason = analyze_m5(data["5min"])
    m1_signal, m1_reason = analyze_m1(data["1min"])

    signals = [
        h1_signal,
        m15_signal,
        m5_signal,
        m1_signal
    ]

    # Сигнал только при совпадении всех четырех условий
    if all(s == "CALL" for s in signals):
        final_signal = "CALL"
        final_reason = "Все четыре таймфрейма подтвердили CALL"

    elif all(s == "PUT" for s in signals):
        final_signal = "PUT"
        final_reason = "Все четыре таймфрейма подтвердили PUT"

    else:
        final_signal = "WAIT"
        final_reason = "Нет полного согласования таймфреймов"

    return {
        "signal": final_signal,
        "reason": final_reason,
        "analysis": {
            "H1": {
                "signal": h1_signal,
                "reason": h1_reason
            },
            "M15": {
                "signal": m15_signal,
                "reason": m15_reason
            },
            "M5": {
                "signal": m5_signal,
                "reason": m5_reason
            },
            "M1": {
                "signal": m1_signal,
                "reason": m1_reason
            }
        }
    }


# --------------------------------------------------
# API: ПОЛУЧЕНИЕ СИГНАЛА
# --------------------------------------------------

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

    # Загружаем четыре таймфрейма
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
        "analysis": result["analysis"]
    }


# --------------------------------------------------
# ПРОВЕРКА СЕРВЕРА
# --------------------------------------------------

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


# --------------------------------------------------
# ЗАПУСК
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
