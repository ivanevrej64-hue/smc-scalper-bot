
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

app = FastAPI(
    title="SMC Signal Bot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production ограничить своим доменом
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY")

ALLOWED_PAIRS = {
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "USD/CHF",
    "EUR/JPY",
}

TIMEFRAMES = {
    "1min": timedelta(minutes=1),
    "5min": timedelta(minutes=5),
    "15min": timedelta(minutes=15),
    "1h": timedelta(hours=1),
}

SWING_LENGTH = 5
CANDLE_LIMIT = 150
CANDLE_SAFETY_SECONDS = 3


# --------------------------------------------------
# МОДЕЛЬ ЗАПРОСА
# --------------------------------------------------

class SignalRequest(BaseModel):
    pair: str = Field(default="EUR/USD")
    timeframe: str = Field(default="1min")


# --------------------------------------------------
# ПОЛУЧЕНИЕ ДАННЫХ TWELVE DATA
# --------------------------------------------------

def get_market_data(pair: str, timeframe: str) -> pd.DataFrame:

    if not TWELVEDATA_KEY:
        raise HTTPException(
            status_code=503,
            detail="Не задан TWELVEDATA_KEY в переменных окружения"
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
                detail="Twelve Data не вернул свечи"
            )

        df.columns = [str(col).lower() for col in df.columns]

        # Нормализация времени
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
                    detail=f"В данных отсутствует колонка {column}"
                )

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna(subset=required)

        # Проверяем корректность OHLC
        df = df[
            (df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
        ]

        # Исключаем незакрытые свечи
        duration = TIMEFRAMES[timeframe]
        now = pd.Timestamp.now(tz="UTC")
        cutoff = now - duration - timedelta(
            seconds=CANDLE_SAFETY_SECONDS
        )

        df = df[df.index <= cutoff]

        if len(df) < 30:
            raise HTTPException(
                status_code=502,
                detail="Недостаточно закрытых свечей для анализа"
            )

        return df.tail(CANDLE_LIMIT)

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Ошибка получения рыночных данных")

        raise HTTPException(
            status_code=502,
            detail="Не удалось получить данные Twelve Data"
        )


# --------------------------------------------------
# ПОИСК ПОДТВЕРЖДЕННЫХ SWING HIGH / SWING LOW
# --------------------------------------------------

def find_confirmed_swings(df: pd.DataFrame, length: int):

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    swing_highs = []
    swing_lows = []

    # Последние length свечей не используются:
    # у них еще нет полного подтверждения справа.
    for i in range(length, len(df) - length):

        left = i - length
        right = i + length + 1

        high_window = highs[left:right]
        low_window = lows[left:right]

        if highs[i] == max(high_window):
            swing_highs.append((i, float(highs[i])))

        if lows[i] == min(low_window):
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


# --------------------------------------------------
# SMC АНАЛИЗ
# --------------------------------------------------

def analyze_smc(df: pd.DataFrame):

    if len(df) < 30:
        return "WAIT", "Недостаточно данных"

    last = df.iloc[-1]
    previous = df.iloc[-2]

    swing_highs, swing_lows = find_confirmed_swings(
        df,
        SWING_LENGTH
    )

    signal = "WAIT"
    reason = "Нет подтвержденного торгового условия"

    # 1. BOS — пробой подтвержденного Swing High
    if swing_highs:
        swing_high_index, swing_high = swing_highs[-1]

        bullish_break = (
            previous["close"] <= swing_high
            and last["close"] > swing_high
        )

        if bullish_break:
            signal = "CALL"
            reason = "Бычий BOS: закрытие свечи выше подтвержденного Swing High"

    # 2. BOS — пробой подтвержденного Swing Low
    if signal == "WAIT" and swing_lows:
        swing_low_index, swing_low = swing_lows[-1]

        bearish_break = (
            previous["close"] >= swing_low
            and last["close"] < swing_low
        )

        if bearish_break:
            signal = "PUT"
            reason = "Медвежий BOS: закрытие свечи ниже подтвержденного Swing Low"

    # 3. FVG — трехсвечная модель.
    # Она становится известна только после закрытия последней свечи.
    if signal == "WAIT" and len(df) >= 3:

        candle_2 = df.iloc[-3]

        bullish_fvg = (
            last["low"] > candle_2["high"]
            and last["close"] > last["open"]
        )

        bearish_fvg = (
            last["high"] < candle_2["low"]
            and last["close"] < last["open"]
        )

        if bullish_fvg:
            signal = "CALL"
            reason = "Образован бычий FVG на закрытой свече"

        elif bearish_fvg:
            signal = "PUT"
            reason = "Образован медвежий FVG на закрытой свече"

    return signal, reason


# --------------------------------------------------
# API: ПОЛУЧЕНИЕ СИГНАЛА
# --------------------------------------------------

@app.post("/api/signal")
def get_signal(req: SignalRequest):

    pair = req.pair.strip().upper()
    timeframe = req.timeframe.strip()

    if pair not in ALLOWED_PAIRS:
        raise HTTPException(
            status_code=400,
            detail="Эта валютная пара не поддерживается"
        )

    if timeframe not in TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail="Недопустимый таймфрейм"
        )

    df = get_market_data(pair, timeframe)

    signal, reason = analyze_smc(df)

    return {
        "signal": signal,
        "reason": reason,
        "pair": pair,
        "timeframe": timeframe,
        "price": float(df["close"].iloc[-1]),
        "candle_time": df.index[-1].isoformat(),
        "candles_used": len(df),
        "strategy": "SMC",
        "strategy_version": "1.0.0"
    }


# --------------------------------------------------
# ПРОВЕРКА РАБОТЫ СЕРВЕРА
# --------------------------------------------------

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "SMC Signal Bot",
        "version": "1.0.0"
    }


@app.get("/api/health")
def api_health():
    return {
        "status": "online",
        "twelvedata_key_configured": bool(TWELVEDATA_KEY)
    }


# --------------------------------------------------
# ЛОКАЛЬНЫЙ ЗАПУСК
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
