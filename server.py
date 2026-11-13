
import os
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from smartmoneyconcepts import smc
from twelvedata import TDClient

# ---------------- CONFIG ----------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SMC Trading Signal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TWELVEDATA_KEY = os.environ.get("TWELVEDATA_KEY")
td = TDClient(apikey=TWELVEDATA_KEY) if TWELVEDATA_KEY else None

ALLOWED_TF = {
    "1min", "5min", "15min", "30min",
    "45min", "1h", "2h", "4h"
}

LOOKBACK = 3
OUTPUT_SIZE = 200


# ---------------- REQUEST MODEL ----------------

class SignalRequest(BaseModel):
    pair: str = Field(default="EUR/USD", min_length=3, max_length=30)
    timeframe: str = "1min"


# ---------------- MARKET DATA ----------------

def load_ohlc(pair: str, timeframe: str):

    if td is None:
        raise ValueError("TWELVEDATA_KEY не задан")

    try:
        ts = td.time_series(
            symbol=pair,
            interval=timeframe,
            outputsize=OUTPUT_SIZE
        )

        df = ts.as_pandas()

    except Exception as e:
        logger.exception("Twelve Data API error")
        raise ValueError(f"Ошибка получения котировок: {str(e)}")

    if df is None or df.empty:
        raise ValueError("Twelve Data не вернул данные")

    # Приводим названия столбцов к нижнему регистру
    df.columns = [str(c).lower() for c in df.columns]

    required = ["open", "high", "low", "close"]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Отсутствует столбец {column}")

    # Старые свечи должны идти первыми
    df = df.sort_index()

    # Если объём отсутствует, добавляем нулевой
    if "volume" not in df.columns:
        df["volume"] = 0

    # Приведение значений к числовому формату
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(
        subset=["open", "high", "low", "close"]
    )

    # Удаляем последнюю незакрытую свечу
    if len(df) > 1:
        df = df.iloc[:-1]

    df = df[["open", "high", "low", "close", "volume"]]

    # Индекс 0..N-1 для SMC
    df = df.reset_index(drop=True)

    if len(df) < 30:
        raise ValueError("Недостаточно свечей для анализа")

    return df


# ---------------- SMC ANALYSIS ----------------

def analyze_market(df):

    swing_hl = smc.swing_highs_lows(
        df,
        swing_length=5
    )

    bos_choch = smc.bos_choch(
        df,
        swing_hl
    )

    fvg = smc.fvg(df)

    last_idx = len(df) - 1
    last = df.iloc[last_idx]

    signal = "WAIT"
    reason = "Нет подтверждённого торгового события"

    bias = "нет данных"
    last_event = None

    # ---------------- BOS / CHoCH ----------------

    events = bos_choch.copy()

    events["BrokenIndex"] = pd.to_numeric(
        events["BrokenIndex"],
        errors="coerce"
    )

    events = events[
        events["BrokenIndex"].notna()
        & (events["BrokenIndex"] <= last_idx)
    ]

    if not events.empty:

        events = events.sort_values("BrokenIndex")

        ev = events.iloc[-1]

        bos = ev["BOS"]
        choch = ev["CHOCH"]

        direction = None
        kind = None

        if pd.notna(bos) and bos in (1, -1):
            direction = int(bos)
            kind = "BOS"

        elif pd.notna(choch) and choch in (1, -1):
            direction = int(choch)
            kind = "CHoCH"

        if direction is not None:

            broken_index = int(ev["BrokenIndex"])
            age = last_idx - broken_index

            bias = "бычий" if direction == 1 else "медвежий"

            last_event = {
                "type": kind,
                "dir": direction,
                "candles_ago": age
            }

            if 0 <= age < LOOKBACK:

                if direction == 1:
                    signal = "CALL"
                    reason = f"Бычий {kind}, {age} свечей назад"

                else:
                    signal = "PUT"
                    reason = f"Медвежий {kind}, {age} свечей назад"

    # ---------------- FVG RETEST ----------------

    if signal == "WAIT":

        active = fvg[
            fvg["FVG"].notna()
            & fvg["MitigatedIndex"].isna()
            & (fvg.index < last_idx - 1)
        ]

        # Проверяем зоны от самой новой к старой
        if not active.empty:

            for idx in reversed(active.index.tolist()):

                zone = active.loc[idx]

                top = float(zone["Top"])
                bottom = float(zone["Bottom"])
                direction = int(zone["FVG"])

                # Проверка фактического пересечения свечи с зоной
                overlaps = (
                    last["low"] <= top
                    and last["high"] >= bottom
                )

                if not overlaps:
                    continue

                bullish_candle = last["close"] > last["open"]
                bearish_candle = last["close"] < last["open"]

                if direction == 1:

                    if bullish_candle and last["close"] > bottom:
                        signal = "CALL"
                        reason = "Подтверждённый отбой от бычьего FVG"
                        break

                elif direction == -1:

                    if bearish_candle and last["close"] < top:
                        signal = "PUT"
                        reason = "Подтверждённый отбой от медвежьего FVG"
                        break

            if signal == "WAIT":
                reason = "Нет подтверждённого отбоя от FVG"

    return {
        "signal": signal,
        "reason": f"{reason} | Тренд: {bias}",
        "price": float(last["close"]),
        "debug": {
            "last_event": last_event,
            "candles": len(df)
        }
    }


# ---------------- API ENDPOINT ----------------

@app.post("/api/signal")
def get_signal(req: SignalRequest):

    if td is None:
        raise HTTPException(
            status_code=500,
            detail="TWELVEDATA_KEY не задан"
        )

    pair = req.pair.strip().upper()

    if not pair:
        raise HTTPException(
            status_code=400,
            detail="Укажите валютную пару"
        )

    if req.timeframe not in ALLOWED_TF:
        raise HTTPException(
            status_code=400,
            detail=f"Недопустимый timeframe: {sorted(ALLOWED_TF)}"
        )

    try:

        df = load_ohlc(pair, req.timeframe)

        result = analyze_market(df)

        result["pair"] = pair
        result["timeframe"] = req.timeframe

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=502,
            detail=str(e)
        )

    except Exception as e:

        logger.exception("Signal analysis error")

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка анализа: {str(e)}"
        )


# ---------------- HEALTH CHECK ----------------

@app.get("/")
def health():

    return {
        "status": "ok",
        "service": "SMC Trading Signal API",
        "twelvedata": td is not None
    }


# ---------------- START SERVER ----------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
