import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from smartmoneyconcepts import smc
from twelvedata import TDClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # в продакшене укажи свой домен
    allow_methods=["*"],
    allow_headers=["*"],
)

TWELVEDATA_KEY = os.environ.get("TWELVEDATA_KEY")
td = TDClient(apikey=TWELVEDATA_KEY) if TWELVEDATA_KEY else None

ALLOWED_TF = {"1min", "5min", "15min", "30min", "45min", "1h", "2h", "4h"}

LOOKBACK = 3  # сколько последних закрытых свечей считаем "свежим" событием


class SignalRequest(BaseModel):
    pair: str = "EUR/USD"
    timeframe: str = "1min"


def load_ohlc(pair: str, timeframe: str):
    ts = td.time_series(symbol=pair, interval=timeframe, outputsize=150)
    df = ts.as_pandas()

    if df is None or df.empty:
        raise ValueError("Нет данных от Twelve Data")

    # Twelve Data отдаёт новые свечи первыми, а SMC ждёт старые первыми
    df = df.sort_index()

    # У форекса нет объёма
    if "volume" not in df.columns:
        df["volume"] = 0

    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Отбрасываем ещё не закрытую свечу, чтобы сигнал не перерисовывался
    df = df.iloc[:-1]

    # Позиционный индекс 0..N-1, с ним работают BrokenIndex/MitigatedIndex
    return df.reset_index(drop=True)


@app.post("/api/signal")
def get_signal(req: SignalRequest):
    if td is None:
        raise HTTPException(status_code=500, detail="TWELVEDATA_KEY не задан")
    if req.timeframe not in ALLOWED_TF:
        raise HTTPException(
            status_code=400,
            detail=f"timeframe должен быть одним из: {sorted(ALLOWED_TF)}",
        )

    try:
        df = load_ohlc(req.pair, req.timeframe)
        if len(df) < 30:
            raise ValueError("Слишком мало свечей для анализа")

        swing_hl = smc.swing_highs_lows(df, swing_length=5)
        bos_choch = smc.bos_choch(df, swing_hl)
        fvg = smc.fvg(df)

        last_idx = len(df) - 1
        last = df.iloc[last_idx]

        signal = "WAIT"
        reason = "Нет свежего пробоя структуры или отбоя от FVG"

        # --- Структура: последний пробой и его давность ---
        events = bos_choch[bos_choch["BrokenIndex"].notna()].copy()
        last_event = None
        bias = "нет данных"
        if not events.empty:
            events = events.sort_values("BrokenIndex")
            ev = events.iloc[-1]
            age = last_idx - int(ev["BrokenIndex"])
            direction = ev["BOS"] if ev["BOS"] in (1, -1) else ev["CHOCH"]
            kind = "BOS" if ev["BOS"] in (1, -1) else "CHoCH"
            bias = "бычий" if direction == 1 else "медвежий"
            last_event = {"type": kind, "dir": int(direction), "candles_ago": age}

            if age < LOOKBACK:
                if direction == 1:
                    signal, reason = "CALL", f"Бычий {kind} ({age} св. назад)"
                else:
                    signal, reason = "PUT", f"Медвежий {kind} ({age} св. назад)"

        # --- FVG: ретест зоны, если свежего пробоя нет ---
        if signal == "WAIT":
            active = fvg[
                fvg["FVG"].notna()
                & (fvg.index < last_idx - 1)
                & (fvg["MitigatedIndex"].fillna(0) == 0)
            ]
            if not active.empty:
                zone = active.iloc[-1]
                bullish_candle = last["close"] > last["open"]
                bearish_candle = last["close"] < last["open"]
                if zone["FVG"] == 1 and last["low"] <= zone["Top"] and bullish_candle:
                    signal, reason = "CALL", "Отбой от бычьего FVG"
                elif zone["FVG"] == -1 and last["high"] >= zone["Bottom"] and bearish_candle:
                    signal, reason = "PUT", "Отбой от медвежьего FVG"
                else:
                    reason += f" | Активный FVG: {'бычий' if zone['FVG'] == 1 else 'медвежий'}"

        return {
            "signal": signal,
            "reason": f"{reason} | Тренд: {bias}",
            "pair": req.pair,
            "price": float(last["close"]),
            "debug": {"last_event": last_event, "candles": len(df)},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
