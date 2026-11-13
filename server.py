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
        reason = "Поиск быстрого импульса..."

        # 1. Структура: ищем пробой, произошедший именно на последней свече
        broken_now = bos_choch[bos_choch["BrokenIndex"] == last_idx]
        if not broken_now.empty:
            row = broken_now.iloc[-1]
            if row["BOS"] == 1:
                signal, reason = "CALL", "Бычий BOS. Импульс вверх."
            elif row["BOS"] == -1:
                signal, reason = "PUT", "Медвежий BOS. Импульс вниз."
            elif row["CHOCH"] == 1:
                signal, reason = "CALL", "Бычий CHoCH. Разворот вверх."
            elif row["CHOCH"] == -1:
                signal, reason = "PUT", "Медвежий CHoCH. Разворот вниз."

        # 2. Если пробоя нет, ищем отбой от последнего неотработанного FVG
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
                    signal, reason = "CALL", "Отбой от бычьего FVG."
                elif zone["FVG"] == -1 and last["high"] >= zone["Bottom"] and bearish_candle:
                    signal, reason = "PUT", "Отбой от медвежьего FVG."

        return {
            "signal": signal,
            "reason": reason,
            "pair": req.pair,
            "price": float(last["close"]),
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
