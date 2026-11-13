from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from twelvedata import TDClient
from smartmoneyconcepts import smc
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TWELVEDATA_KEY = os.environ.get("TWELVEDATA_KEY") 
td = TDClient(apikey=TWELVEDATA_KEY)

class SignalRequest(BaseModel):
    pair: str = "EUR/USD"
    timeframe: str = "1min"

@app.post("/api/signal")
async def get_signal(req: SignalRequest):
    try:
        # Для скальпинга берем 50 свечей
        ts = td.time_series(symbol=req.pair, interval=req.timeframe, outputsize=50)
        df = ts.as_pandas()
        
        df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
        
        # Анализ SMC
        swing_hl = smc.swing_highs_lows(df, swing_length=5) # Уменьшили длину для M1-M5
        fvg = smc.fvg(df)
        bos_choch = smc.bos_choch(df, swing_hl)
        
        last_idx = len(df) - 1
        
        signal = "WAIT"
        reason = "Поиск быстрого импульса..."
        
        # 1. Проверяем слом структуры (BOS)
        if bos_choch['BOS'].iloc[last_idx] == 1:
            signal = "CALL"
            reason = "Бычий BOS (M1). Импульс вверх."
        elif bos_choch['BOS'].iloc[last_idx] == -1:
            signal = "PUT"
            reason = "Медвежий BOS (M1). Импульс вниз."
        
        # 2. Если BOS нет, ищем отбой от Fair Value Gap (FVG)
        if signal == "WAIT":
            if fvg['FVG'].iloc[last_idx] != 0:
                if fvg['FVG'].iloc[last_idx] > 0 and df['close'].iloc[last_idx] > df['open'].iloc[last_idx]:
                    signal = "CALL"
                    reason = "Отбой от бычьего FVG."
                elif fvg['FVG'].iloc[last_idx] < 0 and df['close'].iloc[last_idx] < df['open'].iloc[last_idx]:
                    signal = "PUT"
                    reason = "Отбой от медвежьего FVG."

        return {
            "signal": signal,
            "reason": reason,
            "pair": req.pair,
            "price": float(df['close'].iloc[-1])
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
