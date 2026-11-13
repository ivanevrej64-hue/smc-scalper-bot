from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from twelvedata import TDClient
from smartmoneyconcepts import smc
import os

app = FastAPI()

# Разрешаем запросы из Telegram
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ключ Twelve Data (мы передадим его через Render)
TWELVEDATA_KEY = os.environ.get("TWELVEDATA_KEY") 
td = TDClient(apikey=TWELVEDATA_KEY)

class SignalRequest(BaseModel):
    pair: str = "EUR/USD"
    timeframe: str = "5min"

@app.post("/api/signal")
async def get_signal(req: SignalRequest):
    try:
        ts = td.time_series(symbol=req.pair, interval=req.timeframe, outputsize=100)
        df = ts.as_pandas()
        
        # Переименовываем колонки для библиотеки SMC
        df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
        
        swing_hl = smc.swing_highs_lows(df, swing_length=10)
        fvg = smc.fvg(df)
        bos_choch = smc.bos_choch(df, swing_hl)
        
        last_idx = len(df) - 1
        signal = "WAIT"
        reason = "Нет четкого сетапа SMC"
        
        if bos_choch['BOS'].iloc[last_idx] == 1:
            signal = "CALL"
            reason = "Бычий BOS (слом структуры вверх)."
        elif bos_choch['BOS'].iloc[last_idx] == -1:
            signal = "PUT"
            reason = "Медвежий BOS (слом структуры вниз)."
        
        if fvg['FVG'].iloc[last_idx] != 0:
            reason += " + Обнаружен FVG (имбаланс)."
            
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