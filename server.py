
import os
import math
import logging
from datetime import timedelta
from collections import deque

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from twelvedata import TDClient


# ==================================================
# CONFIGURATION
# ==================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "2.2.0"

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

ANALYSIS_TIMEFRAMES = [
    "1h", "15min", "5min", "1min"
]

CANDLE_LIMIT = 150
SWING_LENGTH = 5
CANDLE_SAFETY_SECONDS = 3

MIN_SCORE = 70

# Журнал хранит последние 500 сигналов
# и последние 500 виртуальных сделок.
SIGNAL_HISTORY = deque(maxlen=500)
VIRTUAL_TRADES = deque(maxlen=500)


# ==================================================
# JSON SAFE CONVERSION
# ==================================================

def make_json_safe(value):

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, deque)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


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

        df.columns = [
            str(c).lower()
            for c in df.columns
        ]

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

        # Используем только закрытые свечи.

        cutoff = (
            pd.Timestamp.now(tz="UTC")
            - TIMEFRAMES[timeframe]
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
            "Ошибка получения данных %s %s",
            pair,
            timeframe
        )

        raise HTTPException(
            status_code=502,
            detail="Ошибка получения данных Twelve Data"
        )


# ==================================================
# ATR
# ==================================================

def calculate_atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(
        window=period,
        min_periods=period
    ).mean()

    value = atr.iloc[-1]

    if pd.isna(value) or not math.isfinite(float(value)):
        return 0.0

    return float(value)


# ==================================================
# CONFIRMED SWINGS
# ==================================================

def find_confirmed_swings(
    df,
    length=SWING_LENGTH
):

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    swing_highs = []
    swing_lows = []

    for i in range(
        length,
        len(df) - length
    ):

        high_window = highs[
            i-length:i+length+1
        ]

        low_window = lows[
            i-length:i+length+1
        ]

        if (
            highs[i] == max(high_window)
            and np.count_nonzero(high_window == highs[i]) == 1
        ):

            swing_highs.append(
                (i, float(highs[i]))
            )

        if (
            lows[i] == min(low_window)
            and np.count_nonzero(low_window == lows[i]) == 1
        ):

            swing_lows.append(
                (i, float(lows[i]))
            )

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
            "reason": (
                "Бычье направление: цена выше EMA20, "
                "EMA20 выше EMA50"
            ),
            "score": 20,
            "ema20": fast,
            "ema50": slow
        }

    if price < fast < slow:

        return {
            "signal": "PUT",
            "reason": (
                "Медвежье направление: цена ниже EMA20, "
                "EMA20 ниже EMA50"
            ),
            "score": 20,
            "ema20": fast,
            "ema50": slow
        }

    return {
        "signal": "WAIT",
        "reason": "EMA не подтверждают единое направление",
        "score": 0,
        "ema20": fast,
        "ema50": slow
    }


# ==================================================
# M15: BOS / CHoCH
# ==================================================

def analyze_m15(df):

    swing_highs, swing_lows = find_confirmed_swings(df)

    if len(swing_highs) < 2 or len(swing_lows) < 2:

        return {
            "signal": "WAIT",
            "structure": "NONE",
            "reason": "Недостаточно подтвержденных Swing High / Low",
            "score": 0
        }

    last = df.iloc[-1]
    previous = df.iloc[-2]

    high_index, high_level = swing_highs[-1]
    low_index, low_level = swing_lows[-1]

    # Не используем слишком старые уровни.

    if len(df) - high_index > 60 or len(df) - low_index > 60:

        return {
            "signal": "WAIT",
            "structure": "NONE",
            "reason": "Последние структурные уровни слишком старые",
            "score": 0
        }

    recent_highs = [
        x[1] for x in swing_highs[-2:]
    ]

    recent_lows = [
        x[1] for x in swing_lows[-2:]
    ]

    bullish_structure = (
        recent_highs[-1] > recent_highs[-2]
        and recent_lows[-1] > recent_lows[-2]
    )

    bearish_structure = (
        recent_highs[-1] < recent_highs[-2]
        and recent_lows[-1] < recent_lows[-2]
    )

    atr = calculate_atr(df)

    buffer = atr * 0.05

    bullish_break = bool(
        previous["close"] <= high_level
        and last["close"] > high_level + buffer
    )

    bearish_break = bool(
        previous["close"] >= low_level
        and last["close"] < low_level - buffer
    )

    if bullish_break and not bearish_break:

        structure = (
            "CHoCH"
            if bearish_structure
            else "BOS"
        )

        return {
            "signal": "CALL",
            "structure": structure,
            "reason": (
                f"Бычий {structure}: закрытие выше "
                "подтвержденного Swing High"
            ),
            "score": 25,
            "level": float(high_level)
        }

    if bearish_break and not bullish_break:

        structure = (
            "CHoCH"
            if bullish_structure
            else "BOS"
        )

        return {
            "signal": "PUT",
            "structure": structure,
            "reason": (
                f"Медвежий {structure}: закрытие ниже "
                "подтвержденного Swing Low"
            ),
            "score": 25,
            "level": float(low_level)
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

    atr = calculate_atr(df)

    if atr <= 0 or len(df) < 30:
        return False, False

    bullish = False
    bearish = False

    if swing_lows:

        index, level = swing_lows[-1]

        recent_enough = len(df) - index <= 30

        bullish = bool(
            recent_enough
            and last["low"] < level - atr * 0.03
            and last["close"] > level
            and last["close"] > last["open"]
        )

    if swing_highs:

        index, level = swing_highs[-1]

        recent_enough = len(df) - index <= 30

        bearish = bool(
            recent_enough
            and last["high"] > level + atr * 0.03
            and last["close"] < level
            and last["close"] < last["open"]
        )

    return bool(bullish), bool(bearish)


# ==================================================
# M5: FAIR VALUE GAP
# ==================================================

def detect_fvg(df):

    if len(df) < 20:
        return "WAIT"

    atr = calculate_atr(df)

    if atr <= 0:
        return "WAIT"

    # Ищем свежие FVG за последние 10 свечей.

    start = max(2, len(df) - 10)

    for i in range(len(df) - 1, start - 1, -1):

        first = df.iloc[i - 2]
        middle = df.iloc[i - 1]
        third = df.iloc[i]

        body = abs(
            float(middle["close"] - middle["open"])
        )

        # Бычий FVG.

        bullish_gap = float(
            third["low"] - first["high"]
        )

        if (
            bullish_gap >= atr * 0.05
            and middle["close"] > middle["open"]
            and body >= atr * 0.5
        ):

            lower = float(first["high"])
            upper = float(third["low"])

            later = df.iloc[i + 1:]

            fully_filled = (
                not later.empty
                and float(later["low"].min()) <= lower
            )

            price = float(df["close"].iloc[-1])

            near_zone = (
                price >= lower - atr * 0.05
                and price <= upper + atr * 0.25
            )

            if not fully_filled and near_zone:
                return "CALL"

        # Медвежий FVG.

        bearish_gap = float(
            first["low"] - third["high"]
        )

        if (
            bearish_gap >= atr * 0.05
            and middle["close"] < middle["open"]
            and body >= atr * 0.5
        ):

            lower = float(third["high"])
            upper = float(first["low"])

            later = df.iloc[i + 1:]

            fully_filled = (
                not later.empty
                and float(later["high"].max()) >= upper
            )

            price = float(df["close"].iloc[-1])

            near_zone = (
                price <= upper + atr * 0.05
                and price >= lower - atr * 0.25
            )

            if not fully_filled and near_zone:
                return "PUT"

    return "WAIT"


# ==================================================
# M5: ORDER BLOCK
# ==================================================

def detect_order_block(df):

    if len(df) < 30:
        return "WAIT"

    atr = calculate_atr(df)

    if atr <= 0:
        return "WAIT"

    last = df.iloc[-1]

    # Ищем последнюю противоположную свечу
    # перед импульсным движением.

    for i in range(
        len(df) - 4,
        max(5, len(df) - 15),
        -1
    ):

        candle = df.iloc[i]

        following = df.iloc[i + 1:min(i + 4, len(df))]

        if following.empty:
            continue

        previous = df.iloc[max(0, i - 5):i]

        if previous.empty:
            continue

        previous_high = float(previous["high"].max())
        previous_low = float(previous["low"].min())

        # Бычий Order Block:
        # последняя медвежья свеча перед импульсом вверх.

        bullish_ob = (
            candle["close"] < candle["open"]
            and float(following["close"].max()) > previous_high
            and float(
                (following["close"] - following["open"]).abs().max()
            ) >= atr * 0.7
        )

        # Медвежий Order Block:
        # последняя бычья свеча перед импульсом вниз.

        bearish_ob = (
            candle["close"] > candle["open"]
            and float(following["close"].min()) < previous_low
            and float(
                (following["close"] - following["open"]).abs().max()
            ) >= atr * 0.7
        )

        zone_low = float(candle["low"])
        zone_high = float(candle["high"])

        # Цена должна вернуться к зоне.

        overlaps = (
            float(last["high"]) >= zone_low
            and float(last["low"]) <= zone_high
        )

        distance = min(
            abs(float(last["close"]) - zone_low),
            abs(float(last["close"]) - zone_high)
        )

        near_zone = distance <= atr * 0.15

        if overlaps or near_zone:

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

    # OB получает 15 баллов.
    # Это позволяет единогласному сетапу
    # набрать минимальные 70 баллов.

    if ob_signal == "CALL":
        call_points += 15

    if ob_signal == "PUT":
        put_points += 15

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
            "bullish": bool(sweep_call),
            "bearish": bool(sweep_put)
        },

        "fvg": fvg_signal,
        "order_block": ob_signal,
        "score": int(score),

        "details": {
            "sweep_call_points": 20 if sweep_call else 0,
            "sweep_put_points": 20 if sweep_put else 0,
            "fvg_call_points": 15 if fvg_signal == "CALL" else 0,
            "fvg_put_points": 15 if fvg_signal == "PUT" else 0,
            "ob_call_points": 15 if ob_signal == "CALL" else 0,
            "ob_put_points": 15 if ob_signal == "PUT" else 0
        }
    }


# ==================================================
# M1: ENTRY CONFIRMATION
# ==================================================

def analyze_m1(df):

    last = df.iloc[-1]
    recent = df.iloc[-5:-1]

    atr = calculate_atr(df)

    if recent.empty or atr <= 0:

        return {
            "signal": "WAIT",
            "reason": "Недостаточно данных для подтверждения",
            "score": 0
        }

    body = abs(
        float(last["close"] - last["open"])
    )

    bullish = bool(
        last["close"] > last["open"]
        and last["close"] > recent["high"].max() + atr * 0.03
        and body >= atr * 0.2
    )

    bearish = bool(
        last["close"] < last["open"]
        and last["close"] < recent["low"].min() - atr * 0.03
        and body >= atr * 0.2
    )

    if bullish:

        return {
            "signal": "CALL",
            "reason": (
                "Бычье закрытие выше максимума "
                "предыдущих четырех свечей"
            ),
            "score": 10
        }

    if bearish:

        return {
            "signal": "PUT",
            "reason": (
                "Медвежье закрытие ниже минимума "
                "предыдущих четырех свечей"
            ),
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

    # Все четыре таймфрейма должны совпадать.

    if (
        all(s == "CALL" for s in directions)
        and call_score >= MIN_SCORE
    ):

        final_signal = "CALL"
        reason = "Все таймфреймы подтвердили бычий сетап"

    elif (
        all(s == "PUT" for s in directions)
        and put_score >= MIN_SCORE
    ):

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

        elif len(set(directions)) > 1:

            reason = (
                "Ожидание: направления таймфреймов "
                "противоречат друг другу"
            )

        else:

            reason = "Ожидание: недостаточно баллов"

    return {
        "signal": final_signal,
        "reason": reason,

        "score": {
            "CALL": int(call_score),
            "PUT": int(put_score),
            "minimum_required": MIN_SCORE
        },

        "analysis": components
    }


# ==================================================
# VIRTUAL TRADE MANAGEMENT
# ==================================================

def settle_virtual_trades(pair, m1_df):

    for trade in VIRTUAL_TRADES:

        if trade["pair"] != pair:
            continue

        if trade["status"] != "OPEN":
            continue

        expiry_at = pd.Timestamp(
            trade["expiry_at"]
        )

        eligible = m1_df[
            m1_df.index >= expiry_at
        ]

        if eligible.empty:
            continue

        exit_candle = eligible.iloc[0]

        exit_time = eligible.index[0]

        exit_price = float(exit_candle["close"])

        entry_price = float(trade["entry_price"])

        if trade["signal"] == "CALL":

            if exit_price > entry_price:
                outcome = "WIN"

            elif exit_price < entry_price:
                outcome = "LOSS"

            else:
                outcome = "DRAW"

        else:

            if exit_price < entry_price:
                outcome = "WIN"

            elif exit_price > entry_price:
                outcome = "LOSS"

            else:
                outcome = "DRAW"

        trade["status"] = "CLOSED"
        trade["outcome"] = outcome
        trade["exit_price"] = exit_price
        trade["exit_time"] = exit_time.isoformat()

        trade["price_change"] = round(
            exit_price - entry_price,
            8
        )

        logger.info(
            "Virtual trade settled: %s %s",
            pair,
            outcome
        )


def create_virtual_trade(
    pair,
    signal,
    candle_time,
    entry_price,
    timeframe
):

    if signal not in ("CALL", "PUT"):
        return None

    trade_id = (
        f"{pair}_{candle_time}_{timeframe}"
    )

    # Не создаём повторную сделку
    # по одной и той же свече.

    for trade in VIRTUAL_TRADES:

        if trade["id"] == trade_id:
            return trade

    entry_time = pd.Timestamp(candle_time)

    expiry_at = (
        entry_time + TIMEFRAMES[timeframe]
    )

    trade = {
        "id": trade_id,
        "pair": pair,
        "signal": signal,
        "timeframe": timeframe,
        "entry_price": float(entry_price),
        "entry_time": entry_time.isoformat(),
        "expiry_at": expiry_at.isoformat(),
        "status": "OPEN",
        "outcome": None,
        "exit_price": None,
        "exit_time": None,
        "price_change": None
    }

    VIRTUAL_TRADES.append(trade)

    return trade


# ==================================================
# SIGNAL HISTORY
# ==================================================

def save_signal_history(
    pair,
    result,
    candle_time,
    price
):

    entry = {
        "pair": pair,
        "signal": result["signal"],
        "reason": result["reason"],
        "price": float(price),
        "candle_time": candle_time,
        "score": result["score"],
        "created_at": pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    }

    SIGNAL_HISTORY.append(entry)

    return entry


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

        market_data[tf] = get_market_data(
            pair,
            tf
        )

    # Сначала проверяем результаты
    # ранее созданных виртуальных сделок.

    settle_virtual_trades(
        pair,
        market_data["1min"]
    )

    result = analyze_multi_timeframe(
        market_data
    )

    last_m1 = market_data["1min"].iloc[-1]

    candle_time = (
        market_data["1min"].index[-1].isoformat()
    )

    price = float(last_m1["close"])

    # Сохраняем каждый результат анализа.

    save_signal_history(
        pair,
        result,
        candle_time,
        price
    )

    # CALL / PUT создают виртуальную сделку.
    # WAIT не создаёт сделку.

    virtual_trade = create_virtual_trade(
        pair=pair,
        signal=result["signal"],
        candle_time=candle_time,
        entry_price=price,
        timeframe=requested_timeframe
    )

    response = {

        "signal": result["signal"],

        "reason": result["reason"],

        "pair": pair,

        "timeframe": "1min",

        "requested_timeframe": requested_timeframe,

        "price": price,

        "candle_time": candle_time,

        "candles_used": {
            tf: int(len(market_data[tf]))
            for tf in ANALYSIS_TIMEFRAMES
        },

        "strategy": "SMC Multi-Timeframe",

        "strategy_version": VERSION,

        "score": result["score"],

        "analysis": result["analysis"],

        "virtual_trade": virtual_trade
    }

    return make_json_safe(response)


# ==================================================
# API: SIGNAL HISTORY
# ==================================================

@app.get("/api/signals/history")
def get_signal_history(
    limit: int = Query(default=50, ge=1, le=500)
):

    records = list(SIGNAL_HISTORY)[-limit:]

    return make_json_safe({
        "count": len(records),
        "history": records
    })


# ==================================================
# API: VIRTUAL TRADES
# ==================================================

@app.get("/api/virtual-trades")
def get_virtual_trades(
    limit: int = Query(default=50, ge=1, le=500)
):

    trades = list(VIRTUAL_TRADES)

    closed = [
        trade for trade in trades
        if trade["status"] == "CLOSED"
    ]

    wins = sum(
        1 for trade in closed
        if trade["outcome"] == "WIN"
    )

    losses = sum(
        1 for trade in closed
        if trade["outcome"] == "LOSS"
    )

    draws = sum(
        1 for trade in closed
        if trade["outcome"] == "DRAW"
    )

    open_count = sum(
        1 for trade in trades
        if trade["status"] == "OPEN"
    )

    win_rate = (
        round(wins / len(closed) * 100, 2)
        if closed else None
    )

    return make_json_safe({

        "strategy_version": VERSION,

        "total_trades": len(trades),

        "open_trades": open_count,

        "closed_trades": len(closed),

        "wins": wins,

        "losses": losses,

        "draws": draws,

        "win_rate_percent": win_rate,

        "trades": trades[-limit:]
    })


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
        "version": VERSION,
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
