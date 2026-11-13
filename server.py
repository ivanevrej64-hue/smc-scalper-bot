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

VERSION = "2.3.0"

app = FastAPI(
    title="SMC Signal Bot",
    version=VERSION,
    description="Multi-timeframe SMC signal analysis and historical backtesting."
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
BACKTEST_CANDLE_LIMIT = 5000
SWING_LENGTH = 5
CANDLE_SAFETY_SECONDS = 3

MIN_SCORE = 70
MIN_M5_SCORE_EDGE = 10

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
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if value is pd.NaT:
        return None

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ==================================================
# REQUEST MODELS
# ==================================================

class SignalRequest(BaseModel):
    pair: str = Field(default="EUR/USD")
    timeframe: str = Field(default="1min")


class BacktestRequest(BaseModel):
    pair: str = Field(default="EUR/USD")
    timeframe: str = Field(default="1min")
    candles: int = Field(default=5000, ge=1000, le=5000)


# ==================================================
# MARKET DATA
# ==================================================

def get_market_data(
    pair: str,
    timeframe: str,
    outputsize: int = CANDLE_LIMIT
) -> pd.DataFrame:

    if not TWELVEDATA_KEY:
        raise HTTPException(
            status_code=503,
            detail="TWELVEDATA_KEY не задан в Render"
        )

    if timeframe not in TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail="Недопустимый таймфрейм"
        )

    outputsize = max(60, min(int(outputsize), BACKTEST_CANDLE_LIMIT))

    try:
        td = TDClient(apikey=TWELVEDATA_KEY)

        response = td.time_series(
            symbol=pair,
            interval=timeframe,
            outputsize=outputsize,
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

            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required)

        valid_ohlc = (
            (df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
        )
        df = df[valid_ohlc]

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
                    f"{pair} {timeframe}. Получено: {len(df)}"
                )
            )

        return df.tail(outputsize)

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
    if df is None or len(df) < period + 1:
        return 0.0

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

def find_confirmed_swings(df, length=SWING_LENGTH):
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    swing_highs = []
    swing_lows = []

    if len(df) < length * 2 + 3:
        return swing_highs, swing_lows

    # Последние length свечей не могут быть подтверждёнными pivot:
    # справа от pivot ещё должно пройти length закрытых свечей.
    for i in range(length, len(df) - length):
        high_window = highs[i - length:i + length + 1]
        low_window = lows[i - length:i + length + 1]

        if (
            highs[i] == np.max(high_window)
            and np.count_nonzero(high_window == highs[i]) == 1
        ):
            swing_highs.append((i, float(highs[i])))

        if (
            lows[i] == np.min(low_window)
            and np.count_nonzero(low_window == lows[i]) == 1
        ):
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


# ==================================================
# H1: MARKET DIRECTION
# ==================================================

def analyze_h1(df):
    close = df["close"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    price = float(close.iloc[-1])
    fast = float(ema20.iloc[-1])
    slow = float(ema50.iloc[-1])

    if price > fast > slow:
        signal = "CALL"
        reason = "Бычье направление: цена выше EMA20, EMA20 выше EMA50"
        score = 20
    elif price < fast < slow:
        signal = "PUT"
        reason = "Медвежье направление: цена ниже EMA20, EMA20 ниже EMA50"
        score = 20
    else:
        signal = "WAIT"
        reason = "EMA не подтверждают единое направление"
        score = 0

    return {
        "signal": signal,
        "reason": reason,
        "score": score,
        "ema20": fast,
        "ema50": slow
    }


# ==================================================
# M15: STRUCTURE / BOS / CHoCH
# ==================================================

def analyze_m15(df):
    swing_highs, swing_lows = find_confirmed_swings(df)

    # Структура должна быть актуальной, а не построенной
    # на уровнях из далёкой части графика.
    recent_highs = [
        item for item in swing_highs
        if len(df) - item[0] <= 80
    ]
    recent_lows = [
        item for item in swing_lows
        if len(df) - item[0] <= 80
    ]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return {
            "signal": "WAIT",
            "structure": "NONE",
            "market_structure": "UNKNOWN",
            "reason": "Недостаточно актуальных подтвержденных Swing High / Low",
            "score": 0
        }

    previous = df.iloc[-2]
    last = df.iloc[-1]

    high_index, high_level = recent_highs[-1]
    low_index, low_level = recent_lows[-1]

    high_values = [item[1] for item in recent_highs[-2:]]
    low_values = [item[1] for item in recent_lows[-2:]]

    bullish_structure = (
        high_values[-1] > high_values[-2]
        and low_values[-1] > low_values[-2]
    )

    bearish_structure = (
        high_values[-1] < high_values[-2]
        and low_values[-1] < low_values[-2]
    )

    if bullish_structure:
        market_structure = "BULLISH"
    elif bearish_structure:
        market_structure = "BEARISH"
    else:
        market_structure = "RANGE_OR_MIXED"

    atr = calculate_atr(df)

    if atr <= 0:
        return {
            "signal": "WAIT",
            "structure": "NONE",
            "market_structure": market_structure,
            "reason": "Не удалось рассчитать ATR для проверки пробоя",
            "score": 0
        }

    buffer = max(atr * 0.03, abs(float(last["close"])) * 0.00001)

    bullish_break = bool(
        float(previous["close"]) <= high_level + buffer
        and float(last["close"]) > high_level + buffer
    )

    bearish_break = bool(
        float(previous["close"]) >= low_level - buffer
        and float(last["close"]) < low_level - buffer
    )

    common = {
        "market_structure": market_structure,
        "swing_high": float(high_level),
        "swing_low": float(low_level),
        "swing_high_age": int(len(df) - high_index),
        "swing_low_age": int(len(df) - low_index),
        "atr": float(atr),
        "break_buffer": float(buffer)
    }

    if bullish_break and not bearish_break:
        structure = (
            "CHoCH" if market_structure == "BEARISH" else "BOS"
        )
        return {
            "signal": "CALL",
            "structure": structure,
            "reason": (
                f"Бычий {structure}: закрытие выше подтвержденного Swing High"
            ),
            "score": 25,
            "level": float(high_level),
            **common
        }

    if bearish_break and not bullish_break:
        structure = (
            "CHoCH" if market_structure == "BULLISH" else "BOS"
        )
        return {
            "signal": "PUT",
            "structure": structure,
            "reason": (
                f"Медвежий {structure}: закрытие ниже подтвержденного Swing Low"
            ),
            "score": 25,
            "level": float(low_level),
            **common
        }

    if bullish_break and bearish_break:
        reason = "Одновременно обнаружены противоположные условия пробоя"
    else:
        reason = "Нет нового закрытого пробоя подтвержденного Swing High / Low"

    return {
        "signal": "WAIT",
        "structure": "NONE",
        "market_structure": market_structure,
        "reason": reason,
        "score": 0,
        **common
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
            and float(last["low"]) < level - atr * 0.03
            and float(last["close"]) > level
            and float(last["close"]) > float(last["open"])
        )

    if swing_highs:
        index, level = swing_highs[-1]
        recent_enough = len(df) - index <= 30

        bearish = bool(
            recent_enough
            and float(last["high"]) > level + atr * 0.03
            and float(last["close"]) < level
            and float(last["close"]) < float(last["open"])
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

    start = max(2, len(df) - 10)

    for i in range(len(df) - 1, start - 1, -1):
        first = df.iloc[i - 2]
        middle = df.iloc[i - 1]
        third = df.iloc[i]

        body = abs(float(middle["close"] - middle["open"]))

        bullish_gap = float(third["low"] - first["high"])

        if (
            bullish_gap >= atr * 0.05
            and float(middle["close"]) > float(middle["open"])
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

        bearish_gap = float(first["low"] - third["high"])

        if (
            bearish_gap >= atr * 0.05
            and float(middle["close"]) < float(middle["open"])
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

    for i in range(
        len(df) - 4,
        max(5, len(df) - 15),
        -1
    ):
        candle = df.iloc[i]
        following = df.iloc[i + 1:min(i + 4, len(df))]
        previous = df.iloc[max(0, i - 5):i]

        if following.empty or previous.empty:
            continue

        previous_high = float(previous["high"].max())
        previous_low = float(previous["low"].min())

        max_following_body = float(
            (following["close"] - following["open"]).abs().max()
        )

        bullish_ob = (
            float(candle["close"]) < float(candle["open"])
            and float(following["close"].max()) > previous_high
            and max_following_body >= atr * 0.7
        )

        bearish_ob = (
            float(candle["close"]) > float(candle["open"])
            and float(following["close"].min()) < previous_low
            and max_following_body >= atr * 0.7
        )

        zone_low = float(candle["low"])
        zone_high = float(candle["high"])

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
            if bullish_ob and not bearish_ob:
                return "CALL"
            if bearish_ob and not bullish_ob:
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
        call_points += 15
    if ob_signal == "PUT":
        put_points += 15

    edge = abs(call_points - put_points)

    if call_points > put_points and edge >= MIN_M5_SCORE_EDGE:
        signal = "CALL"
        score = call_points
    elif put_points > call_points and edge >= MIN_M5_SCORE_EDGE:
        signal = "PUT"
        score = put_points
    else:
        signal = "WAIT"
        score = 0

    if call_points > 0 and put_points > 0:
        reason = (
            "Конфликт условий M5: "
            "недостаточное преимущество одной стороны"
            if signal == "WAIT"
            else "M5: одна сторона имеет достаточное преимущество"
        )
    else:
        reason = "Анализ Sweep, FVG и Order Block"

    return {
        "signal": signal,
        "reason": reason,
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
            "ob_put_points": 15 if ob_signal == "PUT" else 0,
            "call_points_total": int(call_points),
            "put_points_total": int(put_points),
            "directional_edge": int(edge),
            "minimum_edge_required": MIN_M5_SCORE_EDGE
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

    body = abs(float(last["close"] - last["open"]))
    candle_range = float(last["high"] - last["low"])

    if candle_range <= 0:
        return {
            "signal": "WAIT",
            "reason": "Свеча не имеет диапазона",
            "score": 0
        }

    body_ratio = body / candle_range
    normal_range = candle_range <= atr * 2.5
    strong_body = body >= atr * 0.2 and body_ratio >= 0.35

    bullish = bool(
        float(last["close"]) > float(last["open"])
        and float(last["close"]) > float(recent["high"].max()) + atr * 0.03
        and strong_body
        and normal_range
    )

    bearish = bool(
        float(last["close"]) < float(last["open"])
        and float(last["close"]) < float(recent["low"].min()) - atr * 0.03
        and strong_body
        and normal_range
    )

    if bullish:
        return {
            "signal": "CALL",
            "reason": (
                "Бычий пробой максимума четырех свечей "
                "с фильтром тела и размера свечи"
            ),
            "score": 10,
            "body_atr": round(body / atr, 3),
            "body_ratio": round(body_ratio, 3)
        }

    if bearish:
        return {
            "signal": "PUT",
            "reason": (
                "Медвежий пробой минимума четырех свечей "
                "с фильтром тела и размера свечи"
            ),
            "score": 10,
            "body_atr": round(body / atr, 3),
            "body_ratio": round(body_ratio, 3)
        }

    return {
        "signal": "WAIT",
        "reason": "Нет подтверждения пробоем на M1 или свеча слишком слабая",
        "score": 0,
        "body_atr": round(body / atr, 3),
        "body_ratio": round(body_ratio, 3)
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
    elif h1["signal"] == "PUT":
        put_score += 20

    if m15["signal"] == "CALL":
        call_score += 25
    elif m15["signal"] == "PUT":
        put_score += 25

    if m5["signal"] == "CALL":
        call_score += m5["score"]
    elif m5["signal"] == "PUT":
        put_score += m5["score"]

    if m1["signal"] == "CALL":
        call_score += 10
    elif m1["signal"] == "PUT":
        put_score += 10

    final_signal = "WAIT"
    reason = "Не выполнены обязательные условия входа"

    if all(signal == "CALL" for signal in directions) and call_score >= MIN_SCORE:
        final_signal = "CALL"
        reason = "Все таймфреймы подтвердили бычий сетап"

    elif all(signal == "PUT" for signal in directions) and put_score >= MIN_SCORE:
        final_signal = "PUT"
        reason = "Все таймфреймы подтвердили медвежий сетап"

    elif h1["signal"] == "WAIT":
        reason = "Ожидание: H1 не определил направление"

    elif m15["signal"] == "WAIT":
        reason = "Ожидание: M15 не подтвердил BOS или CHoCH"

    elif m5["signal"] == "WAIT":
        reason = "Ожидание: M5 не подтвердил торговую модель"

    elif m1["signal"] == "WAIT":
        reason = "Ожидание: M1 не подтвердил точку входа"

    elif len(set(directions)) > 1:
        reason = "Ожидание: направления таймфреймов противоречат друг другу"

    elif max(call_score, put_score) < MIN_SCORE:
        reason = "Ожидание: недостаточно баллов"

    return {
        "signal": final_signal,
        "reason": reason,
        "score": {
            "CALL": int(call_score),
            "PUT": int(put_score),
            "minimum_required": MIN_SCORE
        },
        "alignment": {
            "H1": h1["signal"],
            "M15": m15["signal"],
            "M5": m5["signal"],
            "M1": m1["signal"],
            "all_timeframes_aligned": len(set(directions)) == 1
        },
        "analysis": components
    }


# ==================================================
# VIRTUAL TRADE MANAGEMENT
# ==================================================

def settle_virtual_trades(pair, m1_df):
    for trade in VIRTUAL_TRADES:
        if trade["pair"] != pair or trade["status"] != "OPEN":
            continue

        expiry_at = pd.Timestamp(trade["expiry_at"])

        eligible = m1_df[m1_df.index >= expiry_at]
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
        trade["price_change"] = round(exit_price - entry_price, 8)

        logger.info("Virtual trade settled: %s %s", pair, outcome)


def create_virtual_trade(
    pair,
    signal,
    candle_time,
    entry_price,
    timeframe
):
    if signal not in ("CALL", "PUT"):
        return None

    trade_id = f"{pair}_{candle_time}_{timeframe}"

    for trade in VIRTUAL_TRADES:
        if trade["id"] == trade_id:
            return trade

    # Сигнал формируется после закрытия M1 свечи.
    # Поэтому фактическое время входа — время открытия свечи + 1 минута.
    signal_candle_open = pd.Timestamp(candle_time)
    entry_time = signal_candle_open + TIMEFRAMES["1min"]
    expiry_at = entry_time + TIMEFRAMES[timeframe]

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

def save_signal_history(pair, result, candle_time, price, requested_timeframe):
    entry = {
        "pair": pair,
        "signal": result["signal"],
        "reason": result["reason"],
        "price": float(price),
        "candle_time": candle_time,
        "requested_timeframe": requested_timeframe,
        "score": result["score"],
        "strategy_version": VERSION,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat()
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

    market_data = {
        tf: get_market_data(pair, tf)
        for tf in ANALYSIS_TIMEFRAMES
    }

    settle_virtual_trades(pair, market_data["1min"])

    result = analyze_multi_timeframe(market_data)

    last_m1 = market_data["1min"].iloc[-1]
    candle_time = market_data["1min"].index[-1].isoformat()
    price = float(last_m1["close"])

    save_signal_history(
        pair,
        result,
        candle_time,
        price,
        requested_timeframe
    )

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
        "alignment": result["alignment"],
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

    wins = sum(1 for trade in closed if trade["outcome"] == "WIN")
    losses = sum(1 for trade in closed if trade["outcome"] == "LOSS")
    draws = sum(1 for trade in closed if trade["outcome"] == "DRAW")
    open_count = sum(1 for trade in trades if trade["status"] == "OPEN")

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
# HISTORICAL BACKTEST HELPERS
# ==================================================

def _closed_candle_slice(df, timeframe, evaluation_time, max_rows=150):
    """
    Возвращает только свечи, которые уже закрылись к evaluation_time.
    Индекс Twelve Data трактуется как время открытия свечи.
    """
    duration = TIMEFRAMES[timeframe]
    latest_allowed_open = evaluation_time - duration

    end_position = int(
        np.searchsorted(
            df.index.asi8,
            latest_allowed_open.value,
            side="right"
        )
    )

    if end_position <= 0:
        return df.iloc[0:0]

    return df.iloc[max(0, end_position - max_rows):end_position]


def run_historical_backtest(pair, expiry_timeframe, candle_count):
    history = {
        tf: get_market_data(pair, tf, outputsize=candle_count)
        for tf in ANALYSIS_TIMEFRAMES
    }

    m1 = history["1min"]

    if len(m1) < 1000:
        raise HTTPException(
            status_code=502,
            detail=f"Для backtest получено мало M1 свечей: {len(m1)}"
        )

    expiry_duration = TIMEFRAMES[expiry_timeframe]
    evaluated = 0
    skipped = 0
    signals = 0
    wins = 0
    losses = 0
    draws = 0
    trades = []

    # Для каждого закрытия M1 свечи строим срезы только из тех
    # старших свечей, которые к этому моменту уже закрылись.
    for i in range(len(m1)):
        m1_open_time = m1.index[i]
        evaluation_time = m1_open_time + TIMEFRAMES["1min"]

        window = {
            tf: _closed_candle_slice(
                history[tf],
                tf,
                evaluation_time,
                max_rows=CANDLE_LIMIT
            )
            for tf in ANALYSIS_TIMEFRAMES
        }

        if any(len(window[tf]) < 60 for tf in ANALYSIS_TIMEFRAMES):
            skipped += 1
            continue

        evaluated += 1
        result = analyze_multi_timeframe(window)

        if result["signal"] not in ("CALL", "PUT"):
            continue

        signals += 1

        entry_price = float(m1.iloc[i]["close"])
        # Вход моделируется после закрытия сигнальной M1 свечи.
        entry_time = m1_open_time + TIMEFRAMES["1min"]
        expiry_at = entry_time + expiry_duration

        future = m1[m1.index >= expiry_at]

        if future.empty:
            continue

        exit_time = future.index[0]
        exit_price = float(future.iloc[0]["close"])

        if result["signal"] == "CALL":
            if exit_price > entry_price:
                outcome = "WIN"
                wins += 1
            elif exit_price < entry_price:
                outcome = "LOSS"
                losses += 1
            else:
                outcome = "DRAW"
                draws += 1
        else:
            if exit_price < entry_price:
                outcome = "WIN"
                wins += 1
            elif exit_price > entry_price:
                outcome = "LOSS"
                losses += 1
            else:
                outcome = "DRAW"
                draws += 1

        if len(trades) < 200:
            trades.append({
                "pair": pair,
                "signal": result["signal"],
                "entry_time": entry_time.isoformat(),
                "entry_price": entry_price,
                "expiry_at": expiry_at.isoformat(),
                "exit_time": exit_time.isoformat(),
                "exit_price": exit_price,
                "outcome": outcome,
                "score": result["score"]
            })

    closed_trades = wins + losses + draws
    win_rate = (
        round(wins / closed_trades * 100, 2)
        if closed_trades else None
    )

    return {
        "strategy_version": VERSION,
        "pair": pair,
        "expiry_timeframe": expiry_timeframe,
        "requested_m1_candles": candle_count,
        "available_candles": {
            tf: int(len(history[tf]))
            for tf in ANALYSIS_TIMEFRAMES
        },
        "evaluated_points": evaluated,
        "skipped_points_insufficient_history": skipped,
        "signals_generated": signals,
        "closed_trades": closed_trades,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate_percent": win_rate,
        "trades_returned": len(trades),
        "trades": trades,
        "limitations": [
            "Backtest использует доступную историю Twelve Data и текущие правила стратегии.",
            "Комиссии, спред, проскальзывание и выплаты брокера не учитываются.",
            "Результат не гарантирует будущую доходность.",
            "Если сделок больше 200, в ответе показаны только первые 200; итоговая статистика учитывает все закрытые сделки."
        ]
    }


# ==================================================
# API: HISTORICAL BACKTEST
# ==================================================

@app.post("/api/backtest")
def historical_backtest(req: BacktestRequest):
    pair = req.pair.strip().upper()
    expiry_timeframe = req.timeframe.strip()

    if pair not in ALLOWED_PAIRS:
        raise HTTPException(
            status_code=400,
            detail="Эта валютная пара не поддерживается"
        )

    if expiry_timeframe not in ("1min", "5min"):
        raise HTTPException(
            status_code=400,
            detail="Для backtest разрешена экспирация 1min или 5min"
        )

    result = run_historical_backtest(
        pair=pair,
        expiry_timeframe=expiry_timeframe,
        candle_count=req.candles
    )

    return make_json_safe(result)


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
