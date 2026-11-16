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

VERSION = "2.4.6"

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
M15_MEMORY_BARS = 8  # Срок жизни подтвержденного BOS/CHoCH после свечи пробоя
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

    # Условия стратегии НЕ изменены.
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

    if signal != "WAIT":
        block_reason = None
    elif fast > slow:
        block_reason = "EMA20 выше EMA50, но закрытие H1 не выше EMA20"
    elif fast < slow:
        block_reason = "EMA20 ниже EMA50, но закрытие H1 не ниже EMA20"
    else:
        block_reason = "EMA20 и EMA50 совпадают"

    return {
        "signal": signal,
        "reason": reason,
        "score": score,
        "ema20": fast,
        "ema50": slow,
        "h1_close": price,
        "h1_candle_time": str(df.index[-1]),
        "close_minus_ema20": float(price - fast),
        "ema20_minus_ema50": float(fast - slow),
        "close_above_ema20": bool(price > fast),
        "close_below_ema20": bool(price < fast),
        "ema20_above_ema50": bool(fast > slow),
        "ema20_below_ema50": bool(fast < slow),
        "call_condition_pass": bool(price > fast > slow),
        "put_condition_pass": bool(price < fast < slow),
        "block_reason": block_reason
    }

# ==================================================
# M15: STRUCTURE / BOS / CHoCH
# ==================================================

def _find_recent_m15_break(df, swing_highs, swing_lows):
    """Находит самое новое пересечение подтвержденного swing за 0..8 M15 свечей.

    Для каждой исторической свечи учитываются только pivots, которые были
    подтверждены (SWING_LENGTH правых закрытых свечей) к моменту события.
    Это предотвращает подглядывание в будущее при backtest и live-анализе.
    События не сохраняются в памяти процесса: они восстанавливаются из OHLC.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    previous_close = np.r_[np.nan, close[:-1]]
    true_range = np.nanmax(np.vstack([
        high - low,
        np.abs(high - previous_close),
        np.abs(low - previous_close)
    ]), axis=0)
    atr_values = pd.Series(true_range).rolling(window=14, min_periods=14).mean().to_numpy()

    newest_idx = len(df) - 1
    oldest_idx = max(1, newest_idx - M15_MEMORY_BARS)

    for i in range(newest_idx, oldest_idx - 1, -1):
        # Максимум SWING_LENGTH свечей справа — к моменту i pivot уже известен.
        cutoff = i - SWING_LENGTH
        available_highs = [
            item for item in swing_highs
            if item[0] <= cutoff and i + 1 - item[0] <= 80
        ]
        available_lows = [
            item for item in swing_lows
            if item[0] <= cutoff and i + 1 - item[0] <= 80
        ]
        if len(available_highs) < 2 or len(available_lows) < 2:
            continue

        event_atr = float(atr_values[i])
        if not math.isfinite(event_atr) or event_atr <= 0:
            continue

        high_level = float(available_highs[-1][1])
        low_level = float(available_lows[-1][1])
        buffer = max(event_atr * 0.03, abs(float(close[i])) * 0.00001)
        high_threshold = high_level + buffer
        low_threshold = low_level - buffer
        bull_cross = bool(close[i - 1] <= high_threshold and close[i] > high_threshold)
        bear_cross = bool(close[i - 1] >= low_threshold and close[i] < low_threshold)
        if not bull_cross and not bear_cross:
            continue

        highs = [item[1] for item in available_highs[-2:]]
        lows = [item[1] for item in available_lows[-2:]]
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            structure_at_event = "BULLISH"
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            structure_at_event = "BEARISH"
        else:
            structure_at_event = "RANGE_OR_MIXED"

        if bull_cross and bear_cross:
            return {
                "signal": "WAIT",
                "structure": "NONE",
                "status": "CONFLICTING_CROSSES",
                "reason_code": "CONFLICTING_BREAKS",
                "event_index": i,
                "event_candle_time": str(df.index[i]),
                "age_bars": int(newest_idx - i),
                "market_structure_at_event": structure_at_event,
                "event_atr": event_atr,
                "event_buffer": float(buffer),
                "event_close": float(close[i]),
                "level": None,
                "threshold": None,
                "invalidation_time": None,
            }

        direction = "CALL" if bull_cross else "PUT"
        threshold = float(high_threshold if bull_cross else low_threshold)
        broken_level = float(high_level if bull_cross else low_level)
        # Старый сигнал нельзя активировать повторно после закрытия за уровнем
        # в противоположную сторону. Новый пробой станет новым событием.
        future_closes = close[i + 1:]
        if direction == "CALL":
            invalidations = np.flatnonzero(future_closes <= threshold)
        else:
            invalidations = np.flatnonzero(future_closes >= threshold)

        invalidated = len(invalidations) > 0
        invalidation_index = i + 1 + int(invalidations[0]) if invalidated else None
        event_type = "CHoCH" if (
            (direction == "CALL" and structure_at_event == "BEARISH") or
            (direction == "PUT" and structure_at_event == "BULLISH")
        ) else "BOS"

        return {
            "signal": "WAIT" if invalidated else direction,
            "structure": event_type,
            "status": "INVALIDATED" if invalidated else ("FRESH" if i == newest_idx else "REMEMBERED"),
            "reason_code": "BREAK_INVALIDATED_BY_CLOSE" if invalidated else ("FRESH_CROSS" if i == newest_idx else "ACTIVE_RECENT_CROSS"),
            "event_index": i,
            "event_candle_time": str(df.index[i]),
            "age_bars": int(newest_idx - i),
            "market_structure_at_event": structure_at_event,
            "event_atr": event_atr,
            "event_buffer": float(buffer),
            "event_close": float(close[i]),
            "level": broken_level,
            "threshold": threshold,
            "invalidation_time": str(df.index[invalidation_index]) if invalidated else None,
        }

    return None


def analyze_m15(df):
    swing_highs, swing_lows = find_confirmed_swings(df)

    recent_highs = [item for item in swing_highs if len(df) - item[0] <= 80]
    recent_lows = [item for item in swing_lows if len(df) - item[0] <= 80]

    memory_base = {
        "memory_window_bars": M15_MEMORY_BARS,
        "memory_status": "NO_RECENT_EVENT",
        "event_age_bars": None,
        "event_candle_time": None,
        "event_level": None,
        "event_break_threshold": None,
        "market_structure_at_event": None,
        "event_invalidation_time": None,
    }

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return {
            "signal": "WAIT",
            "structure": "NONE",
            "market_structure": "UNKNOWN",
            "reason": "Недостаточно актуальных подтвержденных Swing High / Low",
            "diagnostic_reason_code": "NOT_ENOUGH_CONFIRMED_SWINGS",
            "score": 0,
            "confirmed_swing_highs": int(len(recent_highs)),
            "confirmed_swing_lows": int(len(recent_lows)),
            "bullish_break_detected": False,
            "bearish_break_detected": False,
            **memory_base
        }

    previous = df.iloc[-2]
    last = df.iloc[-1]
    high_index, high_level = recent_highs[-1]
    low_index, low_level = recent_lows[-1]

    high_values = [item[1] for item in recent_highs[-2:]]
    low_values = [item[1] for item in recent_lows[-2:]]
    bullish_structure = high_values[-1] > high_values[-2] and low_values[-1] > low_values[-2]
    bearish_structure = high_values[-1] < high_values[-2] and low_values[-1] < low_values[-2]
    market_structure = "BULLISH" if bullish_structure else ("BEARISH" if bearish_structure else "RANGE_OR_MIXED")

    atr = calculate_atr(df)
    if atr <= 0:
        return {
            "signal": "WAIT",
            "structure": "NONE",
            "market_structure": market_structure,
            "reason": "Не удалось рассчитать ATR для проверки пробоя",
            "diagnostic_reason_code": "ATR_NOT_AVAILABLE",
            "score": 0,
            "confirmed_swing_highs": int(len(recent_highs)),
            "confirmed_swing_lows": int(len(recent_lows)),
            "bullish_break_detected": False,
            "bearish_break_detected": False,
            **memory_base
        }

    buffer = max(atr * 0.03, abs(float(last["close"])) * 0.00001)
    bullish_break = bool(float(previous["close"]) <= high_level + buffer and float(last["close"]) > high_level + buffer)
    bearish_break = bool(float(previous["close"]) >= low_level - buffer and float(last["close"]) < low_level - buffer)

    common = {
        "market_structure": market_structure,
        "swing_high": float(high_level),
        "swing_low": float(low_level),
        "swing_high_age": int(len(df) - high_index),
        "swing_low_age": int(len(df) - low_index),
        "atr": float(atr),
        "break_buffer": float(buffer),
        "confirmed_swing_highs": int(len(recent_highs)),
        "confirmed_swing_lows": int(len(recent_lows)),
        "bullish_break_detected": bool(bullish_break),
        "bearish_break_detected": bool(bearish_break),
        "previous_close": float(previous["close"]),
        "last_close": float(last["close"]),
        "last_candle_time": str(df.index[-1]),
        "swing_high_time": str(df.index[high_index]),
        "swing_low_time": str(df.index[low_index]),
        "bullish_break_threshold": float(high_level + buffer),
        "bearish_break_threshold": float(low_level - buffer),
        "previous_at_or_below_bull_threshold": bool(float(previous["close"]) <= high_level + buffer),
        "last_above_bull_threshold": bool(float(last["close"]) > high_level + buffer),
        "previous_at_or_above_bear_threshold": bool(float(previous["close"]) >= low_level - buffer),
        "last_below_bear_threshold": bool(float(last["close"]) < low_level - buffer),
        "close_minus_bull_threshold": float(float(last["close"]) - (high_level + buffer)),
        "close_minus_bear_threshold": float(float(last["close"]) - (low_level - buffer)),
    }

    event = _find_recent_m15_break(df, swing_highs, swing_lows)
    if event is None:
        return {
            "signal": "WAIT",
            "structure": "NONE",
            "reason": "Нет актуального BOS/CHoCH за последние 8 закрытых M15 свечей",
            "diagnostic_reason_code": "NO_RECENT_CONFIRMED_CROSS",
            "score": 0,
            **common,
            **memory_base,
        }

    memory = {
        "memory_window_bars": M15_MEMORY_BARS,
        "memory_status": event["status"],
        "event_age_bars": event["age_bars"],
        "event_candle_time": event["event_candle_time"],
        "event_level": event["level"],
        "event_break_threshold": event["threshold"],
        "market_structure_at_event": event["market_structure_at_event"],
        "event_invalidation_time": event["invalidation_time"],
        "event_atr": event["event_atr"],
        "event_buffer": event["event_buffer"],
        "event_close": event["event_close"],
        "memory_bars_remaining": max(0, M15_MEMORY_BARS - event["age_bars"]),
    }

    if event["signal"] == "WAIT":
        reason = (
            "Последний подтвержденный BOS/CHoCH отменен закрытием за пробитым уровнем"
            if event["status"] == "INVALIDATED"
            else "Противоречивые пробои на одной M15 свече"
        )
        return {
            "signal": "WAIT",
            "structure": "NONE",
            "reason": reason,
            "diagnostic_reason_code": event["reason_code"],
            "score": 0,
            **common,
            **memory,
        }

    direction = event["signal"]
    structure = event["structure"]
    label = "Бычий" if direction == "CALL" else "Медвежий"
    reason = f"{label} {structure}: подтвержденный пробой M15"
    if event["age_bars"] > 0:
        reason += f" сохраняется {event['age_bars']} закрытых свечей"

    return {
        "signal": direction,
        "structure": structure,
        "reason": reason,
        "diagnostic_reason_code": event["reason_code"],
        "score": 25,
        "level": float(event["level"]),
        **common,
        **memory,
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
# M5: DIAGNOSTICS ONLY (do not participate in scoring)
# ==================================================

def _diagnose_m5_sweep(df, atr):
    last = df.iloc[-1]
    highs, lows = find_confirmed_swings(df)
    valid_history = bool(atr > 0 and len(df) >= 30)

    def explain(side, swings):
        if not valid_history:
            return {"eligible": False, "reason": "INSUFFICIENT_HISTORY_OR_ATR"}
        if not swings:
            return {"eligible": False, "reason": "NO_CONFIRMED_SWING"}
        index, level = swings[-1]
        age = len(df) - index  # same age convention as signal detector
        recent = age <= 30
        if side == "bullish":
            wick_threshold = float(level - atr * 0.03)
            wick_pass = bool(float(last["low"]) < wick_threshold)
            recovered = bool(float(last["close"]) > level)
            candle_direction_pass = bool(float(last["close"]) > float(last["open"]))
            wick_price = float(last["low"])
        else:
            wick_threshold = float(level + atr * 0.03)
            wick_pass = bool(float(last["high"]) > wick_threshold)
            recovered = bool(float(last["close"]) < level)
            candle_direction_pass = bool(float(last["close"]) < float(last["open"]))
            wick_price = float(last["high"])
        missing = []
        if not recent:
            missing.append("SWING_OLDER_THAN_30_BARS")
        if not wick_pass:
            missing.append("NO_WICK_BEYOND_SWING_BUFFER")
        if not recovered:
            missing.append("CLOSE_DID_NOT_RECLAIM_SWING")
        if not candle_direction_pass:
            missing.append("CANDLE_DIRECTION_NOT_CONFIRMED")
        return {
            "eligible": True,
            "swing_level": float(level),
            "swing_time": str(df.index[index]),
            "swing_age_bars": int(age),
            "swing_recent": bool(recent),
            "wick_price": wick_price,
            "required_wick_threshold": wick_threshold,
            "wick_condition_pass": wick_pass,
            "close_reclaims_level": recovered,
            "candle_direction_pass": candle_direction_pass,
            "confirmed": bool(recent and wick_pass and recovered and candle_direction_pass),
            "failed_checks": missing
        }

    return {
        "bullish": explain("bullish", lows),
        "bearish": explain("bearish", highs),
        "required_wick_buffer": float(atr * 0.03) if atr > 0 else None,
    }


def _diagnose_m5_fvg(df, atr):
    result = {
        "lookback_candles": 10,
        "minimum_gap_size": float(atr * 0.05) if atr > 0 else None,
        "minimum_middle_body": float(atr * 0.5) if atr > 0 else None,
        "bullish": {"valid_patterns": 0, "filled_patterns": 0, "outside_zone": 0, "latest_pattern": None},
        "bearish": {"valid_patterns": 0, "filled_patterns": 0, "outside_zone": 0, "latest_pattern": None},
        "detected_signal": "WAIT",
    }
    if len(df) < 20 or atr <= 0:
        result["reason"] = "INSUFFICIENT_HISTORY_OR_ATR"
        return result

    start = max(2, len(df) - 10)
    price = float(df["close"].iloc[-1])
    # Same candle triples, order and thresholds as detect_fvg().
    for i in range(len(df) - 1, start - 1, -1):
        first, middle, third = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        body = abs(float(middle["close"] - middle["open"]))
        for direction in ("bullish", "bearish"):
            if direction == "bullish":
                gap = float(third["low"] - first["high"])
                middle_direction = bool(float(middle["close"]) > float(middle["open"]))
                lower = float(first["high"])
                upper = float(third["low"])
                zone_min, zone_max = lower - atr * 0.05, upper + atr * 0.25
            else:
                gap = float(first["low"] - third["high"])
                middle_direction = bool(float(middle["close"]) < float(middle["open"]))
                lower = float(third["high"])
                upper = float(first["low"])
                zone_min, zone_max = lower - atr * 0.25, upper + atr * 0.05
            if not (gap >= atr * 0.05 and middle_direction and body >= atr * 0.5):
                continue
            later = df.iloc[i + 1:]
            if direction == "bullish":
                filled = bool(not later.empty and float(later["low"].min()) <= lower)
            else:
                filled = bool(not later.empty and float(later["high"].max()) >= upper)
            near = bool(zone_min <= price <= zone_max)
            bucket = result[direction]
            bucket["valid_patterns"] += 1
            if filled:
                bucket["filled_patterns"] += 1
            elif not near:
                bucket["outside_zone"] += 1
            pattern = {
                "formation_time": str(df.index[i]),
                "age_bars": int(len(df) - 1 - i),
                "gap_size": float(gap),
                "zone_low": float(lower),
                "zone_high": float(upper),
                "allowed_price_min": float(zone_min),
                "allowed_price_max": float(zone_max),
                "current_price": price,
                "fully_filled": filled,
                "price_in_entry_zone": near,
                "active": bool(not filled and near),
                "distance_to_entry_zone": float(max(zone_min - price, price - zone_max, 0.0)),
            }
            if bucket["latest_pattern"] is None:
                bucket["latest_pattern"] = pattern
            if not filled and near and result["detected_signal"] == "WAIT":
                result["detected_signal"] = "CALL" if direction == "bullish" else "PUT"
                result["selected_pattern"] = {"direction": direction, **pattern}
    if result["detected_signal"] == "WAIT":
        if not result["bullish"]["valid_patterns"] and not result["bearish"]["valid_patterns"]:
            result["reason"] = "NO_VALID_GAP_WITH_MOMENTUM_IN_LAST_10_BARS"
        else:
            result["reason"] = "GAPS_FILLED_OR_PRICE_OUTSIDE_ENTRY_ZONES"
    else:
        result["reason"] = "ACTIVE_FVG_FOUND"
    return result


def _diagnose_m5_order_block(df, atr):
    result = {
        "lookback_candles": 15,
        "minimum_following_body": float(atr * 0.7) if atr > 0 else None,
        "maximum_zone_distance": float(atr * 0.15) if atr > 0 else None,
        "bullish": {"valid_patterns": 0, "zone_revisits": 0, "latest_pattern": None},
        "bearish": {"valid_patterns": 0, "zone_revisits": 0, "latest_pattern": None},
        "detected_signal": "WAIT",
    }
    if len(df) < 30 or atr <= 0:
        result["reason"] = "INSUFFICIENT_HISTORY_OR_ATR"
        return result
    last = df.iloc[-1]
    close_price = float(last["close"])
    # Match the candidate window and order in detect_order_block().
    for i in range(len(df) - 4, max(5, len(df) - 15), -1):
        candle = df.iloc[i]
        following = df.iloc[i + 1:min(i + 4, len(df))]
        previous = df.iloc[max(0, i - 5):i]
        if following.empty or previous.empty:
            continue
        previous_high = float(previous["high"].max())
        previous_low = float(previous["low"].min())
        max_body = float((following["close"] - following["open"]).abs().max())
        bullish_ob = bool(
            float(candle["close"]) < float(candle["open"])
            and float(following["close"].max()) > previous_high
            and max_body >= atr * 0.7
        )
        bearish_ob = bool(
            float(candle["close"]) > float(candle["open"])
            and float(following["close"].min()) < previous_low
            and max_body >= atr * 0.7
        )
        zone_low, zone_high = float(candle["low"]), float(candle["high"])
        overlaps = bool(float(last["high"]) >= zone_low and float(last["low"]) <= zone_high)
        distance = min(abs(close_price - zone_low), abs(close_price - zone_high))
        near = bool(distance <= atr * 0.15)
        for direction, formed in (("bullish", bullish_ob), ("bearish", bearish_ob)):
            if not formed:
                continue
            bucket = result[direction]
            bucket["valid_patterns"] += 1
            if overlaps or near:
                bucket["zone_revisits"] += 1
            pattern = {
                "formation_time": str(df.index[i]),
                "age_bars": int(len(df) - 1 - i),
                "zone_low": zone_low,
                "zone_high": zone_high,
                "previous_structure_high": previous_high,
                "previous_structure_low": previous_low,
                "max_following_body": max_body,
                "last_candle_overlaps_zone": overlaps,
                "close_near_zone": near,
                "distance_to_zone_boundary": float(distance),
                "entry_condition_pass": bool(overlaps or near),
            }
            if bucket["latest_pattern"] is None:
                bucket["latest_pattern"] = pattern
            if (overlaps or near) and result["detected_signal"] == "WAIT":
                result["detected_signal"] = "CALL" if direction == "bullish" else "PUT"
                result["selected_pattern"] = {"direction": direction, **pattern}
    if result["detected_signal"] != "WAIT":
        result["reason"] = "VALID_OB_AND_ZONE_RETEST"
    elif not result["bullish"]["valid_patterns"] and not result["bearish"]["valid_patterns"]:
        result["reason"] = "NO_ORDER_BLOCK_PATTERN_IN_CANDIDATE_WINDOW"
    else:
        result["reason"] = "PRICE_NOT_IN_ORDER_BLOCK_ZONE"
    return result


def diagnose_m5(df, sweep_call, sweep_put, fvg_signal, ob_signal):
    """Information only. Never changes CALL, PUT, WAIT or scores."""
    atr = calculate_atr(df)
    if df is None or df.empty:
        return {"reason": "NO_DATA"}
    sweep = _diagnose_m5_sweep(df, atr)
    fvg = _diagnose_m5_fvg(df, atr)
    ob = _diagnose_m5_order_block(df, atr)
    return {
        "candle_time": str(df.index[-1]),
        "m5_close": float(df["close"].iloc[-1]),
        "atr": float(atr),
        "liquidity_sweep": sweep,
        "fvg": fvg,
        "order_block": ob,
        "consistency_checks": {
            "sweep_bullish_matches": bool(sweep["bullish"].get("confirmed", False) == sweep_call),
            "sweep_bearish_matches": bool(sweep["bearish"].get("confirmed", False) == sweep_put),
            "fvg_matches": bool(fvg["detected_signal"] == fvg_signal),
            "order_block_matches": bool(ob["detected_signal"] == ob_signal),
        }
    }


# ==================================================
# M5: COMBINED SETUP
# ==================================================

def analyze_m5(df, include_diagnostics=True):
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
        },
        **({
            "diagnostics": diagnose_m5(
                df, sweep_call, sweep_put, fvg_signal, ob_signal
            )
        } if include_diagnostics else {})
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
            "score": 0,
            "breakout_call": False,
            "breakout_put": False,
            "body_filter_pass": False,
            "range_filter_pass": False,
            "failed_checks": ["NO_RECENT_CANDLES_OR_INVALID_ATR"]
        }

    body = abs(float(last["close"] - last["open"]))
    candle_range = float(last["high"] - last["low"])

    if candle_range <= 0:
        return {
            "signal": "WAIT",
            "reason": "Свеча не имеет диапазона",
            "score": 0,
            "breakout_call": False,
            "breakout_put": False,
            "body_filter_pass": False,
            "range_filter_pass": False,
            "failed_checks": ["ZERO_CANDLE_RANGE"]
        }

    body_ratio = body / candle_range
    normal_range = candle_range <= atr * 2.5
    strong_body = body >= atr * 0.2 and body_ratio >= 0.35

    # Исходные формулы пробоя сохранены без изменений.
    bullish_breakout = bool(
        float(last["close"]) > float(last["open"])
        and float(last["close"]) > float(recent["high"].max()) + atr * 0.03
    )
    bearish_breakout = bool(
        float(last["close"]) < float(last["open"])
        and float(last["close"]) < float(recent["low"].min()) - atr * 0.03
    )

    bullish = bool(bullish_breakout and strong_body and normal_range)
    bearish = bool(bearish_breakout and strong_body and normal_range)

    close_price = float(last["close"])
    open_price = float(last["open"])
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())
    call_threshold = float(recent_high + atr * 0.03)
    put_threshold = float(recent_low - atr * 0.03)

    failed_checks = []
    if not bullish_breakout and not bearish_breakout:
        failed_checks.append("NO_FOUR_CANDLE_BREAKOUT")
    if body < atr * 0.2:
        failed_checks.append("BODY_BELOW_MINIMUM_ATR")
    if body_ratio < 0.35:
        failed_checks.append("BODY_RATIO_BELOW_0_35")
    if not normal_range:
        failed_checks.append("CANDLE_RANGE_TOO_LARGE")

    diagnostics = {
        "breakout_call": bullish_breakout,
        "breakout_put": bearish_breakout,
        "body_filter_pass": bool(strong_body),
        "range_filter_pass": bool(normal_range),
        "candle_range_atr": round(candle_range / atr, 3),
        "body_atr": round(body / atr, 3),
        "body_ratio": round(body_ratio, 3),
        "m1_candle_time": str(df.index[-1]),
        "open": open_price,
        "close": close_price,
        "atr": float(atr),
        "previous_4_high": recent_high,
        "previous_4_low": recent_low,
        "call_breakout_threshold": call_threshold,
        "put_breakout_threshold": put_threshold,
        "close_minus_call_threshold": float(close_price - call_threshold),
        "close_minus_put_threshold": float(close_price - put_threshold),
        "body_size": float(body),
        "minimum_body_size": float(atr * 0.2),
        "minimum_body_ratio": 0.35,
        "max_normal_candle_range": float(atr * 2.5),
        "failed_checks": failed_checks
    }

    if bullish:
        return {
            "signal": "CALL",
            "reason": (
                "Бычий пробой максимума четырех свечей "
                "с фильтром тела и размера свечи"
            ),
            "score": 10,
            **diagnostics
        }

    if bearish:
        return {
            "signal": "PUT",
            "reason": (
                "Медвежий пробой минимума четырех свечей "
                "с фильтром тела и размера свечи"
            ),
            "score": 10,
            **diagnostics
        }

    return {
        "signal": "WAIT",
        "reason": "Нет подтверждения пробоем на M1 или свеча слишком слабая",
        "score": 0,
        **diagnostics
    }

# ==================================================
# MULTI-TIMEFRAME ENGINE
# ==================================================

def analyze_multi_timeframe(data, include_m5_diagnostics=True):
    h1 = analyze_h1(data["1h"])
    m15 = analyze_m15(data["15min"])
    m5 = analyze_m5(data["5min"], include_diagnostics=include_m5_diagnostics)
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
        "analysis": components,
        "decision_diagnostics": {
            "CALL": {
                "blocking_timeframes": [
                    tf for tf, component in components.items()
                    if component["signal"] != "CALL"
                ],
                "score_shortfall": max(0, MIN_SCORE - int(call_score))
            },
            "PUT": {
                "blocking_timeframes": [
                    tf for tf, component in components.items()
                    if component["signal"] != "PUT"
                ],
                "score_shortfall": max(0, MIN_SCORE - int(put_score))
            },
            "all_timeframes_required": True,
            "minimum_score_required": MIN_SCORE
        }
    }


# ==================================================
# M1 EXPIRY ALIGNMENT
# ==================================================

def _m1_candle_at_expiry(m1_df, expiry_at):
    """Return the *closed* M1 candle ending exactly at expiry_at.

    Twelve Data indexes candles by opening time. Never take the first candle
    opening *after* expiry: that would shift virtual/backtest outcomes by a minute.
    Return None when the precise candle is missing (weekend/data gap/history).
    The caller is responsible for ensuring input M1 candles are already closed.
    """
    expiry_at = pd.Timestamp(expiry_at)
    expected_open = expiry_at - TIMEFRAMES["1min"]
    if expected_open not in m1_df.index:
        return None
    row = m1_df.loc[expected_open]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return {
        "exit_time": expiry_at.isoformat(),
        "exit_candle_open_time": expected_open.isoformat(),
        "exit_price": float(row["close"]),
    }


# ==================================================
# VIRTUAL TRADE MANAGEMENT
# ==================================================

def settle_virtual_trades(pair, m1_df):
    for trade in VIRTUAL_TRADES:
        if trade["pair"] != pair or trade["status"] != "OPEN":
            continue

        expiry_at = pd.Timestamp(trade["expiry_at"])

        # Input only contains fully closed candles. Exact expiry required.
        if m1_df.empty or m1_df.index[-1] + TIMEFRAMES["1min"] < expiry_at:
            continue
        exit_data = _m1_candle_at_expiry(m1_df, expiry_at)
        if exit_data is None:
            # Candle missing: claiming a win/loss from the next candle is invalid.
            trade["status"] = "UNRESOLVED"
            trade["outcome"] = "EXPIRY_CANDLE_MISSING"
            continue

        exit_time = expiry_at
        exit_price = exit_data["exit_price"]
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
        trade["exit_candle_open_time"] = exit_data["exit_candle_open_time"]
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
        "decision_diagnostics": result["decision_diagnostics"],
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
    unresolved_count = sum(1 for trade in trades if trade["status"] == "UNRESOLVED")

    win_rate = (
        round(wins / len(closed) * 100, 2)
        if closed else None
    )

    return make_json_safe({
        "strategy_version": VERSION,
        "total_trades": len(trades),
        "open_trades": open_count,
        "unresolved_trades": unresolved_count,
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
    """Only candles closed by `evaluation_time`, never later.

    Twelve Data stamps the *open* of each OHLC candle. For a candle to
    be visible at evaluation_time, open + timeframe <= evaluation_time.
    An explicit Pandas time-index lookup avoids mixing raw timestamp units.
    This helper also independently validates its own output: fail closed
    rather than silently backtest on future candles.
    """
    evaluation_time = pd.Timestamp(evaluation_time)
    if evaluation_time.tzinfo is None:
        raise ValueError("Backtest evaluation_time must be timezone-aware UTC")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Backtest candles must have a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("Backtest candle indexes must be timezone-aware UTC")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Backtest candles must be chronologically sorted and unique")

    duration = TIMEFRAMES[timeframe]
    last_allowed_open = evaluation_time - duration
    # Native timestamp search on Pandas DatetimeIndex, not nanosecond int64.
    end = int(df.index.searchsorted(last_allowed_open, side="right"))
    result = df.iloc[max(0, end - max_rows):end]
    if len(result) and result.index[-1] + duration > evaluation_time:
        raise RuntimeError(
            f"LOOKAHEAD_BLOCKED: {timeframe}: last_open={result.index[-1]} "
            f"evaluation_time={evaluation_time}"
        )
    return result


def _increment_counter(counter, key):
    key = str(key)
    counter[key] = int(counter.get(key, 0)) + 1


def _classify_evaluation_stage(result):
    """One disjoint reason per M1 evaluation; H1 -> M15 -> M5 -> M1."""
    a = result["analysis"]
    h1 = a["H1"]["signal"]
    if h1 == "WAIT":
        return "H1_WAIT"
    m15 = a["M15"]["signal"]
    if m15 == "WAIT":
        return "M15_WAIT"
    if m15 != h1:
        return "M15_OPPOSITE"
    m5 = a["M5"]["signal"]
    if m5 == "WAIT":
        return "M5_WAIT"
    if m5 != h1:
        return "M5_OPPOSITE"
    m1 = a["M1"]["signal"]
    if m1 == "WAIT":
        return "M1_WAIT"
    if m1 != h1:
        return "M1_OPPOSITE"
    return "SIGNAL" if result["signal"] in ("CALL", "PUT") else "SCORE_BELOW_70"


def _new_funnel_counts():
    return {
        "H1_WAIT": 0,
        "M15_WAIT": 0,
        "M15_OPPOSITE": 0,
        "M5_WAIT": 0,
        "M5_OPPOSITE": 0,
        "M1_WAIT": 0,
        "M1_OPPOSITE": 0,
        "SCORE_BELOW_70": 0,
        "SIGNAL": 0,
    }


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
    unresolved_expiry = 0
    trades = []

    # These are *M1 evaluation observations*, not unique H1/M15/M5 events.
    evaluation_funnel = _new_funnel_counts()
    directional_funnel = {
        side: {
            "h1_direction": 0,
            "h1_m15_aligned": 0,
            "h1_m15_m5_aligned": 0,
            "all_four_aligned": 0,
            "confirmed_signals": 0,
        }
        for side in ("CALL", "PUT")
    }
    m5_wait_reasons = {
        "unique_closed_m5_candles": 0,
        "fvg_reasons": {},
        "order_block_reasons": {},
        "sweep_failed_checks": {},
        "diagnostic_mismatch_counts": {},
    }
    m1_entry_wait_checks = {"evaluations": 0, "failed_checks": {}}
    diagnosed_m5_candles = set()

    timeframe_signals = {
        tf: {"CALL": 0, "PUT": 0, "WAIT": 0}
        for tf in ("H1", "M15", "M5", "M1")
    }
    m15_structures = {"BOS": 0, "CHoCH": 0, "NONE": 0}
    m15_market_structures = {}
    m15_swing_totals = {"highs": 0, "lows": 0, "observations": 0}
    m15_breaks = {"bullish": 0, "bearish": 0}
    m15_memory_status_counts = {}  # Наблюдения, не уникальные торговые события
    m15_memory_age_counts = {}
    m5_components = {
        "sweep_bullish": 0, "sweep_bearish": 0,
        "fvg_call": 0, "fvg_put": 0, "fvg_wait": 0,
        "ob_call": 0, "ob_put": 0, "ob_wait": 0,
        "conflicting_setups": 0
    }
    m1_filters = {
        "breakout_call": 0, "breakout_put": 0,
        "body_filter_pass": 0, "body_filter_fail": 0,
        "range_filter_pass": 0, "range_filter_fail": 0
    }
    final_signals = {"CALL": 0, "PUT": 0, "WAIT": 0}
    final_wait_reasons = {}

    # Audit only: chronological integrity and representative market snapshots.
    # This does not change signal scoring, filters, or trade settlement.
    audit_last_candle_times = {tf: set() for tf in ANALYSIS_TIMEFRAMES}
    audit_signal_signatures = set()
    audit_m1_close_values = set()
    audit_samples = []
    audit_temporal_violations = 0
    audit_m1_window_mismatch = 0
    audit_first_signature = None
    audit_signature_changes = 0
    audit_sample_indices = {0, len(m1) // 4, len(m1) // 2,
                            (3 * len(m1)) // 4, len(m1) - 1}

    # Для каждого закрытия M1 строим срез только из свечей старших ТФ,
    # которые уже закрылись к моменту оценки. Торговые правила не меняются.
    for i in range(len(m1)):
        m1_open_time = m1.index[i]
        evaluation_time = m1_open_time + TIMEFRAMES["1min"]

        window = {
            tf: _closed_candle_slice(
                history[tf], tf, evaluation_time, max_rows=CANDLE_LIMIT
            )
            for tf in ANALYSIS_TIMEFRAMES
        }

        if any(len(window[tf]) < 60 for tf in ANALYSIS_TIMEFRAMES):
            skipped += 1
            continue

        # Independent time guard: every single evaluation must be chronological.
        # If one slice is wrong, STOP immediately; a backtest based on future
        # candles must never return plausible-looking WIN/LOSS/WAIT statistics.
        bad_windows = {
            tf: {
                "last_open": window[tf].index[-1].isoformat(),
                "should_be_closed_by": evaluation_time.isoformat(),
            }
            for tf in ANALYSIS_TIMEFRAMES
            if window[tf].index[-1] + TIMEFRAMES[tf] > evaluation_time
        }
        if window["1min"].index[-1] != m1_open_time:
            bad_windows["1min"] = {
                "last_open": window["1min"].index[-1].isoformat(),
                "expected_open": m1_open_time.isoformat(),
            }
        if bad_windows:
            logger.error(
                "Backtest chronological guard failed: pair=%s index=%d %s",
                pair, i, bad_windows,
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "BACKTEST_LOOKAHEAD_BLOCKED",
                    "message": "Некорректные временные срезы: будущие свечи или M1 не совпадает с проверяемым моментом",
                    "evaluation_time": evaluation_time.isoformat(),
                    "bad_windows": bad_windows,
                    "strategy_version": VERSION,
                },
            )

        evaluated += 1
        result = analyze_multi_timeframe(window, include_m5_diagnostics=False)
        analysis = result["analysis"]

        for tf in ANALYSIS_TIMEFRAMES:
            last_open = window[tf].index[-1]
            audit_last_candle_times[tf].add(last_open.isoformat())
            if last_open + TIMEFRAMES[tf] > evaluation_time:
                audit_temporal_violations += 1
        if window["1min"].index[-1] != m1_open_time:
            audit_m1_window_mismatch += 1
        audit_m1_close_values.add(float(window["1min"].iloc[-1]["close"]))

        signature = (
            analysis["H1"]["signal"], analysis["M15"]["signal"],
            analysis["M15"].get("structure"),
            analysis["M15"].get("event_age_bars"),
            analysis["M5"]["signal"], analysis["M5"].get("fvg"),
            analysis["M5"].get("order_block"),
            analysis["M1"]["signal"],
            analysis["M1"].get("breakout_call"),
            analysis["M1"].get("breakout_put"),
        )
        audit_signal_signatures.add(signature)
        if audit_first_signature is None:
            audit_first_signature = signature
        elif signature != audit_first_signature:
            audit_signature_changes += 1

        if i in audit_sample_indices or evaluated == 1:
            audit_samples.append({
                "evaluation_time": evaluation_time.isoformat(),
                "m1_open_time": m1_open_time.isoformat(),
                "last_closed_candle_open_times": {
                    tf: window[tf].index[-1].isoformat()
                    for tf in ANALYSIS_TIMEFRAMES
                },
                "last_closed_candle_prices": {
                    tf: float(window[tf].iloc[-1]["close"])
                    for tf in ANALYSIS_TIMEFRAMES
                },
                "m15_memory_age": analysis["M15"].get("event_age_bars"),
                "alignment": result["alignment"],
                "final_signal": result["signal"],
            })

        # Частота направлений каждого отдельного фильтра.
        for label in ("H1", "M15", "M5", "M1"):
            _increment_counter(timeframe_signals[label], analysis[label]["signal"])

        # Диагностика M15: тип структуры, состояние рынка, доступность swing.
        m15 = analysis["M15"]
        _increment_counter(m15_structures, m15.get("structure", "NONE"))
        _increment_counter(m15_market_structures, m15.get("market_structure", "UNKNOWN"))
        m15_swing_totals["highs"] += int(m15.get("confirmed_swing_highs", 0))
        m15_swing_totals["lows"] += int(m15.get("confirmed_swing_lows", 0))
        m15_swing_totals["observations"] += 1
        if m15.get("bullish_break_detected"):
            m15_breaks["bullish"] += 1
        if m15.get("bearish_break_detected"):
            m15_breaks["bearish"] += 1
        _increment_counter(m15_memory_status_counts, m15.get("memory_status", "UNKNOWN"))
        if m15.get("memory_status") in ("FRESH", "REMEMBERED"):
            _increment_counter(m15_memory_age_counts, m15.get("event_age_bars", "UNKNOWN"))

        # Диагностика отдельных моделей M5, включая конфликты направлений.
        m5 = analysis["M5"]
        sweep = m5.get("liquidity_sweep", {})
        if sweep.get("bullish"):
            m5_components["sweep_bullish"] += 1
        if sweep.get("bearish"):
            m5_components["sweep_bearish"] += 1
        fvg = m5.get("fvg", "WAIT")
        ob = m5.get("order_block", "WAIT")
        _increment_counter(m5_components, f"fvg_{str(fvg).lower()}")
        _increment_counter(m5_components, f"ob_{str(ob).lower()}")
        details = m5.get("details", {})
        if details.get("call_points_total", 0) > 0 and details.get("put_points_total", 0) > 0:
            m5_components["conflicting_setups"] += 1

        # Диагностика M1: отдельно пробой, тело и размер свечи.
        m1_analysis = analysis["M1"]
        for field in ("breakout_call", "breakout_put", "body_filter_pass", "range_filter_pass"):
            if m1_analysis.get(field):
                m1_filters[field] += 1
        if "body_filter_pass" in m1_analysis:
            fail_key = "body_filter_fail" if not m1_analysis["body_filter_pass"] else None
            if fail_key:
                m1_filters[fail_key] += 1
        if "range_filter_pass" in m1_analysis:
            fail_key = "range_filter_fail" if not m1_analysis["range_filter_pass"] else None
            if fail_key:
                m1_filters[fail_key] += 1

        # Disjoint funnel stages: each snapshot counted once.
        stage = _classify_evaluation_stage(result)
        evaluation_funnel[stage] += 1
        side = analysis["H1"]["signal"]
        if side in directional_funnel:
            dfunnel = directional_funnel[side]
            dfunnel["h1_direction"] += 1
            if analysis["M15"]["signal"] == side:
                dfunnel["h1_m15_aligned"] += 1
                if analysis["M5"]["signal"] == side:
                    dfunnel["h1_m15_m5_aligned"] += 1
                    if analysis["M1"]["signal"] == side:
                        dfunnel["all_four_aligned"] += 1
                        if result["signal"] == side:
                            dfunnel["confirmed_signals"] += 1

            # Rich M5 diagnostics only when this is the first blocking stage;
            # deduplicate by closed M5 candle + direction to avoid counting
            # the same M5 observation once per M1 minute.
            if stage == "M5_WAIT":
                m5_key = (window["5min"].index[-1], side)
                if m5_key not in diagnosed_m5_candles:
                    diagnosed_m5_candles.add(m5_key)
                    diag = diagnose_m5(
                        window["5min"],
                        bool(m5["liquidity_sweep"]["bullish"]),
                        bool(m5["liquidity_sweep"]["bearish"]),
                        m5["fvg"], m5["order_block"]
                    )
                    m5_wait_reasons["unique_closed_m5_candles"] += 1
                    _increment_counter(m5_wait_reasons["fvg_reasons"],
                                       diag["fvg"].get("reason", "UNKNOWN"))
                    _increment_counter(m5_wait_reasons["order_block_reasons"],
                                       diag["order_block"].get("reason", "UNKNOWN"))
                    sweep_side = "bullish" if side == "CALL" else "bearish"
                    for check in diag["liquidity_sweep"][sweep_side].get("failed_checks", []):
                        _increment_counter(m5_wait_reasons["sweep_failed_checks"], check)
                    for check, passed in diag["consistency_checks"].items():
                        if not passed:
                            _increment_counter(m5_wait_reasons["diagnostic_mismatch_counts"], check)

            # We analyze M1 blockers only after H1, M15 and M5 all agree.
            if stage == "M1_WAIT":
                m1_entry_wait_checks["evaluations"] += 1
                for check in m1_analysis.get("failed_checks", []):
                    _increment_counter(m1_entry_wait_checks["failed_checks"], check)

        _increment_counter(final_signals, result["signal"])
        if result["signal"] == "WAIT":
            _increment_counter(final_wait_reasons, result["reason"])
            continue

        signals += 1
        entry_price = float(m1.iloc[i]["close"])
        entry_time = m1_open_time + TIMEFRAMES["1min"]
        expiry_at = entry_time + expiry_duration
        # Exact expiration price: M1 candle opens at expiry - 1m and
        # closes at expiry. Missing candle means unresolved, never a guess.
        exit_data = _m1_candle_at_expiry(m1, expiry_at)
        if exit_data is None:
            unresolved_expiry += 1
            continue

        exit_time = expiry_at
        exit_price = exit_data["exit_price"]

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
                "exit_candle_open_time": exit_data["exit_candle_open_time"],
                "exit_price": exit_price,
                "outcome": outcome,
                "score": result["score"]
            })

    closed_trades = wins + losses + draws
    win_rate = round(wins / closed_trades * 100, 2) if closed_trades else None

    audit_warnings = []
    if audit_temporal_violations:
        audit_warnings.append("FUTURE_CANDLES_IN_HISTORICAL_SLICES")
    if audit_m1_window_mismatch:
        audit_warnings.append("M1_WINDOW_NOT_ALIGNED_WITH_EVALUATION")
    if evaluated >= 100:
        for tf in ("1min", "5min", "15min", "1h"):
            if len(audit_last_candle_times[tf]) <= 1:
                audit_warnings.append(f"FROZEN_LAST_CLOSED_CANDLE_{tf}")
        if len(audit_signal_signatures) == 1:
            audit_warnings.append("IDENTICAL_SIGNAL_DIAGNOSTICS_AT_ALL_TIMES")
        if len(audit_m1_close_values) == 1:
            audit_warnings.append("IDENTICAL_M1_CLOSE_AT_ALL_TIMES")
    if audit_temporal_violations or audit_m1_window_mismatch:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "BACKTEST_LOOKAHEAD_BLOCKED",
                "message": "Исторические свечи содержат будущее относительно времени проверки",
                "future_candle_violations": audit_temporal_violations,
                "m1_window_alignment_violations": audit_m1_window_mismatch,
                "strategy_version": VERSION,
            },
        )
    audit = {
        "status": "SUSPICIOUS" if audit_warnings else "CHECKED",
        "warnings": audit_warnings,
        "slice_engine": "pandas_datetime_searchsorted_v2",
        "integrity_guard_enabled": True,
        "time_basis": "UTC; Twelve Data candles indexed by opening time",
        "history_ranges": {
            tf: {
                "rows": int(len(history[tf])),
                "first_candle_open": history[tf].index[0].isoformat(),
                "last_candle_open": history[tf].index[-1].isoformat(),
            }
            for tf in ANALYSIS_TIMEFRAMES
        },
        "unique_last_closed_candle_times": {
            tf: len(audit_last_candle_times[tf])
            for tf in ANALYSIS_TIMEFRAMES
        },
        "unique_m1_close_values": len(audit_m1_close_values),
        "unique_signal_diagnostic_signatures": len(audit_signal_signatures),
        "observations_differing_from_first_signature": audit_signature_changes,
        "future_candle_violations": audit_temporal_violations,
        "m1_window_alignment_violations": audit_m1_window_mismatch,
        "sample_evaluations": audit_samples,
    }

    return {
        "strategy_version": VERSION,
        "pair": pair,
        "expiry_timeframe": expiry_timeframe,
        "requested_m1_candles": candle_count,
        "available_candles": {tf: int(len(history[tf])) for tf in ANALYSIS_TIMEFRAMES},
        "evaluated_points": evaluated,
        "skipped_points_insufficient_history": skipped,
        "signals_generated": signals,
        "closed_trades": closed_trades,
        "unresolved_expiry_candles": unresolved_expiry,
        "signals_per_1000_evaluations": round(signals * 1000 / evaluated, 2) if evaluated else 0,
        "signal_rate_percent": round(signals * 100 / evaluated, 3) if evaluated else 0,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate_percent": win_rate,
        "trades_returned": len(trades),
        "trades": trades,
        "backtest_data_audit": audit,
        "diagnostics": {
            "timeframe_signal_counts": timeframe_signals,
            "m15": {
                "structure_counts": m15_structures,
                "market_structure_counts": m15_market_structures,
                "average_confirmed_swings_per_evaluation": {
                    "highs": round(m15_swing_totals["highs"] / m15_swing_totals["observations"], 2) if evaluated else 0,
                    "lows": round(m15_swing_totals["lows"] / m15_swing_totals["observations"], 2) if evaluated else 0
                },
                "break_detections": m15_breaks,
                "memory_window_bars": M15_MEMORY_BARS,
                "memory_status_observations": m15_memory_status_counts,
                "memory_age_observations": m15_memory_age_counts
            },
            "m5": m5_components,
            "m1": m1_filters,
            "final_signal_counts": final_signals,
            "final_wait_reasons": final_wait_reasons,
            "wait_funnel": {
                "evaluations": evaluated,
                "stages": evaluation_funnel,
                "directional": directional_funnel,
                "description": (
                    "Каждая проверенная закрытая M1 свеча учитывается один раз; "
                    "ступени воронки взаимоисключающие. H1/M15/M5 события "
                    "могут повторяться в разных M1 наблюдениях."
                ),
            },
            "targeted_blockers": {
                "m5_when_h1_m15_aligned": m5_wait_reasons,
                "m1_when_h1_m15_m5_aligned": m1_entry_wait_checks,
            }
        },
        "limitations": [
            "Backtest использует доступную историю Twelve Data и текущие правила стратегии.",
            "Комиссии, спред, проскальзывание и выплаты брокера не учитываются.",
            "Результат не гарантирует будущую доходность.",
            "Если сделок больше 200, в ответе показаны только первые 200; итоговая статистика учитывает все закрытые сделки.",
            "Статистика M15 memory считает наблюдения на M1-срезах; это не уникальные события BOS/CHoCH.",
            "Диагностика показывает частоту условий фильтров, но сама по себе не доказывает прибыльность стратегии.",
            "Экспирация измеряется по закрытию M1 свечи ровно в момент expiry_at; если такой свечи нет, исход остаётся неопределённым.",
            "Статистика воронки — по проверкам M1, а не независимым торговым возможностям; соседние проверки статистически зависимы.",
            "WIN/LOSS не учитывает цену входа/выхода на брокере и скорость исполнения.",
            "backtest_data_audit проверяет временную целостность и разнообразие результатов; предупреждения требуют проверки входных свечей."
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
        "twelvedata_key_configured": bool(TWELVEDATA_KEY),
        "backtest_slice_engine": "pandas_datetime_searchsorted_v2"
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
