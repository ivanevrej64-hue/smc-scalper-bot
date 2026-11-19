import os
import math
import logging
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from collections import deque
import threading

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

VERSION = "2.4.19"

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
# Fetch M1 first in LIVE scans; scoring order and historical backtest stay untouched.
# The same shared cutoff remains in force for all four downloads.
LIVE_FETCH_TIMEFRAMES = ["1min", "1h", "15min", "5min"]

CANDLE_LIMIT = 150
BACKTEST_CANDLE_LIMIT = 5000
FX_MARKET_TIMEZONE = ZoneInfo("America/New_York")
PAYOUT_RATES = (0.70, 0.80, 0.90)
SWING_LENGTH = 5
M15_MEMORY_BARS = 8  # Срок жизни подтвержденного BOS/CHoCH после свечи пробоя
CANDLE_SAFETY_SECONDS = 3
# A 1-minute paper signal observed >30 seconds after candle close is too late
# to model a practical entry. This is a safety limit, NOT an optimisation of the strategy.
PAPER_MAX_SIGNAL_AGE_SECONDS = 30
# Conservative, bounded refresh of a missing closed M1 bar. The external
# cron-job.org request has a 30-second timeout: at most 2 more M1 calls, never
# extend the 30-second signal freshness window, and reserve time for analysis.
M1_RETRY_ENABLED = os.getenv("M1_RETRY_ENABLED", "true").lower() in ("1", "true", "yes")
M1_RETRY_MAX_ATTEMPTS = 2
M1_RETRY_WAIT_SECONDS = 5.0
M1_RETRY_SAFETY_MARGIN_SECONDS = 3.0

MIN_SCORE = 70
MIN_M5_SCORE_EDGE = 10

SIGNAL_HISTORY = deque(maxlen=500)
VIRTUAL_TRADES = deque(maxlen=500)


# ==================================================
# TWELVE DATA — PROCESS-LOCAL CREDIT BUDGET (BASIC PLAN)
# ==================================================
# Twelve Data Basic: 8 credits/minute, 800 credits/day; /time_series is
# one credit per requested symbol. Keep one credit of headroom for occasional
# independent API users. This counter protects only THIS Python process;
# other programs sharing the key, multi-worker processes, and restarts are
# detected through the provider's 429 response but are not pre-counted.
TD_PROVIDER_MINUTE_CREDITS = max(4, int(os.getenv("TD_PROVIDER_MINUTE_CREDITS", "8")))
TD_PROVIDER_DAILY_CREDITS = max(4, int(os.getenv("TD_PROVIDER_DAILY_CREDITS", "800")))
TD_MINUTE_RESERVE = max(0, int(os.getenv("TD_MINUTE_RESERVE", "1")))
TD_LOCAL_MINUTE_BUDGET = max(1, TD_PROVIDER_MINUTE_CREDITS - TD_MINUTE_RESERVE)
TD_LIVE_SCAN_COST = len(ANALYSIS_TIMEFRAMES)  # 4 required calls before M1 retries.

_TD_RATE_LOCK = threading.Lock()
_TD_LIVE_SCAN_LOCK = threading.Lock()
_TD_USAGE = {
    "utc_minute": None,
    "utc_day": None,
    "minute_used": 0,
    "day_used": 0,
    "upstream_block_minute": None,
    "upstream_block_day": None,
    "last_upstream_429_utc": None,
}


def _td_reset_windows_locked(now):
    minute = int(now.timestamp() // 60)
    day = int(now.timestamp() // 86400)
    if _TD_USAGE["utc_minute"] != minute:
        _TD_USAGE["utc_minute"] = minute
        _TD_USAGE["minute_used"] = 0
        _TD_USAGE["upstream_block_minute"] = None
    if _TD_USAGE["utc_day"] != day:
        _TD_USAGE["utc_day"] = day
        _TD_USAGE["day_used"] = 0
        _TD_USAGE["upstream_block_day"] = None


def _td_quota_snapshot():
    with _TD_RATE_LOCK:
        now = pd.Timestamp.now(tz="UTC")
        _td_reset_windows_locked(now)
        state = dict(_TD_USAGE)
    return {
        "scope": "current_process_only",
        "provider_plan_minute_limit": TD_PROVIDER_MINUTE_CREDITS,
        "local_minute_budget": TD_LOCAL_MINUTE_BUDGET,
        "local_minute_used": state["minute_used"],
        "local_minute_remaining": max(0, TD_LOCAL_MINUTE_BUDGET - state["minute_used"]),
        "provider_plan_daily_limit": TD_PROVIDER_DAILY_CREDITS,
        "local_day_used": state["day_used"],
        "local_day_remaining": max(0, TD_PROVIDER_DAILY_CREDITS - state["day_used"]),
        "upstream_minute_blocked": state["upstream_block_minute"] == state["utc_minute"],
        "upstream_day_blocked": state["upstream_block_day"] == state["utc_day"],
        "last_upstream_429_utc": state["last_upstream_429_utc"],
        "warning": "External clients sharing the key and other worker processes are not counted.",
    }


def _td_budget_error(reason, now=None, m1_diagnostics=None):
    now = now or pd.Timestamp.now(tz="UTC")
    remaining_today = 86400 - int(now.timestamp()) % 86400
    remaining_minute = 60 - int(now.timestamp()) % 60
    daily = reason in ("LOCAL_DAILY_BUDGET", "UPSTREAM_DAILY_QUOTA")
    retry_after = remaining_today if daily else remaining_minute
    detail = {
        "code": "API_RATE_LIMITED",
        "reason": reason,
        "message": "Twelve Data credit budget exhausted. This scan did not run; retry after quota reset.",
        "retry_after_seconds": int(retry_after),
        "quota_scope": "local_process_or_upstream_429",
    }
    if m1_diagnostics is not None:
        detail["m1_diagnostics"] = m1_diagnostics
    return HTTPException(status_code=429, detail=detail, headers={"Retry-After": str(retry_after)})


def _td_claim_credits(credits=1, reserve_only=False):
    """Check/claim credits atomically BEFORE making an actual network request.

    For a 4-timeframe scan, reserve_only=True rejects an obviously
    unaffordable scan at the start, without spending 1-3 credits on a partial
    analysis. Each network request still claims a credit in its actual minute.
    """
    with _TD_RATE_LOCK:
        now = pd.Timestamp.now(tz="UTC")
        _td_reset_windows_locked(now)
        reason = None
        if _TD_USAGE["upstream_block_day"] == _TD_USAGE["utc_day"]:
            reason = "UPSTREAM_DAILY_QUOTA"
        elif _TD_USAGE["upstream_block_minute"] == _TD_USAGE["utc_minute"]:
            reason = "UPSTREAM_MINUTE_QUOTA"
        elif _TD_USAGE["day_used"] + credits > TD_PROVIDER_DAILY_CREDITS:
            reason = "LOCAL_DAILY_BUDGET"
        elif _TD_USAGE["minute_used"] + credits > TD_LOCAL_MINUTE_BUDGET:
            reason = "LOCAL_MINUTE_BUDGET"
        if reason:
            raise _td_budget_error(reason, now)
        if not reserve_only:
            _TD_USAGE["minute_used"] += credits
            _TD_USAGE["day_used"] += credits


def _td_is_provider_429(exc):
    message = str(exc).lower()
    return ("429" in message and ("credit" in message or "rate" in message or "too many" in message))


def _td_note_provider_429(exc):
    message = str(exc).lower()
    daily = "daily" in message or "per day" in message
    with _TD_RATE_LOCK:
        now = pd.Timestamp.now(tz="UTC")
        _td_reset_windows_locked(now)
        if daily:
            _TD_USAGE["upstream_block_day"] = _TD_USAGE["utc_day"]
        else:
            _TD_USAGE["upstream_block_minute"] = _TD_USAGE["utc_minute"]
        _TD_USAGE["last_upstream_429_utc"] = now.isoformat()
    logger.warning("Twelve Data upstream 429: quota exhausted (%s); back off until UTC reset", "daily" if daily else "minute")
    return _td_budget_error("UPSTREAM_DAILY_QUOTA" if daily else "UPSTREAM_MINUTE_QUOTA", now)



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
    # Влияет только на исторические проверки, не на /api/signal.
    market_mode: str = Field(default="regular_forex")


class BacktestCompareRequest(BaseModel):
    pair: str = Field(default="EUR/USD")
    weeks: int = Field(default=2, ge=1, le=3)
    candles_per_week: int = Field(default=1500, ge=1000, le=3000)


class BacktestRobustnessRequest(BaseModel):
    # Only historical, completed regular-Forex weekly windows are tested.
    pair: str = Field(default="EUR/USD")
    weeks: int = Field(default=3, ge=1, le=6)
    candles_per_week: int = Field(default=3000, ge=1000, le=3000)
    expiry_timeframe: str = Field(default="1min")
    # Treat signals closer than this as the same trading opportunity.
    cooldown_minutes: int = Field(default=5, ge=0, le=30)


# ==================================================
# MARKET DATA
# ==================================================

def get_market_data(
    pair: str,
    timeframe: str,
    outputsize: int = CANDLE_LIMIT,
    end_date=None,
    as_of_utc=None,
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

        request_args = {
            "symbol": pair,
            "interval": timeframe,
            "outputsize": outputsize,
            "timezone": "UTC",
        }
        if end_date is not None:
            bound = pd.Timestamp(end_date)
            if bound.tzinfo is None:
                raise HTTPException(
                    status_code=400,
                    detail="end_date должен содержать часовой пояс",
                )
            request_args["end_date"] = bound.tz_convert("UTC").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        provider_request_started_at = pd.Timestamp.now(tz="UTC")
        _td_claim_credits(1)
        response = td.time_series(**request_args)

        df = response.as_pandas()
        provider_response_received_at = pd.Timestamp.now(tz="UTC")

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
        if df.empty:
            raise HTTPException(status_code=502, detail=f"No valid source timestamps: {pair} {timeframe}")
        raw_provider_times = df.index.copy()  # Before OHLC validation.

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
        if df.empty:
            raise HTTPException(status_code=502, detail=f"No valid source OHLC: {pair} {timeframe}")

        # One shared observation clock for H1/M15/M5/M1. A live request must
        # never include a candle which was unfinished at analysis start.
        as_of = pd.Timestamp.now(tz="UTC") if as_of_utc is None else pd.Timestamp(as_of_utc)
        if pd.isna(as_of) or as_of.tzinfo is None:
            raise HTTPException(status_code=400, detail="as_of_utc must be timezone-aware")
        as_of = as_of.tz_convert("UTC")
        cutoff = as_of - TIMEFRAMES[timeframe] - timedelta(seconds=CANDLE_SAFETY_SECONDS)
        expected_open = cutoff.floor(pd.Timedelta(TIMEFRAMES[timeframe]))
        raw_latest_open = pd.Timestamp(raw_provider_times[-1])
        raw_has_expected = bool(expected_open in raw_provider_times)
        valid_has_expected = bool(expected_open in df.index)
        # Classify *the actual provider response* before the no-look-ahead filter.
        # This distinguishes a delayed/missing bar from an accidental rejection
        # of a bar which had already arrived. No secret/API key is included.
        raw_lag_seconds = float((expected_open - raw_latest_open).total_seconds())
        raw_incomplete = bool(raw_latest_open + TIMEFRAMES[timeframe] > as_of)

        df = df[df.index <= cutoff]

        if len(df) < 60:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Недостаточно закрытых свечей: "
                    f"{pair} {timeframe}. Получено: {len(df)}"
                )
            )

        latest_usable_open = pd.Timestamp(df.index[-1])
        if raw_has_expected and not valid_has_expected:
            diagnosis = "EXPECTED_BAR_REJECTED_INVALID_OHLC"
        elif raw_has_expected and latest_usable_open < expected_open:
            diagnosis = "FILTER_REJECTED_AVAILABLE_EXPECTED_BAR"
        elif not raw_has_expected and raw_latest_open < expected_open:
            diagnosis = "PROVIDER_HAS_NOT_RETURNED_EXPECTED_BAR"
        elif not raw_has_expected:
            diagnosis = "EXPECTED_BAR_MISSING_IN_PROVIDER_SERIES"
        else:
            diagnosis = "EXPECTED_CLOSED_BAR_AVAILABLE"
        diagnostics = {
            "provider": "Twelve Data",
            "interval": timeframe,
            "request_started_at_utc": provider_request_started_at.isoformat(),
            "response_received_at_utc": provider_response_received_at.isoformat(),
            "analysis_cutoff_utc": as_of.isoformat(),
            "expected_latest_closed_open_utc": expected_open.isoformat(),
            "raw_latest_open_utc": raw_latest_open.isoformat(),
            "raw_latest_may_be_open_candle": raw_incomplete,
            "raw_has_expected_closed_bar": raw_has_expected,
            "expected_bar_passed_ohlc_validation": valid_has_expected,
            "raw_latest_minus_expected_seconds": round(-raw_lag_seconds, 3),
            "filtered_latest_open_utc": latest_usable_open.isoformat(),
            "filtered_out_newer_raw_bar": bool(raw_latest_open > latest_usable_open),
            "diagnosis": diagnosis,
        }
        result = df.tail(outputsize)
        result.attrs["live_provider_diagnostics"] = diagnostics
        return result

    except HTTPException:
        raise

    except Exception as exc:
        if _td_is_provider_429(exc):
            raise _td_note_provider_429(exc) from exc
        logger.exception(
            "Ошибка получения данных %s %s",
            pair,
            timeframe
        )
        raise HTTPException(
            status_code=502,
            detail="Ошибка получения данных Twelve Data"
        ) from exc


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
            "all_timeframes_aligned": len(set(directions)) == 1 and directions[0] in ("CALL", "PUT")
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


def create_virtual_trade(pair, signal, candle_time, entry_price, timeframe):
    """Deprecated 2.4.16: never invent a past execution at candle close.

    Keep the signature for compatibility with older integrations; no fills
    or outcomes may be created without an observed DEMO broker entry.
    """
    return None


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
# LIVE DATA FRESHNESS (NO LOOK-AHEAD, NO RETROSPECTIVE EXECUTIONS)
# ==================================================

def _live_data_freshness(market_data, analysis_at_utc, observed_at_utc):
    """How old were the last fully closed candles when the analysis was observed?

    Uses the actual END of each candle, not its Twelve Data opening timestamp.
    H1/M15/M5 are contextual; only M1 freshness gates a 1-minute signal.
    No price or fill is reconstructed from a past candle after the fact.
    """
    analysis_at = pd.Timestamp(analysis_at_utc).tz_convert("UTC")
    observed_at = pd.Timestamp(observed_at_utc).tz_convert("UTC")
    candles = {}
    for tf in ANALYSIS_TIMEFRAMES:
        last_open = pd.Timestamp(market_data[tf].index[-1]).tz_convert("UTC")
        last_close = last_open + TIMEFRAMES[tf]
        candles[tf] = {
            "last_candle_open_utc": last_open.isoformat(),
            "last_candle_close_utc": last_close.isoformat(),
            "age_seconds": round(float((observed_at - last_close).total_seconds()), 3),
        }

    m1_close = pd.Timestamp(candles["1min"]["last_candle_close_utc"])
    age_seconds = float((observed_at - m1_close).total_seconds())
    if age_seconds < 0:
        status = "NOT_CLOSED"
        reason = "M1 candle has not closed at observation time"
    elif age_seconds > PAPER_MAX_SIGNAL_AGE_SECONDS:
        status = "STALE"
        reason = (
            f"M1 signal candle is {age_seconds:.1f}s old; "
            f"maximum {PAPER_MAX_SIGNAL_AGE_SECONDS}s"
        )
    else:
        status = "FRESH"
        reason = "M1 candle is recently closed"

    return {
        "status": status,
        "reason": reason,
        "analysis_at_utc": analysis_at.isoformat(),
        "observed_at_utc": observed_at.isoformat(),
        "signal_available_at_utc": m1_close.isoformat(),
        "signal_age_seconds": round(age_seconds, 3),
        "max_signal_age_seconds": PAPER_MAX_SIGNAL_AGE_SECONDS,
        "candles": candles,
    }


# ==================================================
# BOUNDED M1 BAR REFRESH (ONLY WHILE A FRESH SIGNAL IS STILL POSSIBLE)
# ==================================================

def _refresh_missing_m1_bar(pair, market_data, analysis_at_utc):
    """Retry only the missing closed M1 bar; never reuse unfinished candles.

    The shared analysis cutoff stays fixed. A refresh only supplies a candle
    which had already closed at the start of the initial four-timeframe scan.
    Requests are capped to fit the existing 30s cron-job.org HTTP timeout.
    """
    import time as _time

    initial = market_data["1min"]
    initial_open = pd.Timestamp(initial.index[-1])
    as_of = pd.Timestamp(analysis_at_utc).tz_convert("UTC")
    cutoff = as_of - TIMEFRAMES["1min"] - timedelta(seconds=CANDLE_SAFETY_SECONDS)
    expected_open = cutoff.floor("min")
    expected_close = expected_open + TIMEFRAMES["1min"]
    # Leave a few seconds to evaluate MTF, persist the scan and respond.
    deadline = expected_close + timedelta(
        seconds=PAPER_MAX_SIGNAL_AGE_SECONDS - M1_RETRY_SAFETY_MARGIN_SECONDS
    )
    meta = {
        "enabled": bool(M1_RETRY_ENABLED),
        "initial_latest_closed_open_utc": initial_open.isoformat(),
        "expected_latest_closed_open_utc": expected_open.isoformat(),
        "attempts": 0,
        "history": [],
        "updated": False,
        "stop_reason": None,
    }
    if initial_open >= expected_open:
        meta["stop_reason"] = "ALREADY_LATEST_CLOSED_BAR"
        return meta
    if not M1_RETRY_ENABLED:
        meta["stop_reason"] = "RETRY_DISABLED"
        return meta

    for i in range(M1_RETRY_MAX_ATTEMPTS):
        seconds_left = (deadline - pd.Timestamp.now(tz="UTC")).total_seconds()
        # Avoid an expensive call once it cannot complete before expiry.
        if seconds_left < M1_RETRY_WAIT_SECONDS + 1.0:
            meta["stop_reason"] = "FRESHNESS_DEADLINE_NEAR"
            break
        _time.sleep(M1_RETRY_WAIT_SECONDS)
        try:
            candidate = get_market_data(pair, "1min", as_of_utc=as_of)
        except HTTPException as exc:
            # A transient provider failure on a retry must never erase the
            # already valid initial snapshot or create a false signal.
            meta["history"].append({
                "attempt": i + 1,
                "result": "API_RATE_LIMITED" if exc.status_code == 429 else "RETRY_FAILED",
                "http_status": exc.status_code,
            })
            if exc.status_code == 429:
                meta["stop_reason"] = "API_RATE_LIMITED"
                meta["rate_limit_reason"] = (
                    exc.detail.get("reason", "LOCAL_MINUTE_BUDGET")
                    if isinstance(exc.detail, dict) else "LOCAL_MINUTE_BUDGET"
                )
                break
            continue
        candidate_open = pd.Timestamp(candidate.index[-1])
        provider_diag = candidate.attrs.get("live_provider_diagnostics", {})
        meta["attempts"] += 1
        meta["history"].append({
            "attempt": i + 1,
            "latest_closed_open_utc": candidate_open.isoformat(),
            "provider_raw_latest_open_utc": provider_diag.get("raw_latest_open_utc"),
            "provider_diagnosis": provider_diag.get("diagnosis"),
        })
        if candidate_open > pd.Timestamp(market_data["1min"].index[-1]):
            market_data["1min"] = candidate
            meta["updated"] = True
        if candidate_open >= expected_open:
            meta["stop_reason"] = "EXPECTED_BAR_RECOVERED"
            break
    if meta["stop_reason"] is None:
        meta["stop_reason"] = "RETRIES_EXHAUSTED"
    return meta


# ==================================================
# v2.4.19 — READ-ONLY LIVE DIAGNOSTICS; STRATEGY UNCHANGED
# ==================================================

def _live_fetch_pipeline_diagnostics(analysis_at, observed_at, data,
                                    initial_m1_diag, final_m1_diag, retry_audit):
    """Report timestamps without re-requesting any candle or adding credits."""
    analysis_at = pd.Timestamp(analysis_at).tz_convert("UTC")
    observed_at = pd.Timestamp(observed_at).tz_convert("UTC")
    per_tf = {}
    for tf in LIVE_FETCH_TIMEFRAMES:
        diag = data[tf].attrs.get("live_provider_diagnostics") or {}
        request_at = diag.get("request_started_at_utc")
        response_at = diag.get("response_received_at_utc")
        duration_ms = None
        if request_at and response_at:
            duration_ms = round(float((pd.Timestamp(response_at) - pd.Timestamp(request_at)).total_seconds() * 1000), 1)
        per_tf[tf] = {
            "last_closed_open_utc": pd.Timestamp(data[tf].index[-1]).isoformat(),
            "request_started_at_utc": request_at,
            "response_received_at_utc": response_at,
            "provider_elapsed_ms": duration_ms,
            "provider_diagnosis": diag.get("diagnosis"),
        }
    return {
        "fetch_order": list(LIVE_FETCH_TIMEFRAMES),
        "m1_priority": True,
        "shared_analysis_at_utc": analysis_at.isoformat(),
        "observed_at_utc": observed_at.isoformat(),
        "pipeline_elapsed_ms": round(float((observed_at - analysis_at).total_seconds() * 1000), 1),
        "m1_initial_provider_diagnostics": initial_m1_diag,
        "m1_final_provider_diagnostics": final_m1_diag,
        "m1_refresh_attempts": int(retry_audit.get("attempts", 0)),
        "m1_refresh_stop_reason": retry_audit.get("stop_reason"),
        "per_timeframe": per_tf,
        "note": "Provider timing is measured at Render, not exchange or broker execution time.",
    }


def _live_decision_audit(signal_data):
    """A compact interpretation of the original raw multi-timeframe decision.

    This is read-only. It MUST NOT modify the score, direction or freshness gate.
    """
    analysis = signal_data.get("analysis") or {}
    alignment = signal_data.get("alignment") or {}
    score = signal_data.get("score") or {}
    raw = signal_data.get("raw_signal_before_freshness_gate", signal_data.get("signal", "WAIT"))
    by_tf = {}
    for tf in ("H1", "M15", "M5", "M1"):
        component = analysis.get(tf) or {}
        by_tf[tf] = {"signal": str(component.get("signal", alignment.get(tf, "UNKNOWN"))),
                     "reason": component.get("reason")}
    wait_timeframes = [tf for tf, item in by_tf.items() if item["signal"] == "WAIT"]
    signals = [item["signal"] for item in by_tf.values()]
    if raw in ("CALL", "PUT"):
        primary = "SETUP_CONFIRMED"
    elif wait_timeframes:
        primary = wait_timeframes[0]
    elif len(set(signals)) > 1:
        primary = "DIRECTION_CONFLICT"
    else:
        primary = "INSUFFICIENT_SCORE"
    h1 = analysis.get("H1") or {}
    m15 = analysis.get("M15") or {}
    m5 = analysis.get("M5") or {}
    m1 = analysis.get("M1") or {}
    m5diag = m5.get("diagnostics") or {}
    return make_json_safe({
        "raw_signal": raw,
        "raw_reason": signal_data.get("raw_reason_before_freshness_gate", signal_data.get("reason")),
        "primary_blocker": primary,
        "wait_timeframes": wait_timeframes,
        "per_timeframe": by_tf,
        "score": {"CALL": score.get("CALL"), "PUT": score.get("PUT"),
                  "minimum_required": score.get("minimum_required", MIN_SCORE)},
        "h1": {"block_reason": h1.get("block_reason"),
               "close": h1.get("h1_close"), "ema20": h1.get("ema20"), "ema50": h1.get("ema50"),
               "close_minus_ema20": h1.get("close_minus_ema20"),
               "ema20_minus_ema50": h1.get("ema20_minus_ema50")},
        "m15": {"reason_code": m15.get("diagnostic_reason_code"),
                "memory_status": m15.get("memory_status"),
                "market_structure": m15.get("market_structure"),
                "event_age_bars": m15.get("event_age_bars")},
        "m5": {"sweep_bullish": (m5.get("liquidity_sweep") or {}).get("bullish"),
               "sweep_bearish": (m5.get("liquidity_sweep") or {}).get("bearish"),
               "fvg_reason": (m5diag.get("fvg") or {}).get("reason"),
               "ob_reason": (m5diag.get("order_block") or {}).get("reason")},
        "m1": {"breakout_call": m1.get("breakout_call"),
               "breakout_put": m1.get("breakout_put"),
               "failed_checks": m1.get("failed_checks", [])},
    })


def _paper_scan_diagnostic_payload(timing):
    """Keep the existing DB schema; append a versioned namespace to JSON."""
    payload = dict((timing or {}).get("candles") or {})
    if (timing or {}).get("decision_audit") is not None:
        payload["_v2419_decision_audit"] = timing["decision_audit"]
    if (timing or {}).get("live_pipeline") is not None:
        payload["_v2419_live_pipeline"] = timing["live_pipeline"]
    return payload


def _paper_parse_diagnostic_payload(payload):
    import json
    try:
        data = json.loads(payload or "{}")
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def _paper_diagnostic_summary(rows, hours, truncated=False):
    """Summarise persisted scans, excluding legacy rows from blocker statistics.

    H1/M15/etc blocker distributions use only FRESH analyzed scans so stale
    market snapshots do not get mistaken for executable missed setups.
    """
    from collections import Counter
    status_counts, primary, tf_wait, h1_reasons, m15_codes, m5_reasons = (Counter() for _ in range(6))
    m1_signals, raw_signals, provider_diagnoses, refresh_results = (Counter() for _ in range(4))
    m5_fvg_reasons, m5_ob_reasons, m1_failed_checks = (Counter() for _ in range(3))
    fresh_ages, stale_ages = [], []
    analyzed, legacy, fresh_analyzed, stale_analyzed = 0, 0, 0, 0
    for row in rows:
        row = dict(row)
        status_counts[str(row.get("result_status") or "UNKNOWN")] += 1
        payload = _paper_parse_diagnostic_payload(row.get("last_candles_json"))
        decision = payload.get("_v2419_decision_audit")
        m1 = payload.get("1min") or {}
        pdiag = m1.get("provider_diagnostics") or {}
        refresh = m1.get("m1_refresh") or {}
        if isinstance(pdiag, dict) and pdiag.get("diagnosis"):
            provider_diagnoses[str(pdiag["diagnosis"])] += 1
        if isinstance(refresh, dict) and refresh.get("stop_reason"):
            refresh_results[str(refresh["stop_reason"])] += 1
        if not isinstance(decision, dict):
            legacy += 1
            continue
        analyzed += 1
        fresh = row.get("freshness_status") == "FRESH"
        age = row.get("signal_age_seconds")
        if not fresh:
            stale_analyzed += 1
            if isinstance(age, (float, int)):
                stale_ages.append(float(age))
            continue
        fresh_analyzed += 1
        if isinstance(age, (float, int)):
            fresh_ages.append(float(age))
        primary[str(decision.get("primary_blocker") or "UNKNOWN")] += 1
        raw_signals[str(decision.get("raw_signal") or "UNKNOWN")] += 1
        for tf in decision.get("wait_timeframes") or []:
            if tf in ("H1", "M15", "M5", "M1"):
                tf_wait[tf] += 1
        h1 = decision.get("h1") or {}
        if (decision.get("per_timeframe") or {}).get("H1", {}).get("signal") == "WAIT":
            h1_reasons[str(h1.get("block_reason") or "UNKNOWN")] += 1
        m15 = decision.get("m15") or {}
        if (decision.get("per_timeframe") or {}).get("M15", {}).get("signal") == "WAIT":
            m15_codes[str(m15.get("reason_code") or "UNKNOWN")] += 1
        m5 = decision.get("m5") or {}
        if (decision.get("per_timeframe") or {}).get("M5", {}).get("signal") == "WAIT":
            m5_reasons[str((decision.get("per_timeframe") or {}).get("M5", {}).get("reason") or "UNKNOWN")] += 1
            if m5.get("fvg_reason"):
                m5_fvg_reasons[str(m5["fvg_reason"])] += 1
            if m5.get("ob_reason"):
                m5_ob_reasons[str(m5["ob_reason"])] += 1
        m1 = decision.get("m1") or {}
        if (decision.get("per_timeframe") or {}).get("M1", {}).get("signal") == "WAIT":
            for reason_code in m1.get("failed_checks") or []:
                m1_failed_checks[str(reason_code)] += 1
        m1_signals[str((decision.get("per_timeframe") or {}).get("M1", {}).get("signal") or "UNKNOWN")] += 1
    mean = lambda ages: round(sum(ages) / len(ages), 2) if ages else None
    return {
        "window_hours": hours,
        "scans_examined": len(rows),
        "truncated_to_safety_limit": bool(truncated),
        "status_counts": dict(status_counts),
        "diagnostic_profiles_available": analyzed,
        "legacy_or_unanalyzed_scans": legacy,
        "fresh_analyzed_scans": fresh_analyzed,
        "stale_analyzed_scans": stale_analyzed,
        "fresh_percentage_of_analyzed": round(100 * fresh_analyzed / analyzed, 1) if analyzed else None,
        "fresh_m1_avg_age_seconds": mean(fresh_ages),
        "stale_m1_avg_age_seconds": mean(stale_ages),
        "primary_blockers_fresh_only": dict(primary),
        "wait_timeframes_fresh_only": dict(tf_wait),
        "h1_wait_reasons_fresh_only": dict(h1_reasons),
        "m15_wait_reason_codes_fresh_only": dict(m15_codes),
        "m5_wait_reasons_fresh_only": dict(m5_reasons),
        "m5_fvg_reasons_fresh_only": dict(m5_fvg_reasons),
        "m5_order_block_reasons_fresh_only": dict(m5_ob_reasons),
        "m1_failed_checks_fresh_only": dict(m1_failed_checks),
        "m1_signals_fresh_only": dict(m1_signals),
        "raw_signals_fresh_only": dict(raw_signals),
        "provider_m1_diagnoses_all_recent_scans": dict(provider_diagnoses),
        "m1_refresh_stop_reasons_all_recent_scans": dict(refresh_results),
        "interpretation": "Blocker stats count fresh analyzed scans only; one scan may have multiple WAIT timeframes. Historical rows without v2.4.19 metadata are excluded. No brokerage or profitability claim.",
    }


def _paper_query_diagnostic_summary(hours=24):
    _paper_initialize()
    cutoff = _paper_iso(_paper_now() - pd.Timedelta(hours=hours))
    cap = 5000
    with _paper_db() as conn:
        rows = _paper_exec(conn, """
            SELECT result_status, strategy_version, signal_age_seconds,
                   freshness_status, last_candles_json
            FROM paper_scan_runs WHERE scanned_at_utc >= ?
            ORDER BY scanned_at_utc DESC LIMIT ?
        """, (cutoff, cap+1)).fetchall()
    return _paper_diagnostic_summary(rows[:cap], hours, truncated=len(rows)>cap)


# ==================================================
# API: SIGNAL
# ==================================================

@app.post("/api/signal")
def get_signal(req: SignalRequest):
    """No overlapping live scans inside this process; fail fast when quota is low."""
    if not _TD_LIVE_SCAN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=429, detail={
            "code": "SCAN_BUSY",
            "message": "Another live scan is already running; duplicate request rejected without spending API credits.",
            "retry_after_seconds": 5,
        }, headers={"Retry-After": "5"})
    try:
        _td_claim_credits(TD_LIVE_SCAN_COST, reserve_only=True)
        return _get_signal_core(req)
    finally:
        _TD_LIVE_SCAN_LOCK.release()


def _get_signal_core(req: SignalRequest):
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

    # All four downloads share the same snapshot cutoff even if Twelve Data
    # requests cross a minute/5-minute boundary while they are in flight.
    analysis_at_utc = pd.Timestamp.now(tz="UTC")
    # M1 is the only timeframe with a 30-second freshness deadline. Request it
    # before H1/M15/M5; the latter retain their original scoring semantics.
    # Do not refresh another as-of timestamp: this preserves candle safety.
    market_data = {
        tf: get_market_data(pair, tf, as_of_utc=analysis_at_utc)
        for tf in LIVE_FETCH_TIMEFRAMES
    }
    initial_m1_diagnostics = dict(
        market_data["1min"].attrs.get("live_provider_diagnostics") or {}
    )
    retry_audit = _refresh_missing_m1_bar(pair, market_data, analysis_at_utc)
    if retry_audit.get("stop_reason") == "API_RATE_LIMITED":
        raise _td_budget_error(
            retry_audit.get("rate_limit_reason", "LOCAL_MINUTE_BUDGET"),
            m1_diagnostics={
                "provider_diagnostics": market_data["1min"].attrs.get("live_provider_diagnostics"),
                "m1_refresh": retry_audit,
            },
        )

    result = analyze_multi_timeframe(market_data)
    observed_at_utc = pd.Timestamp.now(tz="UTC")
    freshness = _live_data_freshness(market_data, analysis_at_utc, observed_at_utc)
    # The raw source bar and the filtered closed bar are both reported.
    # With no creds in output, this is safe for Swagger and scan monitoring.
    for tf in ANALYSIS_TIMEFRAMES:
        diagnostics = market_data[tf].attrs.get("live_provider_diagnostics")
        if diagnostics:
            freshness["candles"][tf]["provider_diagnostics"] = diagnostics
    freshness["m1_refresh"] = retry_audit
    freshness.setdefault("candles", {}).setdefault("1min", {})["m1_refresh"] = retry_audit
    final_m1_diagnostics = (
        market_data["1min"].attrs.get("live_provider_diagnostics") or {}
    )
    live_pipeline = _live_fetch_pipeline_diagnostics(
        analysis_at_utc, observed_at_utc, market_data,
        initial_m1_diagnostics, final_m1_diagnostics, retry_audit,
    )
    raw_signal = result["signal"]
    raw_reason = result["reason"]
    if freshness["status"] != "FRESH":
        # Diagnostics and scores are preserved but an old signal must never
        # be presented as an executable, newly observed CALL or PUT.
        result["signal"] = "WAIT"
        result["reason"] = "Ожидание: данные M1 устарели или свеча ещё не закрылась"

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

    # Do NOT create backdated virtual fills at the prior candle's close.
    # The authenticated paper workflow records an observed signal; the user
    # must record an actual DEMO fill and exit separately to assess outcomes.
    virtual_trade = None

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
        "virtual_trade": virtual_trade,
        "virtual_trade_note": "No retrospective entry/fill from historical candle close; use authenticated paper trade entry for DEMO fills.",
        "raw_signal_before_freshness_gate": raw_signal,
        "raw_reason_before_freshness_gate": raw_reason,
        "freshness": freshness,
        "live_pipeline": live_pipeline,
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
# NORMAL FOREX SESSION AND WEEKLY HISTORY
# ==================================================

def _regular_forex_candle_mask(df, timeframe):
    """True only when the WHOLE candle is in the normal Forex session.

    The usual convention is Sunday 17:00 to Friday 17:00 New York time.
    The IANA timezone handles Eastern daylight saving automatically.
    Source timestamps represent candle opens, not closes.
    """
    if df.empty:
        return np.zeros(0, dtype=bool)
    start_ny = df.index.tz_convert(FX_MARKET_TIMEZONE)
    end_ny = (df.index + pd.Timedelta(TIMEFRAMES[timeframe])).tz_convert(
        FX_MARKET_TIMEZONE
    )
    opening_minute = 17 * 60

    def in_session(idx, is_close):
        weekday = idx.dayofweek
        minutes = idx.hour * 60 + idx.minute + idx.second / 60.0
        friday_ok = (minutes <= opening_minute) if is_close else (minutes < opening_minute)
        return (
            (weekday >= 0) & (weekday <= 3)
        ) | (
            (weekday == 4) & friday_ok
        ) | (
            (weekday == 6) & (minutes >= opening_minute)
        )

    return np.asarray(in_session(start_ny, False) & in_session(end_ny, True), dtype=bool)


def _regular_forex_filter(df, timeframe):
    return df.loc[_regular_forex_candle_mask(df, timeframe)].copy()


def _last_completed_friday_anchors(number_of_weeks, now=None):
    """Fridays at 17:00 New York wall clock, latest fully completed first."""
    now_ny = (datetime.now(FX_MARKET_TIMEZONE) if now is None else
              pd.Timestamp(now).tz_convert(FX_MARKET_TIMEZONE).to_pydatetime())
    friday_date = now_ny.date() - timedelta(days=(now_ny.weekday() - 4) % 7)
    this_friday_end = datetime.combine(friday_date, time(17), FX_MARKET_TIMEZONE)
    if this_friday_end > now_ny:
        friday_date -= timedelta(days=7)
    return [
        pd.Timestamp(datetime.combine(
            friday_date - timedelta(days=7 * i), time(17), FX_MARKET_TIMEZONE
        ))
        for i in range(number_of_weeks)
    ]


def _payout_analysis(wins, losses, draws):
    closed = wins + losses + draws
    decided = wins + losses
    result = {}
    for rate in PAYOUT_RATES:
        net_units = wins * rate - losses
        result[str(int(rate * 100))] = {
            "payout_percent": int(rate * 100),
            "net_units_per_1_unit_stake": round(net_units, 4),
            "return_on_total_stakes_percent": (
                round(net_units * 100 / closed, 2) if closed else None
            ),
            "break_even_win_rate_excluding_draws_percent": round(
                100 / (1 + rate), 2
            ),
        }
    return {
        "payout_rates": result,
        "win_rate_excluding_draws_percent": (
            round(100 * wins / decided, 2) if decided else None
        ),
        "draw_policy": "RETURN_STAKE_ZERO_PROFIT",
        "stake_unit": 1,
        "closed_trades": int(closed),
        "notes": (
            "Модель бинарных опционов: WIN +ставка×выплата, LOSS -ставка, "
            "DRAW возврат ставки. Не учитывает реальное исполнение брокера."
        ),
    }


def _expiry_outcome(signal, entry_price, exit_price):
    diff = exit_price - entry_price
    if diff == 0:
        return "DRAW"
    if (signal == "CALL" and diff > 0) or (signal == "PUT" and diff < 0):
        return "WIN"
    return "LOSS"


def _settle_signal_events_for_expiry(events, m1, expiry_timeframe):
    """Compare expiries on the SAME entry signals and SAME M1 dataset."""
    wins = losses = draws = unresolved = 0
    for event in events:
        expiry = event["entry_time"] + TIMEFRAMES[expiry_timeframe]
        data = _m1_candle_at_expiry(m1, expiry)
        if data is None:
            unresolved += 1
            continue
        outcome = _expiry_outcome(
            event["signal"], event["entry_price"], data["exit_price"]
        )
        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSS":
            losses += 1
        else:
            draws += 1
    closed = wins + losses + draws
    return {
        "expiry_timeframe": expiry_timeframe,
        "signals_generated": len(events),
        "closed_trades": closed,
        "unresolved_expiry_candles": unresolved,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate_percent": round(100 * wins / closed, 2) if closed else None,
        "payout_analysis": _payout_analysis(wins, losses, draws),
    }

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


def run_historical_backtest(
    pair, expiry_timeframe, candle_count,
    market_mode="regular_forex", history_override=None,
    collect_signal_events=False,
):
    if market_mode not in ("regular_forex", "all_data"):
        raise HTTPException(status_code=400, detail="Недопустимый market_mode")

    raw_history = (history_override if history_override is not None else {
        tf: get_market_data(pair, tf, outputsize=candle_count)
        for tf in ANALYSIS_TIMEFRAMES
    })
    raw_counts = {tf: int(len(raw_history[tf])) for tf in ANALYSIS_TIMEFRAMES}
    history = {
        tf: (_regular_forex_filter(raw_history[tf], tf)
             if market_mode == "regular_forex" else raw_history[tf].copy())
        for tf in ANALYSIS_TIMEFRAMES
    }
    if any(len(df) == 0 for df in history.values()):
        raise HTTPException(status_code=422, detail="Нет свечей внутри выбранной торговой сессии")
    excluded_counts = {
        tf: raw_counts[tf] - len(history[tf]) for tf in ANALYSIS_TIMEFRAMES
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
    signal_events = [] if collect_signal_events else None

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
        if signal_events is not None:
            signal_events.append({
                "signal": result["signal"],
                "entry_price": entry_price,
                "entry_time": entry_time,
            })
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

    output = {
        "strategy_version": VERSION,
        "pair": pair,
        "expiry_timeframe": expiry_timeframe,
        "market_mode": market_mode,
        "session_filter": {
            "timezone": "America/New_York",
            "hours": "Sunday 17:00 through Friday 17:00 ET",
            "whole_candle_must_fit_session": True,
            "raw_candles": raw_counts,
            "excluded_candles": {k: int(v) for k, v in excluded_counts.items()},
            "retained_candles": {tf: int(len(history[tf])) for tf in ANALYSIS_TIMEFRAMES},
        },
        "payout_analysis": _payout_analysis(wins, losses, draws),
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
    if signal_events is not None:
        output["_signal_events_for_comparison"] = signal_events
    return output


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
        candle_count=req.candles,
        market_mode=req.market_mode,
    )

    return make_json_safe(result)


@app.post("/api/backtest/compare")
def compare_expiry_across_weeks(req: BacktestCompareRequest):
    """Compare 1m and 5m settlement from identical signals in separate weeks.

    One Twelve Data request per TF per week. Historical windows are anchored
    at the LAST fully completed Friday 17:00 ET and at previous Fridays.
    The window uses up to candles_per_week M1 source candles: it is a SAMPLE,
    not necessarily a complete trading week.
    """
    pair = req.pair.strip().upper()
    if pair not in ALLOWED_PAIRS:
        raise HTTPException(status_code=400, detail="Эта валютная пара не поддерживается")

    weekly = []
    failures = []
    total = {expiry: {"signals_generated": 0, "closed_trades": 0,
                      "unresolved_expiry_candles": 0,
                      "wins": 0, "losses": 0, "draws": 0}
             for expiry in ("1min", "5min")}

    for week_end in _last_completed_friday_anchors(req.weeks):
        try:
            input_history = {
                tf: get_market_data(
                    pair, tf, outputsize=req.candles_per_week,
                    end_date=week_end.tz_convert("UTC"),
                )
                for tf in ANALYSIS_TIMEFRAMES
            }
            report = run_historical_backtest(
                pair=pair,
                expiry_timeframe="1min",
                candle_count=req.candles_per_week,
                market_mode="regular_forex",
                history_override=input_history,
                collect_signal_events=True,
            )
            events = report.pop("_signal_events_for_comparison")
            clean_m1 = _regular_forex_filter(input_history["1min"], "1min")

            one_min = {
                "expiry_timeframe": "1min",
                "signals_generated": report["signals_generated"],
                "closed_trades": report["closed_trades"],
                "unresolved_expiry_candles": report["unresolved_expiry_candles"],
                "wins": report["wins"], "losses": report["losses"],
                "draws": report["draws"],
                "win_rate_percent": report["win_rate_percent"],
                "payout_analysis": report["payout_analysis"],
            }
            five_min = _settle_signal_events_for_expiry(
                events, clean_m1, "5min"
            )
            for expiry, result in (("1min", one_min), ("5min", five_min)):
                for metric in total[expiry]:
                    total[expiry][metric] += int(result[metric])

            weekly.append({
                "week_end_new_york": week_end.isoformat(),
                "week_end_utc": week_end.tz_convert("UTC").isoformat(),
                "evaluated_points": report["evaluated_points"],
                "skipped_points_insufficient_history": report[
                    "skipped_points_insufficient_history"
                ],
                "session_filter": report["session_filter"],
                "data_audit": {
                    "status": report["backtest_data_audit"]["status"],
                    "warnings": report["backtest_data_audit"]["warnings"],
                    "future_candle_violations": report["backtest_data_audit"][
                        "future_candle_violations"
                    ],
                    "m1_window_alignment_violations": report["backtest_data_audit"][
                        "m1_window_alignment_violations"
                    ],
                },
                "expiry_results": {"1min": one_min, "5min": five_min},
            })
        except HTTPException as exc:
            if isinstance(exc.detail, dict) and exc.detail.get("error") == "BACKTEST_LOOKAHEAD_BLOCKED":
                # A chronology violation invalidates the entire comparison.
                raise
            failures.append({
                "week_end_new_york": week_end.isoformat(),
                "error": make_json_safe(exc.detail),
                "status_code": exc.status_code,
            })

    if not weekly:
        raise HTTPException(
            status_code=502,
            detail={"error": "NO_VALID_WEEKS", "failures": failures,
                    "hint": "Проверьте лимит Twelve Data и доступность истории"},
        )

    overall = {}
    for expiry, stats in total.items():
        wins, losses, draws = stats["wins"], stats["losses"], stats["draws"]
        overall[expiry] = {
            **stats,
            "win_rate_percent": (
                round(100 * wins / stats["closed_trades"], 2)
                if stats["closed_trades"] else None
            ),
            "payout_analysis": _payout_analysis(wins, losses, draws),
        }

    return make_json_safe({
        "strategy_version": VERSION,
        "pair": pair,
        "market_mode": "regular_forex",
        "requested_weeks": req.weeks,
        "completed_weeks": len(weekly),
        "partial": bool(failures),
        "candles_per_week_requested": req.candles_per_week,
        "week_sampling": "Отдельные исторические окна перед пятничным закрытием; не вся неделя",
        "rates_percent": [70, 80, 90],
        "weeks": weekly,
        "failures": failures,
        "overall": overall,
        "limitations": [
            "До четырёх запросов Twelve Data на неделю; тариф и доступная история могут ограничивать результат.",
            "Свечи внутри обычной сессии Forex (America/New_York), не котировки OTC.",
            "Выборка ограничена candles_per_week свечами M1 на неделю; это не полные пять торговых дней.",
            "Обе экспирации проверены на ОДНИХ сигналах и одной M1 истории для каждого окна.",
            "Свечи выходных отбрасываются до расчётов индикаторов; результат не совместим с OTC котировками брокера.",
            "Не учитываются спред, задержки исполнения и реальные выплаты; ничья означает возврат ставки.",
            "Исторические сигналы соседних M1 свечей зависимы, выборка может быть малой.",
        ],
    })


# ==================================================
# ROBUSTNESS BACKTEST: EXECUTION PROXIES AND COOLDOWN
# ==================================================

def _pip_size_for_pair(pair):
    """Conventional pip sizes (0.01 for JPY-quote pairs, else 0.0001)."""
    return 0.01 if pair.upper().endswith("/JPY") else 0.0001


def _select_cooldown_events(events, cooldown_minutes):
    """Select chronologically separated entries; no strategy rule is changed.

    The original raw events are still included in the report. A cooldown is a
    reporting variant only, not a new /api/signal condition.
    """
    ordered = sorted(events, key=lambda e: pd.Timestamp(e["entry_time"]))
    accepted = []
    previous = None
    spacing = pd.Timedelta(minutes=cooldown_minutes)
    for event in ordered:
        timestamp = pd.Timestamp(event["entry_time"])
        if timestamp.tzinfo is None:
            raise ValueError("Robustness event timestamps must be timezone-aware")
        if previous is None or timestamp - previous >= spacing:
            accepted.append(event)
            previous = timestamp
    return accepted


def _simulate_execution(events, m1, pair, expiry_timeframe,
                        delay_minutes=0, adverse_slippage_pips=0.0,
                        collect_hours=False):
    """Evaluate an explicitly labeled *M1 candle-open execution proxy*.

    No tick data exists, so no sub-minute latency or actual fill prices are
    inferred. The original signal appears after a completed M1 candle (time T).
    Entry is the M1 candle OPEN at T + delay_minutes, if present. Exit uses the
    M1 close ending at entry time + requested expiry. An unfavorable pip offset
    is applied to entry alone for sensitivity testing. Weekend gaps or missing
    expiry candles produce UNRESOLVED, never invented WIN/LOSS results.
    """
    if expiry_timeframe not in ("1min", "5min"):
        raise ValueError("Unsupported expiry timeframe")
    if delay_minutes not in (0, 1):
        raise ValueError("Execution delay must be 0 or 1 whole minute")
    if adverse_slippage_pips not in (0.0, 0.2, 0.5):
        raise ValueError("Unsupported slippage scenario")
    pip = _pip_size_for_pair(pair)
    counts = {"signals_generated": int(len(events)), "closed_trades": 0,
              "unresolved_entry_candles": 0, "unresolved_expiry_candles": 0,
              "wins": 0, "losses": 0, "draws": 0}
    hour_counts = {}
    gap_pips = []
    gap_directional_pips = []
    for event in events:
        signal = event["signal"]
        if signal not in ("CALL", "PUT"):
            raise ValueError("Invalid historical signal direction")
        signal_time = pd.Timestamp(event["entry_time"])
        if signal_time.tzinfo is None:
            raise ValueError("Robustness event time must be timezone-aware")
        entry_time = signal_time + pd.Timedelta(minutes=delay_minutes)
        # An entry candle must actually exist in the filtered regular-Forex M1.
        if entry_time not in m1.index:
            counts["unresolved_entry_candles"] += 1
            continue
        entry_row = m1.loc[entry_time]
        if isinstance(entry_row, pd.DataFrame):
            raise ValueError("Duplicate entry candle; cannot evaluate fill")
        entry_open = float(entry_row["open"])
        reference_close = float(event["entry_price"])
        if delay_minutes == 0:
            gap = (entry_open - reference_close) / pip
            gap_pips.append(gap)
            gap_directional_pips.append(gap if signal == "CALL" else -gap)

        if signal == "CALL":
            simulated_entry = entry_open + adverse_slippage_pips * pip
        else:
            simulated_entry = entry_open - adverse_slippage_pips * pip
        if simulated_entry <= 0:
            counts["unresolved_entry_candles"] += 1
            continue
        expiry_at = entry_time + TIMEFRAMES[expiry_timeframe]
        exit_info = _m1_candle_at_expiry(m1, expiry_at)
        if exit_info is None:
            counts["unresolved_expiry_candles"] += 1
            continue
        outcome = _expiry_outcome(signal, simulated_entry, exit_info["exit_price"])
        counts[{"WIN":"wins", "LOSS":"losses", "DRAW":"draws"}[outcome]] += 1
        counts["closed_trades"] += 1
        if collect_hours:
            hour_key = entry_time.tz_convert(FX_MARKET_TIMEZONE).strftime("%H:00")
            hour = hour_counts.setdefault(hour_key, {"wins": 0, "losses": 0, "draws": 0})
            hour[{"WIN":"wins", "LOSS":"losses", "DRAW":"draws"}[outcome]] += 1

    wins, losses, draws = (counts[k] for k in ("wins", "losses", "draws"))
    closed = counts["closed_trades"]
    result = {
        **counts,
        "win_rate_percent": round(wins * 100 / closed, 2) if closed else None,
        "payout_analysis": _payout_analysis(wins, losses, draws),
        "entry_basis": "M1_OPEN_AT_SIGNAL_CLOSE_PLUS_WHOLE_MINUTE_DELAY",
        "expiry_basis": "M1_CLOSE_EXACTLY_AT_EXPIRY",
        "delay_minutes": delay_minutes,
        "adverse_slippage_pips": adverse_slippage_pips,
    }
    if collect_hours:
        result["by_entry_hour_new_york"] = {
            hour: {
                **c,
                "trades": sum(c.values()),
                "win_rate_excluding_draws_percent": (
                    round(100 * c["wins"] / (c["wins"] + c["losses"]), 2)
                    if c["wins"] + c["losses"] else None
                ),
                "net_units_80pct_per_unit_stake": round(c["wins"] * 0.80 - c["losses"], 4),
            }
            for hour, c in sorted(hour_counts.items())
        }
        result["signal_close_to_next_open_gap_pips"] = {
            "observations": len(gap_pips),
            "mean_absolute": round(float(np.mean(np.abs(gap_pips))), 4) if gap_pips else None,
            "max_absolute": round(float(np.max(np.abs(gap_pips))), 4) if gap_pips else None,
            "mean_adverse_directional": round(float(np.mean(gap_directional_pips)), 4)
                if gap_directional_pips else None,
        }
    return result


def _robustness_result_summary(result):
    """Compact snapshot for each weekly scenario."""
    return {
        k: result[k] for k in (
            "signals_generated", "closed_trades", "unresolved_entry_candles",
            "unresolved_expiry_candles", "wins", "losses", "draws", "win_rate_percent",
            "payout_analysis"
        )
    }


def _combine_robustness_reports(reports):
    totals = {
        "signals_generated": 0, "closed_trades": 0,
        "unresolved_entry_candles": 0, "unresolved_expiry_candles": 0,
        "wins": 0, "losses": 0, "draws": 0,
    }
    for report in reports:
        for k in totals:
            totals[k] += int(report[k])
    wins, losses, draws = totals["wins"], totals["losses"], totals["draws"]
    return {
        **totals,
        "win_rate_percent": (
            round(100 * wins / totals["closed_trades"], 2)
            if totals["closed_trades"] else None
        ),
        "payout_analysis": _payout_analysis(wins, losses, draws),
    }


# ==================================================
# PRICE/TIME AUDIT: NO TRADING LOGIC CHANGES
# ==================================================

def _pip_summary(values):
    """Distribution of REAL adjacent M1 opening/closing gaps in pips."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "observations": 0, "mean_pips": None, "median_pips": None,
            "mean_absolute_pips": None, "median_absolute_pips": None,
            "p90_absolute_pips": None, "p99_absolute_pips": None,
            "maximum_absolute_pips": None, "positive": 0, "negative": 0,
            "zero": 0, "absolute_at_least_half_pip": 0,
            "absolute_at_least_one_pip": 0,
        }
    abs_vals = np.abs(arr)
    epsilon = 1e-8
    return {
        "observations": int(arr.size),
        "mean_pips": round(float(np.mean(arr)), 5),
        "median_pips": round(float(np.median(arr)), 5),
        "mean_absolute_pips": round(float(np.mean(abs_vals)), 5),
        "median_absolute_pips": round(float(np.median(abs_vals)), 5),
        "p90_absolute_pips": round(float(np.percentile(abs_vals, 90)), 5),
        "p99_absolute_pips": round(float(np.percentile(abs_vals, 99)), 5),
        "maximum_absolute_pips": round(float(np.max(abs_vals)), 5),
        "positive": int(np.sum(arr > epsilon)),
        "negative": int(np.sum(arr < -epsilon)),
        "zero": int(np.sum(np.abs(arr) <= epsilon)),
        "absolute_at_least_half_pip": int(np.sum(abs_vals + epsilon >= 0.5)),
        "absolute_at_least_one_pip": int(np.sum(abs_vals + epsilon >= 1.0)),
    }


def _audit_execution_price_gaps(events, m1, pair, expiry_timeframe):
    """Compare signal-close to next-M1-open, keeping SAME exit timestamp.

    This does NOT simulate tick-level fills and does NOT alter outcomes used
    anywhere else in the bot. It is deliberately a diagnostic-only function.
    """
    if expiry_timeframe not in ("1min", "5min"):
        raise ValueError("Unsupported audit expiry")
    if not m1.index.is_unique or not m1.index.is_monotonic_increasing:
        raise HTTPException(
            status_code=422,
            detail="PRICE_AUDIT_REQUIRES_SORTED_UNIQUE_M1_CANDLES",
        )
    pip = _pip_size_for_pair(pair)
    step = pd.Timedelta(minutes=1)
    tolerance = max(pip * 1e-5, 1e-12)
    close = m1["close"].to_numpy(dtype=float)
    opening = m1["open"].to_numpy(dtype=float)
    # IMPORTANT: .asi8 uses the index's underlying datetime unit (ns/us/ms/s),
    # whereas Timedelta.value always uses nanoseconds. Comparing the raw ints
    # incorrectly produced ZERO adjacent candles on some pandas versions.
    # Compare actual timestamps directly so this works for any datetime unit.
    valid = np.asarray((m1.index[1:] - m1.index[:-1]) == step, dtype=bool)
    # Only CONSECUTIVE fully closed M1 candles, never crossing weekends/gaps.
    adjacent_gaps = ((opening[1:] - close[:-1]) / pip)[valid].tolist()
    adjacent_pairs = int(np.count_nonzero(valid))
    baseline_coverage = {
        "m1_candles": int(len(m1)),
        "candidate_pairs": max(0, int(len(m1)) - 1),
        "one_minute_adjacent_pairs": adjacent_pairs,
        "nonadjacent_pairs": max(0, int(len(m1)) - 1 - adjacent_pairs),
        "datetime_index_dtype": str(m1.index.dtype),
        "comparison_method": "timestamp_timedelta_equality_unit_independent",
        "sufficient_for_baseline": bool(adjacent_pairs >= 100),
    }
    rows = []
    all_gaps = []
    adverse_gaps = []
    consistency = {
        "signals_seen": int(len(events)),
        "verified_signal_closes": 0,
        "reference_close_mismatches": 0,
        "missing_signal_candle": 0,
        "missing_next_open_candle": 0,
        "missing_expiry_candle": 0,
        "non_finite_price": 0,
        "nonpositive_prices": 0,
        "invalid_time_order": 0,
    }
    matrix = {
        "WIN": {"WIN": 0, "LOSS": 0, "DRAW": 0},
        "LOSS": {"WIN": 0, "LOSS": 0, "DRAW": 0},
        "DRAW": {"WIN": 0, "LOSS": 0, "DRAW": 0},
    }
    for event in sorted(events, key=lambda v: pd.Timestamp(v["entry_time"])):
        side = event["signal"]
        if side not in ("CALL", "PUT"):
            raise ValueError("Expected CALL/PUT signal event")
        signal_time = pd.Timestamp(event["entry_time"])
        if signal_time.tzinfo is None:
            raise ValueError("Historical signal timestamp is naive")
        candle_open = signal_time - step
        if candle_open not in m1.index:
            consistency["missing_signal_candle"] += 1
            continue
        if signal_time not in m1.index:
            consistency["missing_next_open_candle"] += 1
            continue
        source_close = float(m1.at[candle_open, "close"])
        event_close = float(event["entry_price"])
        next_open = float(m1.at[signal_time, "open"])
        if not all(math.isfinite(v) for v in (source_close, event_close, next_open)):
            consistency["non_finite_price"] += 1
            continue
        if min(source_close, event_close, next_open) <= 0:
            consistency["nonpositive_prices"] += 1
            continue
        if abs(source_close - event_close) > tolerance:
            consistency["reference_close_mismatches"] += 1
            # A mismatch is a test-integrity failure, not a trade to use.
            continue
        consistency["verified_signal_closes"] += 1
        expiry_at = signal_time + TIMEFRAMES[expiry_timeframe]
        if not candle_open < signal_time < expiry_at:
            consistency["invalid_time_order"] += 1
            continue
        exit_info = _m1_candle_at_expiry(m1, expiry_at)
        if exit_info is None:
            consistency["missing_expiry_candle"] += 1
            continue
        exit_price = float(exit_info["exit_price"])
        if not math.isfinite(exit_price) or exit_price <= 0:
            consistency["non_finite_price"] += 1
            continue
        gap = (next_open - source_close) / pip
        # POSITIVE is adverse to CALL when open rises, adverse to PUT when open falls.
        adverse = gap if side == "CALL" else -gap
        old_outcome = _expiry_outcome(side, event_close, exit_price)
        next_outcome = _expiry_outcome(side, next_open, exit_price)
        matrix[old_outcome][next_outcome] += 1
        all_gaps.append(float(gap))
        adverse_gaps.append(float(adverse))
        rows.append({
            "side": side,
            "signal_candle_open_utc": candle_open.isoformat(),
            "signal_available_at_utc": signal_time.isoformat(),
            "next_m1_open_utc": signal_time.isoformat(),
            "expiry_utc": expiry_at.isoformat(),
            "exit_candle_open_utc": exit_info["exit_candle_open_time"],
            "signal_candle_close": round(source_close, 7),
            "next_candle_open": round(next_open, 7),
            "expiry_close": round(exit_price, 7),
            "raw_gap_pips": round(float(gap), 5),
            "adverse_directional_gap_pips": round(float(adverse), 5),
            "close_entry_outcome": old_outcome,
            "next_open_outcome": next_outcome,
            "outcome_changed": old_outcome != next_outcome,
        })
    positive_adverse = sum(a > 1e-8 for a in adverse_gaps)
    favorable = sum(a < -1e-8 for a in adverse_gaps)
    neutral = len(adverse_gaps) - positive_adverse - favorable
    changes = sum(r["outcome_changed"] for r in rows)
    alerts = []
    if any(consistency[k] for k in (
        "reference_close_mismatches", "missing_signal_candle",
        "missing_next_open_candle", "invalid_time_order", "non_finite_price",
        "nonpositive_prices",
    )):
        alerts.append("EVENT_PRICE_OR_TIME_MISMATCH")
    if len(adverse_gaps) >= 8 and float(np.mean(adverse_gaps)) <= -0.25:
        alerts.append("SYSTEMATIC_FAVORABLE_NEXT_OPEN_FOR_SIGNAL_DIRECTION")
    if len(adverse_gaps) >= 8 and favorable / len(adverse_gaps) >= 0.80:
        alerts.append("UNUSUAL_FAVORABLE_GAP_FRACTION_AT_SIGNALS")
    if len(adjacent_gaps) < 100:
        alerts.append("INSUFFICIENT_ADJACENT_M1_BASELINE")
    if len(adjacent_gaps) >= 100 and float(np.mean(np.abs(adjacent_gaps))) >= 0.30:
        alerts.append("MATERIAL_GAPS_COMMON_IN_UNFILTERED_M1_DATA")
    if len(all_gaps) >= 8 and float(np.mean(np.abs(all_gaps))) >= 0.50:
        alerts.append("MATERIAL_GAPS_AT_SIGNAL_TIMES")
    if len(rows) < len(events):
        alerts.append("UNRESOLVED_OR_UNVERIFIED_AUDIT_EVENTS")
    # Keep output short enough to paste into chat; make samples deterministic.
    selected = set(range(min(4, len(rows))))
    selected.update(sorted(range(len(rows)), key=lambda i: abs(rows[i]["raw_gap_pips"]),
                           reverse=True)[:5])
    selected.update(i for i, row in enumerate(rows) if row["outcome_changed"])
    samples = [rows[i] for i in sorted(selected)[:25]]
    return {
        "status": "REVIEW_REQUIRED" if alerts else "CHECKED",
        "warnings": alerts,
        "price_basis": "Twelve Data M1 candle close/open, not broker bid/ask or tick fills",
        "pip_size": pip,
        "signal_close_to_next_open": _pip_summary(all_gaps),
        "adverse_gap_by_trade_direction": {
            **_pip_summary(adverse_gaps),
            "favorable": favorable,
            "adverse": positive_adverse,
            "neutral": neutral,
            "by_direction": {
                side: {
                    **_pip_summary([r["raw_gap_pips"] for r in rows if r["side"] == side]),
                    "favorable": sum(r["adverse_directional_gap_pips"] < -1e-8 for r in rows if r["side"] == side),
                    "adverse": sum(r["adverse_directional_gap_pips"] > 1e-8 for r in rows if r["side"] == side),
                }
                for side in ("CALL", "PUT")
            },
        },
        "all_adjacent_m1_close_to_next_open_gaps": _pip_summary(adjacent_gaps),
        "baseline_coverage": baseline_coverage,
        "gap_comparison": {
            "mean_absolute_signal_gap_pips": round(float(np.mean(np.abs(all_gaps))), 5) if all_gaps else None,
            "mean_absolute_baseline_gap_pips": round(float(np.mean(np.abs(adjacent_gaps))), 5) if adjacent_gaps else None,
            "signal_vs_baseline_mean_abs_ratio": round(
                float(np.mean(np.abs(all_gaps)) / np.mean(np.abs(adjacent_gaps))), 3
            ) if all_gaps and adjacent_gaps and float(np.mean(np.abs(adjacent_gaps))) > 1e-9 else None,
            "baseline_positive_gap_fraction_percent": round(
                100 * sum(g > 1e-8 for g in adjacent_gaps) / len(adjacent_gaps), 2
            ) if adjacent_gaps else None,
        },
        "outcome_effect_same_expiry": {
            "compared_settled_trades": len(rows),
            "outcome_changed_count": changes,
            "outcome_transition_matrix": matrix,
            "close_entry": {
                k: sum(row["close_entry_outcome"] == k for row in rows)
                for k in ("WIN", "LOSS", "DRAW")
            },
            "next_open_entry": {
                k: sum(row["next_open_outcome"] == k for row in rows)
                for k in ("WIN", "LOSS", "DRAW")
            },
            "expiry_time_changed": False,
        },
        "data_consistency": consistency,
        "sample_trades": samples,
        "_private_all_gaps": all_gaps,
        "_private_adverse_gaps": adverse_gaps,
        "_private_adjacent_gaps": adjacent_gaps,
        "_private_rows": rows,
    }


@app.post("/api/backtest/execution-audit")
def backtest_execution_price_audit(req: BacktestRobustnessRequest):
    """Investigate suspicious close-to-open gaps without modifying /api/signal.

    Uses the same independent weekly windows and historic signal generation as
    /api/backtest/robustness; returns detailed audit instead of large P&L data.
    """
    pair = req.pair.strip().upper()
    if pair not in ALLOWED_PAIRS:
        raise HTTPException(status_code=400, detail="Эта валютная пара не поддерживается")
    timeframe = req.expiry_timeframe.strip()
    if timeframe not in ("1min", "5min"):
        raise HTTPException(status_code=400, detail="expiry_timeframe должен быть 1min или 5min")
    weekly = []
    failures = []
    agg_raw = []
    agg_adverse = []
    agg_adjacent = []
    aggregate_consistency = {}
    outcome_matrix = {
        k: {kk: 0 for kk in ("WIN", "LOSS", "DRAW")}
        for k in ("WIN", "LOSS", "DRAW")
    }
    combined_rows = []
    total_events = 0
    total_independent = 0
    aggregate_warnings = set()

    for week_end in _last_completed_friday_anchors(req.weeks):
        try:
            raw = {
                tf: get_market_data(
                    pair, tf, outputsize=req.candles_per_week,
                    end_date=week_end.tz_convert("UTC"),
                )
                for tf in ANALYSIS_TIMEFRAMES
            }
            base = run_historical_backtest(
                pair=pair, expiry_timeframe=timeframe,
                candle_count=req.candles_per_week,
                market_mode="regular_forex", history_override=raw,
                collect_signal_events=True,
            )
            events = base.pop("_signal_events_for_comparison")
            m1 = _regular_forex_filter(raw["1min"], "1min")
            report = _audit_execution_price_gaps(events, m1, pair, timeframe)
            # Compare close-entry audit to the existing legacy backtest. If
            # different, surface the discrepancy rather than silently claiming P&L.
            legacy_counts = {"WIN": base["wins"], "LOSS": base["losses"], "DRAW": base["draws"]}
            if report["outcome_effect_same_expiry"]["close_entry"] != legacy_counts:
                report["warnings"].append("AUDIT_DISAGREES_WITH_LEGACY_BACKTEST")
                report["status"] = "REVIEW_REQUIRED"
            for k, val in report["data_consistency"].items():
                aggregate_consistency[k] = aggregate_consistency.get(k, 0) + val
            for old_side, transitions in report["outcome_effect_same_expiry"]["outcome_transition_matrix"].items():
                for new_side, count in transitions.items():
                    outcome_matrix[old_side][new_side] += count
            agg_raw.extend(report.pop("_private_all_gaps"))
            agg_adverse.extend(report.pop("_private_adverse_gaps"))
            agg_adjacent.extend(report.pop("_private_adjacent_gaps"))
            combined_rows.extend(report.pop("_private_rows"))
            aggregate_warnings.update(report["warnings"])
            independent = _select_cooldown_events(events, req.cooldown_minutes)
            total_events += len(events)
            total_independent += len(independent)
            weekly.append({
                "week_end_new_york": week_end.isoformat(),
                "evaluated_points": base["evaluated_points"],
                "signals_generated": len(events),
                "independent_signals_after_cooldown": len(independent),
                "data_audit": {
                    "status": base["backtest_data_audit"]["status"],
                    "future_candle_violations": base["backtest_data_audit"]["future_candle_violations"],
                    "m1_window_alignment_violations": base["backtest_data_audit"]["m1_window_alignment_violations"],
                },
                "execution_price_audit": report,
            })
        except HTTPException as exc:
            if isinstance(exc.detail, dict) and exc.detail.get("error") == "BACKTEST_LOOKAHEAD_BLOCKED":
                raise
            failures.append({"week_end_new_york": week_end.isoformat(),
                             "status_code": exc.status_code,
                             "error": make_json_safe(exc.detail)})

    if not weekly:
        raise HTTPException(status_code=502, detail={"error": "NO_VALID_WEEKS",
                                                      "failures": failures})
    total_wins_close = {
        k: sum(outcome_matrix[k].values()) for k in ("WIN", "LOSS", "DRAW")
    }
    total_wins_open = {
        k: sum(outcome_matrix[j][k] for j in ("WIN", "LOSS", "DRAW"))
        for k in ("WIN", "LOSS", "DRAW")
    }
    gap_warnings = []
    if len(agg_adverse) >= 8 and np.mean(agg_adverse) <= -0.25:
        gap_warnings.append("SYSTEMATIC_FAVORABLE_NEXT_OPEN_FOR_SIGNAL_DIRECTION")
    if len(agg_adjacent) < 100:
        gap_warnings.append("INSUFFICIENT_ADJACENT_M1_BASELINE")
    if len(agg_adjacent) >= 100 and np.mean(np.abs(agg_adjacent)) >= 0.30:
        gap_warnings.append("MATERIAL_GAPS_COMMON_IN_UNFILTERED_M1_DATA")
    if len(agg_raw) >= 8 and np.mean(np.abs(agg_raw)) >= 0.50:
        gap_warnings.append("MATERIAL_GAPS_AT_SIGNAL_TIMES")
    aggregate_warnings.update(gap_warnings)
    return make_json_safe({
        "strategy_version": VERSION,
        "pair": pair,
        "expiry_timeframe": timeframe,
        "market_mode": "regular_forex",
        "requested_weeks": req.weeks,
        "completed_weeks": len(weekly),
        "partial": bool(failures),
        "candles_per_week_requested": req.candles_per_week,
        "cooldown_minutes": req.cooldown_minutes,
        "status": "REVIEW_REQUIRED" if aggregate_warnings else "CHECKED",
        "warnings": sorted(aggregate_warnings),
        "overall": {
            "signal_events": total_events,
            "cooldown_selected_events": total_independent,
            "signal_close_to_next_open": _pip_summary(agg_raw),
            "adverse_gap_by_trade_direction": {
                **_pip_summary(agg_adverse),
                "favorable": int(sum(v < -1e-8 for v in agg_adverse)),
                "adverse": int(sum(v > 1e-8 for v in agg_adverse)),
                "neutral": int(sum(abs(v) <= 1e-8 for v in agg_adverse)),
            },
            "baseline_all_adjacent_m1_gaps": _pip_summary(agg_adjacent),
            "baseline_comparison": {
                "mean_abs_signal_gap_pips": round(float(np.mean(np.abs(agg_raw))), 5) if agg_raw else None,
                "mean_abs_baseline_gap_pips": round(float(np.mean(np.abs(agg_adjacent))), 5) if agg_adjacent else None,
                "signal_vs_baseline_abs_gap_ratio": round(
                    float(np.mean(np.abs(agg_raw)) / np.mean(np.abs(agg_adjacent))), 3
                ) if agg_raw and agg_adjacent and float(np.mean(np.abs(agg_adjacent))) > 1e-9 else None,
                "baseline_positive_gap_fraction_percent": round(
                    100 * sum(v > 1e-8 for v in agg_adjacent) / len(agg_adjacent), 2
                ) if agg_adjacent else None,
                "baseline_adequate": len(agg_adjacent) >= 100,
            },
            "data_consistency": aggregate_consistency,
            "outcome_effect_same_expiry": {
                "close_entry": total_wins_close,
                "next_open_entry": total_wins_open,
                "changed_outcomes": sum(
                    outcome_matrix[old][new] for old in outcome_matrix
                    for new in outcome_matrix[old] if old != new
                ),
                "matrix": outcome_matrix,
                "expiry_time_changed": False,
            },
        },
        "weeks": weekly,
        "failures": failures,
        "limitations": [
            "Обнаружение необычных разрывов — сигнал о проверке качества данных, а не доказательство ошибки поставщика.",
            "Котировки Twelve Data M1 не воспроизводят bid/ask и точку исполнения брокера.",
            "Цена после секундной задержки неизвестна по минутным свечам.",
            "Для close-entry и next-open-entry используется ОДНА И ТА ЖЕ свеча экспирации.",
            "Систематически выгодный next-open может быть следствием смещения котировок или правил формирования свечей.",
            "Исторические сигналы близки по времени и не являются независимыми наблюдениями.",
            "Соседние M1 свечи сравниваются по разнице времён, независимо от единиц хранения pandas DatetimeIndex.",
            "Ни один показатель аудита не изменяет торговые фильтры или реальные /api/signal ответы.",
        ],
    })


@app.post("/api/backtest/robustness")
def backtest_execution_robustness(req: BacktestRobustnessRequest):
    """Separate historical weeks, entry sensitivity and cooldown selection.

    Strategy generation is unmodified and uses the same historical engine.
    Fractions of a minute cannot be backtested without tick/second data.
    """
    pair = req.pair.strip().upper()
    if pair not in ALLOWED_PAIRS:
        raise HTTPException(status_code=400, detail="Эта валютная пара не поддерживается")
    expiry_timeframe = req.expiry_timeframe.strip()
    if expiry_timeframe not in ("1min", "5min"):
        raise HTTPException(status_code=400, detail="expiry_timeframe должен быть 1min или 5min")
    scenarios = {
        "next_open_0pip": (0, 0.0),
        "next_open_0_2pip": (0, 0.2),
        "next_open_0_5pip": (0, 0.5),
        "one_minute_later_0pip": (1, 0.0),
    }
    week_results = []
    failures = []
    by_scenario = {name: [] for name in scenarios}
    independent_by_scenario = {name: [] for name in scenarios}
    legacy_results = []
    overall_hours = {}
    cooldown_hours = {}
    overall_gap_pips = []
    excluded_total = 0

    for week_end in _last_completed_friday_anchors(req.weeks):
        try:
            raw = {
                tf: get_market_data(
                    pair, tf, outputsize=req.candles_per_week,
                    end_date=week_end.tz_convert("UTC")
                )
                for tf in ANALYSIS_TIMEFRAMES
            }
            report = run_historical_backtest(
                pair=pair,
                expiry_timeframe=expiry_timeframe,
                candle_count=req.candles_per_week,
                market_mode="regular_forex",
                history_override=raw,
                collect_signal_events=True,
            )
            events = report.pop("_signal_events_for_comparison")
            m1 = _regular_forex_filter(raw["1min"], "1min")
            independent = _select_cooldown_events(events, req.cooldown_minutes)
            week_reports = {}
            independent_reports = {}
            for name, (delay, slip) in scenarios.items():
                details = _simulate_execution(
                    events, m1, pair, expiry_timeframe,
                    delay_minutes=delay, adverse_slippage_pips=slip,
                    collect_hours=(name == "next_open_0pip"),
                )
                by_scenario[name].append(details)
                week_reports[name] = _robustness_result_summary(details)
                ind = _simulate_execution(
                    independent, m1, pair, expiry_timeframe,
                    delay_minutes=delay, adverse_slippage_pips=slip,
                    collect_hours=(name == "next_open_0pip"),
                )
                independent_by_scenario[name].append(ind)
                independent_reports[name] = _robustness_result_summary(ind)
                if name == "next_open_0pip":
                    for hour, counts in ind["by_entry_hour_new_york"].items():
                        total = cooldown_hours.setdefault(hour, {"wins": 0, "losses": 0, "draws": 0})
                        for key in total:
                            total[key] += int(counts[key])
                    for hour, counts in details["by_entry_hour_new_york"].items():
                        total = overall_hours.setdefault(hour, {"wins": 0, "losses": 0, "draws": 0})
                        for key in total:
                            total[key] += int(counts[key])
                    gap = details["signal_close_to_next_open_gap_pips"]
                    if gap["observations"]:
                        overall_gap_pips.append({
                            "week_end_new_york": week_end.isoformat(), **gap,
                        })
            baseline_legacy = {
                "signals_generated": report["signals_generated"],
                "closed_trades": report["closed_trades"],
                "unresolved_entry_candles": 0,
                "unresolved_expiry_candles": report["unresolved_expiry_candles"],
                "wins": report["wins"], "losses": report["losses"], "draws": report["draws"],
            }
            legacy_results.append(baseline_legacy)
            excluded_total += len(events) - len(independent)
            week_results.append({
                "week_end_new_york": week_end.isoformat(),
                "week_end_utc": week_end.tz_convert("UTC").isoformat(),
                "evaluated_points": report["evaluated_points"],
                "signals_generated": len(events),
                "independent_signals": len(independent),
                "excluded_by_cooldown": len(events) - len(independent),
                "data_audit": {
                    "status": report["backtest_data_audit"]["status"],
                    "warnings": report["backtest_data_audit"]["warnings"],
                    "future_candle_violations": report["backtest_data_audit"]["future_candle_violations"],
                    "m1_window_alignment_violations": report["backtest_data_audit"]["m1_window_alignment_violations"],
                },
                "legacy_close_price_entry": _combine_robustness_reports([baseline_legacy]),
                "scenarios": week_reports,
                "cooldown_scenarios": independent_reports,
            })
        except HTTPException as exc:
            if isinstance(exc.detail, dict) and exc.detail.get("error") == "BACKTEST_LOOKAHEAD_BLOCKED":
                raise
            failures.append({
                "week_end_new_york": week_end.isoformat(),
                "status_code": exc.status_code,
                "error": make_json_safe(exc.detail),
            })

    if not week_results:
        raise HTTPException(
            status_code=502,
            detail={"error":"NO_VALID_WEEKS", "failures": failures},
        )
    hour_result = {
        hour: {
            **counts, "trades": sum(counts.values()),
            "win_rate_excluding_draws_percent": (
                round(counts["wins"] * 100 / (counts["wins"] + counts["losses"]), 2)
                if counts["wins"] + counts["losses"] else None
            ),
            "net_units_80pct_per_unit_stake": round(counts["wins"] * 0.8 - counts["losses"], 4),
        }
        for hour, counts in sorted(overall_hours.items())
    }
    cooldown_hour_result = {
        hour: {
            **counts, "trades": sum(counts.values()),
            "win_rate_excluding_draws_percent": (
                round(counts["wins"] * 100 / (counts["wins"] + counts["losses"]), 2)
                if counts["wins"] + counts["losses"] else None
            ),
            "net_units_80pct_per_unit_stake": round(counts["wins"] * 0.8 - counts["losses"], 4),
        }
        for hour, counts in sorted(cooldown_hours.items())
    }
    return make_json_safe({
        "strategy_version": VERSION,
        "pair": pair,
        "expiry_timeframe": expiry_timeframe,
        "requested_weeks": req.weeks,
        "completed_weeks": len(week_results),
        "partial": bool(failures),
        "candles_per_week_requested": req.candles_per_week,
        "market_mode": "regular_forex",
        "cooldown_minutes": req.cooldown_minutes,
        "execution_proxy": (
            "Signal computed after M1 close T. Execute at the next available M1 candle "
            "open T, or after a full 1-minute delay at T+1min; optional adverse "
            "0.2/0.5 pip entry-price shift. No second-level/tick fills are modeled."
        ),
        "legacy_close_price_entry": _combine_robustness_reports(legacy_results),
        "overall": {
            name: _combine_robustness_reports(results)
            for name, results in by_scenario.items()
        },
        "independent_cooldown": {
            "excluded_signals_by_cooldown": excluded_total,
            "by_entry_hour_new_york": cooldown_hour_result,
            "overall": {
                name: _combine_robustness_reports(results)
                for name, results in independent_by_scenario.items()
            },
        },
        "by_entry_hour_new_york": hour_result,
        "weekly_price_gap_diagnostics": overall_gap_pips,
        "weeks": week_results,
        "failures": failures,
        "limitations": [
            "Диагностика исполнения приблизительная: M1 OHLC не содержит цены через 5/10/15 секунд после сигнала.",
            "Вариант next_open использует цену открытия следующей минутной свечи, но не реальный брокерский fill.",
            "Вариант one_minute_later меняет момент входа на +1 минуту, экспирация отсчитывается от нового входа.",
            "Сдвиг цены 0.2/0.5 пипса — стресс-сценарий, не фактическая оценка проскальзывания.",
            "Cooldown отбирает разнесённые сигналы для анализа; это не гарантия статистической независимости и не изменение /api/signal.",
            "Ранее исследованные недели не являются независимой от подбора модели контрольной выборкой.",
            "Одна неделя — лишь выборка до candles_per_week свечей, не полные пять дней.",
            "Не учитываются отдельные правила Pocket Option, OTC, реальные выплаты и задержки платформы.",
            "Ничьи считаются возвратом ставки; соседние сигналы статистически зависимы.",
            "Отчёт по часам основан на небольшом числе сигналов, использовать его как подбор лучших часов нельзя без внешней проверки.",
        ],
    })


# ==================================================
# CONSERVATIVE EXECUTION AUDIT - NO FAVORABLE GAPS ASSUMED
# ==================================================

def _conservative_fill_test(events, m1, pair, expiry_timeframe):
    """Pessimistic historical settlement without using favorable close/open gaps.

    This is a sensitivity test, NOT a claim to reproduce actual broker fills.
    Signal is available at T after the preceding M1 candle closes.
    Settlement is M1 closing at T + expiry_timeframe.
    When a later M1 OPEN would have improved the entry, the pessimistic
    scenario uses the less favorable *reference close* instead. When OPEN
    is worse, it uses OPEN. Additional 0.2/0.5 pip moves are adverse.
    """
    if expiry_timeframe not in ('1min', '5min'):
        raise ValueError('Only 1min and 5min expiries supported')
    pip = _pip_size_for_pair(pair)
    one_minute = pd.Timedelta(minutes=1)
    names = ('reference_close', 'observed_next_open', 'no_favorable_gap',
             'no_favorable_gap_plus_0_2pip', 'no_favorable_gap_plus_0_5pip')
    counters = {
        name: {'signals_generated': len(events), 'closed_trades': 0,
               'wins': 0, 'losses': 0, 'draws': 0,
               'unresolved_price_or_time': 0,
               'unresolved_expiry_candles': 0}
        for name in names
    }
    favorable_gap_events = 0
    adverse_gap_events = 0
    neutral_gap_events = 0
    total_pips_favorable_removed = 0.0
    observations = 0
    example_rows = []
    for event in sorted(events, key=lambda x: pd.Timestamp(x['entry_time'])):
        side = event['signal']
        if side not in ('CALL', 'PUT'):
            raise ValueError('Expected CALL/PUT event')
        t = pd.Timestamp(event['entry_time'])
        if t.tzinfo is None:
            raise ValueError('Naive event timestamp')
        preceding_open = t - one_minute
        if preceding_open not in m1.index or t not in m1.index:
            for counter in counters.values():
                counter['unresolved_price_or_time'] += 1
            continue
        source_close = float(m1.at[preceding_open, 'close'])
        reference_close = float(event['entry_price'])
        next_open = float(m1.at[t, 'open'])
        if (not all(math.isfinite(x) and x > 0 for x in
                    (source_close, reference_close, next_open))):
            for counter in counters.values():
                counter['unresolved_price_or_time'] += 1
            continue
        if abs(source_close - reference_close) > max(1e-8, pip / 1000.0):
            # Never silently price an event against a different source candle.
            raise ValueError('Signal event entry_price does not match preceding closed candle')
        expiry_at = t + TIMEFRAMES[expiry_timeframe]
        exit_info = _m1_candle_at_expiry(m1, expiry_at)
        if exit_info is None:
            for counter in counters.values():
                counter['unresolved_expiry_candles'] += 1
            continue
        exit_price = float(exit_info['exit_price'])
        if not math.isfinite(exit_price) or exit_price <= 0:
            for counter in counters.values():
                counter['unresolved_expiry_candles'] += 1
            continue
        # Lower entry is favorable for CALL, higher entry favorable for PUT.
        guarded = max(source_close, next_open) if side == 'CALL' else min(source_close, next_open)
        favorable_in_pips = ((source_close - next_open) / pip
                             if side == 'CALL' else (next_open - source_close) / pip)
        if favorable_in_pips > 1e-8:
            favorable_gap_events += 1
            total_pips_favorable_removed += favorable_in_pips
        elif favorable_in_pips < -1e-8:
            adverse_gap_events += 1
        else:
            neutral_gap_events += 1
        observations += 1
        adverse_sign = 1 if side == 'CALL' else -1
        prices = {
            'reference_close': source_close,
            'observed_next_open': next_open,
            'no_favorable_gap': guarded,
            'no_favorable_gap_plus_0_2pip': guarded + adverse_sign * 0.2 * pip,
            'no_favorable_gap_plus_0_5pip': guarded + adverse_sign * 0.5 * pip,
        }
        outcomes = {}
        for name, entry_price in prices.items():
            outcome = _expiry_outcome(side, entry_price, exit_price)
            outcomes[name] = outcome
            counts = counters[name]
            counts['closed_trades'] += 1
            counts[{'WIN': 'wins', 'LOSS': 'losses', 'DRAW': 'draws'}[outcome]] += 1
        # Detailed examples are diagnostic only, never used for computing rates.
        if len(example_rows) < 8 and (favorable_in_pips > 1e-8 or
                                     outcomes['observed_next_open'] != outcomes['no_favorable_gap']):
            example_rows.append({
                'direction': side, 'entry_time_utc': t.isoformat(),
                'expiry_at_utc': expiry_at.isoformat(),
                'signal_close': round(source_close, 7),
                'observed_next_open': round(next_open, 7),
                'conservative_entry': round(guarded, 7),
                'exit_close': round(exit_price, 7),
                'favorable_gap_pips': round(max(0.0, favorable_in_pips), 4),
                'observed_outcome': outcomes['observed_next_open'],
                'conservative_outcome': outcomes['no_favorable_gap'],
            })
    totals = {}
    for name, counts in counters.items():
        wins, losses, draws = (counts[k] for k in ('wins', 'losses', 'draws'))
        total = counts['closed_trades']
        totals[name] = {
            **counts,
            'win_rate_percent': round(100 * wins / total, 2) if total else None,
            'payout_analysis': _payout_analysis(wins, losses, draws),
        }
    return {
        'scenario_results': totals,
        'gap_dependency': {
            'verified_events': observations,
            'favorable_gap_events': favorable_gap_events,
            'adverse_gap_events': adverse_gap_events,
            'neutral_gap_events': neutral_gap_events,
            'total_favorable_pips_removed': round(total_pips_favorable_removed, 4),
            'avg_favorable_pips_removed_per_verified_signal': (
                round(total_pips_favorable_removed / observations, 4)
                if observations else None
            ),
        },
        'sample_trades': example_rows,
    }


def _combine_conservative_scenarios(reports):
    if not reports:
        return {}
    result = {}
    for name in reports[0]:
        counts = {key: sum(int(r[name][key]) for r in reports)
                  for key in ('signals_generated', 'closed_trades', 'wins',
                              'losses', 'draws', 'unresolved_price_or_time',
                              'unresolved_expiry_candles')}
        wins, losses, draws = counts['wins'], counts['losses'], counts['draws']
        result[name] = {
            **counts,
            'win_rate_percent': (round(100 * wins / counts['closed_trades'], 2)
                                 if counts['closed_trades'] else None),
            'payout_analysis': _payout_analysis(wins, losses, draws),
        }
    return result


@app.post('/api/backtest/conservative')
def backtest_conservative_execution(req: BacktestRobustnessRequest):
    """Compare historical observed fills with conservative no-gap-advantage fills.

    No modification to strategy filters, production /api/signal or execution.
    Use same weekly signal events as robustness and same expiry for ALL cases.
    """
    pair = req.pair.strip().upper()
    if pair not in ALLOWED_PAIRS:
        raise HTTPException(status_code=400, detail='Эта валютная пара не поддерживается')
    expiry = req.expiry_timeframe.strip()
    if expiry not in ('1min', '5min'):
        raise HTTPException(status_code=400, detail='expiry_timeframe должен быть 1min или 5min')
    week_results = []
    failures = []
    raw_reports = []
    cooldown_reports = []
    gap_flags = set()
    gap_totals = {'verified_events': 0, 'favorable_gap_events': 0,
                  'adverse_gap_events': 0, 'neutral_gap_events': 0,
                  'total_favorable_pips_removed': 0.0}
    gap_examples = []

    for week_end in _last_completed_friday_anchors(req.weeks):
        try:
            raw = {
                tf: get_market_data(pair, tf, outputsize=req.candles_per_week,
                                    end_date=week_end.tz_convert('UTC'))
                for tf in ANALYSIS_TIMEFRAMES
            }
            base = run_historical_backtest(
                pair=pair, expiry_timeframe=expiry,
                candle_count=req.candles_per_week, market_mode='regular_forex',
                history_override=raw, collect_signal_events=True,
            )
            events = base.pop('_signal_events_for_comparison')
            m1 = _regular_forex_filter(raw['1min'], '1min')
            audit = _audit_execution_price_gaps(events, m1, pair, expiry)
            gap_flags.update(audit['warnings'])
            raw_test = _conservative_fill_test(events, m1, pair, expiry)
            cooldown = _select_cooldown_events(events, req.cooldown_minutes)
            cooldown_test = _conservative_fill_test(cooldown, m1, pair, expiry)
            for k in gap_totals:
                gap_totals[k] += raw_test['gap_dependency'][k]
            raw_reports.append(raw_test['scenario_results'])
            cooldown_reports.append(cooldown_test['scenario_results'])
            gap_examples.extend(raw_test['sample_trades'])
            week_results.append({
                'week_end_new_york': week_end.isoformat(),
                'evaluated_points': base['evaluated_points'],
                'signals': len(events),
                'after_cooldown': len(cooldown),
                'time_audit': {
                    'status': base['backtest_data_audit']['status'],
                    'future_candle_violations': base['backtest_data_audit']['future_candle_violations'],
                    'm1_window_alignment_violations': base['backtest_data_audit']['m1_window_alignment_violations'],
                },
                'price_audit': {
                    'status': audit['status'], 'warnings': audit['warnings'],
                    'mean_abs_signal_gap_pips': audit['gap_comparison']['mean_absolute_signal_gap_pips'],
                    'mean_abs_baseline_gap_pips': audit['gap_comparison']['mean_absolute_baseline_gap_pips'],
                    'baseline_observations': audit['all_adjacent_m1_close_to_next_open_gaps']['observations'],
                },
                'scenarios': raw_test['scenario_results'],
                'cooldown_scenarios': cooldown_test['scenario_results'],
            })
        except HTTPException as exc:
            if isinstance(exc.detail, dict) and exc.detail.get('error') == 'BACKTEST_LOOKAHEAD_BLOCKED':
                raise
            failures.append({'week_end_new_york': week_end.isoformat(),
                             'status_code': exc.status_code,
                             'error': make_json_safe(exc.detail)})
        except ValueError as exc:
            failures.append({'week_end_new_york': week_end.isoformat(),
                             'status_code': 422, 'error': str(exc)})

    if not week_results:
        raise HTTPException(status_code=502,
                            detail={'error': 'NO_VALID_WEEKS', 'failures': failures})
    verified_events = gap_totals['verified_events']
    gap_totals['total_favorable_pips_removed'] = round(
        gap_totals['total_favorable_pips_removed'], 4)
    gap_totals['avg_favorable_pips_removed_per_verified_signal'] = (
        round(gap_totals['total_favorable_pips_removed'] / verified_events, 4)
        if verified_events else None
    )
    gap_status = 'REVIEW_REQUIRED' if gap_flags else 'CHECKED'
    return make_json_safe({
        'strategy_version': VERSION,
        'pair': pair, 'market_mode': 'regular_forex',
        'expiry_timeframe': expiry,
        'requested_weeks': req.weeks,
        'completed_weeks': len(week_results),
        'partial': bool(failures),
        'candles_per_week_requested': req.candles_per_week,
        'cooldown_minutes': req.cooldown_minutes,
        'data_quality_status': gap_status,
        'data_quality_warnings': sorted(gap_flags),
        'live_trading_validation': 'NOT_VALIDATED',
        'method': {
            'reference_close': 'Non-executable reference: close of the signal candle',
            'observed_next_open': 'Historical next M1 OPEN; NOT a broker execution price',
            'no_favorable_gap': 'CALL=max(signal close, next open); PUT=min(signal close, next open)',
            'no_favorable_gap_plus_0_2pip': 'No favorable gap + 0.2 pip adverse entry adjustment',
            'no_favorable_gap_plus_0_5pip': 'No favorable gap + 0.5 pip adverse entry adjustment',
            'expiry': 'All scenarios use the same closed M1 candle at entry clock T + expiry',
        },
        'overall': _combine_conservative_scenarios(raw_reports),
        'cooldown_overall': _combine_conservative_scenarios(cooldown_reports),
        'gap_dependency': gap_totals,
        'sample_trades': gap_examples[:12],
        'weeks': week_results,
        'failures': failures,
        'limitations': [
            'Это консервативная оценка чувствительности, а не моделирование реального исполнения брокером.',
            'Сценарий no_favorable_gap удаляет положительный эффект разницы Close→Open, но не устраняет прочие систематические погрешности.',
            'Данные Twelve Data не являются котировками Pocket Option и не отражают OTC цены.',
            'Не учитываются задержки 5–15 секунд, bid/ask, фактические выплаты и правила ничьих.',
            'Четыре короткие недельные выборки и сигналы внутри одной недели не являются независимыми испытаниями.',
            'Положительный результат теста не означает доказанную доходность; для живой торговли нужна независимая проверка исполнения.',
            'Торговые фильтры и эндпоинт /api/signal не изменены.',
        ],
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
        "twelvedata_key_configured": bool(TWELVEDATA_KEY),
        "backtest_slice_engine": "pandas_datetime_searchsorted_v2",
        "regular_forex_sessions": True,
        "weekly_expiry_comparison": True,
        "execution_robustness_backtest": True,
        "execution_price_audit": True,
        "execution_gap_baseline_unit_safe": True,
        "conservative_execution_backtest": True,
        "forward_paper_test": True,
        "paper_storage_configured": (PAPER_DB_TYPE == "postgres" or (PAPER_DB_TYPE == "sqlite" and (not os.getenv("RENDER") or os.getenv("PAPER_SQLITE_PERSISTENT_ACK") == "YES"))),
        "paper_api_key_configured": len(PAPER_API_KEY) >= 16,
        "paper_mode": "DEMO_ONLY",
        "paper_capture_requires_authenticated_scan": True,
        "paper_scan_monitoring": True,
        "paper_scan_receipt_required": True,
        "live_candle_snapshot_clock": "shared_analysis_timestamp_v1",
        "m1_paper_freshness_guard_seconds": PAPER_MAX_SIGNAL_AGE_SECONDS,
        "m1_source_timing_audit": True,
        "m1_missing_closed_bar_bounded_retry": bool(M1_RETRY_ENABLED),
        "m1_max_refresh_attempts": M1_RETRY_MAX_ATTEMPTS,
        "m1_live_fetch_priority": True,
        "paper_decision_audit_persisted": True,
        "paper_diagnostics_report": True,
        "td_credit_guard_enabled": True,
        "td_local_minute_budget": TD_LOCAL_MINUTE_BUDGET,
        "td_provider_minute_limit": TD_PROVIDER_MINUTE_CREDITS,
        "td_local_daily_budget": TD_PROVIDER_DAILY_CREDITS,
        "td_overlapping_live_scans_blocked": True,
        "td_upstream_429_circuit_breaker": True,
        "stale_scan_audit": True,
        "retrospective_virtual_fills_disabled": True,
    }


# ==================================================
# v2.4.16 — FORWARD PAPER TEST + FRESHNESS AUDIT
# ==================================================
# This extension does not change SMC signal calculations or backtest rules.
# The public /api/signal response now includes freshness and blocks stale signals.
# It records *observed forward* opportunities only when an authenticated
# client explicitly invokes POST /api/paper/scan.

import csv
import hmac
import io
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from fastapi import Depends, Header
from fastapi.responses import Response

PAPER_API_KEY = os.getenv("PAPER_API_KEY", "")
PAPER_DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
PAPER_SQLITE_PATH = os.getenv("PAPER_SQLITE_PATH", "").strip()
PAPER_DB_TYPE = "postgres" if PAPER_DATABASE_URL else ("sqlite" if PAPER_SQLITE_PATH else "disabled")
_PAPER_DB_READY = False
_PAPER_DB_LOCK = threading.Lock()

PAPER_DDL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id TEXT PRIMARY KEY,
    pair TEXT NOT NULL,
    side TEXT NOT NULL,
    expiry_timeframe TEXT NOT NULL,
    signal_candle_open_utc TEXT NOT NULL,
    signal_available_at_utc TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    signal_age_seconds DOUBLE PRECISION NOT NULL,
    signal_close DOUBLE PRECISION NOT NULL,
    signal_reason TEXT NOT NULL,
    score_call INTEGER NOT NULL,
    score_put INTEGER NOT NULL,
    strategy_version TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    status TEXT NOT NULL,
    entry_at_utc TEXT,
    entry_price DOUBLE PRECISION,
    expiry_at_utc TEXT,
    stake DOUBLE PRECISION,
    payout_percent DOUBLE PRECISION,
    broker TEXT,
    exit_at_utc TEXT,
    exit_price DOUBLE PRECISION,
    price_outcome TEXT,
    broker_outcome TEXT,
    effective_outcome TEXT,
    net_profit DOUBLE PRECISION,
    entry_adverse_gap_pips DOUBLE PRECISION,
    void_reason TEXT,
    notes TEXT,
    updated_at_utc TEXT NOT NULL,
    UNIQUE (pair, side, expiry_timeframe, signal_candle_open_utc)
)
"""


PAPER_SCAN_DDL = """
CREATE TABLE IF NOT EXISTS paper_scan_runs (
    id TEXT PRIMARY KEY,
    pair TEXT NOT NULL,
    expiry_timeframe TEXT NOT NULL,
    scanned_at_utc TEXT NOT NULL,
    result_status TEXT NOT NULL,
    signal_candle_open_utc TEXT,
    trade_id TEXT,
    reason TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    analysis_at_utc TEXT,
    signal_available_at_utc TEXT,
    signal_age_seconds DOUBLE PRECISION,
    freshness_status TEXT,
    freshness_reason TEXT,
    last_candles_json TEXT
)
"""


class PaperScanRequest(BaseModel):
    pair: str = Field(default="EUR/USD")
    timeframe: str = Field(default="1min")


class PaperEntryRequest(BaseModel):
    entry_at_utc: datetime
    entry_price: float = Field(gt=0, allow_inf_nan=False)
    stake: float = Field(default=1.0, gt=0, le=100000, allow_inf_nan=False)
    payout_percent: float = Field(default=80.0, ge=0, le=100, allow_inf_nan=False)
    broker: str = Field(default="DEMO", min_length=1, max_length=100)
    expiry_at_utc: datetime | None = None
    notes: str = Field(default="", max_length=500)


class PaperExitRequest(BaseModel):
    exit_at_utc: datetime
    exit_price: float = Field(gt=0, allow_inf_nan=False)
    broker_outcome: str | None = Field(default=None)
    notes: str = Field(default="", max_length=500)


class PaperVoidRequest(BaseModel):
    reason: str = Field(min_length=5, max_length=500)


def _paper_auth(x_paper_key: str | None = Header(default=None, alias="X-Paper-Key")):
    # Authentication is required even to list/export paper results.
    if len(PAPER_API_KEY) < 16:
        raise HTTPException(status_code=503, detail="Configure PAPER_API_KEY (minimum 16 characters) in Render")
    if not x_paper_key:
        raise HTTPException(status_code=401, detail="X-Paper-Key header is required")
    if not hmac.compare_digest(PAPER_API_KEY, x_paper_key):
        raise HTTPException(status_code=403, detail="Invalid paper API key")
    return True


def _paper_require_storage_config():
    if PAPER_DB_TYPE == "disabled":
        raise HTTPException(
            status_code=503,
            detail="Configure DATABASE_URL (Postgres) or PAPER_SQLITE_PATH on a persistent disk; do not use an ephemeral Render filesystem",
        )
    if PAPER_DB_TYPE == "sqlite" and os.getenv("RENDER") and os.getenv("PAPER_SQLITE_PERSISTENT_ACK") != "YES":
        raise HTTPException(
            status_code=503,
            detail="For SQLite on Render, attach a persistent disk and set PAPER_SQLITE_PERSISTENT_ACK=YES",
        )


@contextmanager
def _paper_db():
    _paper_require_storage_config()
    if PAPER_DB_TYPE == "postgres":
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError:
            raise HTTPException(status_code=503, detail="Install psycopg[binary] in requirements.txt")
        try:
            with psycopg.connect(PAPER_DATABASE_URL, row_factory=dict_row, connect_timeout=8) as conn:
                yield conn
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Paper Postgres connection/operation error")
            raise HTTPException(status_code=503, detail="Paper Postgres is unavailable") from exc
    else:
        path = Path(PAPER_SQLITE_PATH).expanduser().resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), timeout=15)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=15000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Paper SQLite connection/operation error")
            raise HTTPException(status_code=503, detail="Paper SQLite is unavailable") from exc


def _paper_exec(conn, sql, params=()):
    if PAPER_DB_TYPE == "postgres":
        sql = sql.replace("?", "%s")
    return conn.execute(sql, params)


def _paper_initialize():
    global _PAPER_DB_READY
    if _PAPER_DB_READY:
        return
    with _PAPER_DB_LOCK:
        if _PAPER_DB_READY:
            return
        with _paper_db() as conn:
            _paper_exec(conn, PAPER_DDL)
            _paper_exec(conn, "CREATE INDEX IF NOT EXISTS paper_status_idx ON paper_trades(status, observed_at_utc)")
            _paper_exec(conn, PAPER_SCAN_DDL)
            # Existing 2.4.15 databases already have this table: migrate in
            # place without dropping scans or paper trades (SQLite + Postgres).
            extra_columns = (
                ("analysis_at_utc", "TEXT"),
                ("signal_available_at_utc", "TEXT"),
                ("signal_age_seconds", "DOUBLE PRECISION"),
                ("freshness_status", "TEXT"),
                ("freshness_reason", "TEXT"),
                ("last_candles_json", "TEXT"),
            )
            if PAPER_DB_TYPE == "postgres":
                for column_name, column_type in extra_columns:
                    _paper_exec(conn, f"ALTER TABLE paper_scan_runs ADD COLUMN IF NOT EXISTS {column_name} {column_type}")
            else:
                existing = {row["name"] for row in _paper_exec(conn, "PRAGMA table_info(paper_scan_runs)").fetchall()}
                for column_name, column_type in extra_columns:
                    if column_name not in existing:
                        _paper_exec(conn, f"ALTER TABLE paper_scan_runs ADD COLUMN {column_name} {column_type}")
            _paper_exec(conn, "CREATE INDEX IF NOT EXISTS paper_runs_time_idx ON paper_scan_runs(scanned_at_utc)")
        _PAPER_DB_READY = True


def _paper_iso(value):
    # ISO-8601 UTC, preserving microseconds; strict timezone required.
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or pd.isna(ts):
        raise HTTPException(status_code=422, detail="Time must include an explicit timezone, e.g. +00:00")
    return ts.tz_convert("UTC").isoformat()


def _paper_time(value):
    return pd.Timestamp(value).tz_convert("UTC")


def _paper_dict(row):
    return dict(row) if row is not None else None


def _paper_fetch(conn, trade_id):
    return _paper_dict(_paper_exec(conn, "SELECT * FROM paper_trades WHERE id=?", (trade_id,)).fetchone())


def _paper_now():
    return pd.Timestamp.now(tz="UTC")


def _paper_public_trade(row):
    if row is None:
        return None
    result = dict(row)
    result.pop("analysis_json", None)
    result["outcome_mismatch"] = bool(
        row.get("broker_outcome") and row.get("price_outcome")
        and row["broker_outcome"] != row["price_outcome"]
    )
    return make_json_safe(result)


def _paper_record_scan_result(pair: str, timeframe: str, result: dict, candle_time=None, reason="", timing=None):
    """Persist every authenticated scan and return an independently verified DB receipt.

    IMPORTANT: Successful HTTP 200 with WAIT used to be returned even when the
    scan-log insert failed. In 2.4.14, logging failures must surface as HTTP 503,
    so GitHub Actions cannot report a false successful scan.

    For a confirmed CALL/PUT, the paper opportunity may already have committed
    before this log is written; clients may retry, as trade insertion is deduped.
    """
    status = str(result.get("status", "UNKNOWN"))
    trade = result.get("trade") or {}
    timing = timing or {}
    scan_id = str(uuid.uuid4())
    try:
        with _paper_db() as conn:
            cursor = _paper_exec(conn, """
                INSERT INTO paper_scan_runs (
                    id, pair, expiry_timeframe, scanned_at_utc, result_status,
                    signal_candle_open_utc, trade_id, reason, strategy_version,
                    analysis_at_utc, signal_available_at_utc, signal_age_seconds,
                    freshness_status, freshness_reason, last_candles_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                scan_id, pair, timeframe, _paper_iso(_paper_now()),
                status, str(candle_time) if candle_time is not None else None,
                trade.get("id"), str(reason or result.get("reason") or "")[:500], VERSION,
                timing.get("analysis_at_utc"), timing.get("signal_available_at_utc"),
                timing.get("signal_age_seconds"), timing.get("status"),
                str(timing.get("reason") or "")[:500],
                __import__("json").dumps(_paper_scan_diagnostic_payload(timing), ensure_ascii=False),
            ))
            if cursor.rowcount != 1:
                raise RuntimeError("Scan log insert affected no rows")

        # A second connection confirms that the committed row is readable by
        # the same backend used by GET /api/paper/monitor.
        with _paper_db() as conn:
            receipt = _paper_exec(
                conn,
                "SELECT id FROM paper_scan_runs WHERE id=?", (scan_id,),
            ).fetchone()
        if receipt is None:
            raise RuntimeError("Committed scan log receipt is not readable")
    except Exception as exc:
        logger.exception("Paper scan logging FAILED: no durable receipt for scan_id=%s", scan_id)
        raise HTTPException(
            status_code=503,
            detail="Paper scan result could not be verified in storage; inspect Render server logs",
        ) from exc
    return {
        **result,
        "scan_logged": True,
        "scan_log_id": scan_id,
        "signal_age_seconds": timing.get("signal_age_seconds"),
        "data_freshness_status": timing.get("status"),
        "max_signal_age_seconds": PAPER_MAX_SIGNAL_AGE_SECONDS,
    }


@app.get("/api/paper/monitor", dependencies=[Depends(_paper_auth)])
def paper_monitor(limit: int = Query(default=12, ge=1, le=50)):
    """Report actual API scans, not assumed external scheduler availability."""
    _paper_initialize()
    cutoff = _paper_iso(_paper_now() - pd.Timedelta(hours=24))
    with _paper_db() as conn:
        recent = _paper_exec(conn, """
            SELECT id, pair, expiry_timeframe, scanned_at_utc, result_status,
                   signal_candle_open_utc, trade_id, reason, strategy_version,
                   analysis_at_utc, signal_available_at_utc, signal_age_seconds,
                   freshness_status, freshness_reason, last_candles_json
            FROM paper_scan_runs ORDER BY scanned_at_utc DESC LIMIT ?
        """, (limit,)).fetchall()
        counts = _paper_exec(conn, """
            SELECT result_status, COUNT(*) AS n FROM paper_scan_runs
            WHERE scanned_at_utc >= ? GROUP BY result_status
        """, (cutoff,)).fetchall()
        total = _paper_exec(conn, "SELECT COUNT(*) AS n FROM paper_scan_runs").fetchone()["n"]
    runs = []
    for row in recent:
        item = _paper_dict(row)
        payload = item.pop("last_candles_json", None)
        try:
            candles = __import__("json").loads(payload or "{}")
            m1 = candles.get("1min") or {}
            item["m1_provider_diagnostics"] = m1.get("provider_diagnostics")
            item["m1_refresh"] = m1.get("m1_refresh")
            item["decision_audit"] = candles.get("_v2419_decision_audit")
            item["live_pipeline"] = candles.get("_v2419_live_pipeline")
        except (ValueError, TypeError, AttributeError):
            item["m1_provider_diagnostics"] = None
            item["m1_refresh"] = None
            item["decision_audit"] = None
            item["live_pipeline"] = None
        runs.append(item)
    return {
        "version": VERSION,
        "mode": "DEMO_ONLY",
        "scheduler": "EXTERNAL_REQUIRED_NOT_VERIFIED",
        "total_logged_scans": int(total),
        "last_scan_at_utc": runs[0]["scanned_at_utc"] if runs else None,
        "last_scan_status": runs[0]["result_status"] if runs else None,
        "scans_last_24_hours": int(sum(row["n"] for row in counts)),
        "status_counts_last_24_hours": {row["result_status"]: int(row["n"]) for row in counts},
        "recent_scans": runs,
        "max_signal_age_seconds": PAPER_MAX_SIGNAL_AGE_SECONDS,
        "twelvedata_budget": _td_quota_snapshot(),
        "diagnostics_last_24_hours": _paper_query_diagnostic_summary(hours=24),
        "note": "Shows authenticated scan calls, whether automatic or manual. Local API counters do not include other applications or worker processes. No broker execution.",
    }


@app.get("/api/paper/diagnostics", dependencies=[Depends(_paper_auth)])
def paper_diagnostics(hours: int = Query(default=24, ge=1, le=168)):
    """Persistent, read-only root-cause counters; no Twelve Data calls."""
    return make_json_safe({
        "version": VERSION,
        "mode": "DEMO_ONLY",
        "statistics": _paper_query_diagnostic_summary(hours=hours),
        "twelvedata_budget": _td_quota_snapshot(),
    })


@app.get("/api/paper/health", dependencies=[Depends(_paper_auth)])
def paper_health():
    _paper_initialize()
    with _paper_db() as conn:
        count = _paper_exec(conn, "SELECT COUNT(*) AS n FROM paper_trades").fetchone()["n"]
    return {"status": "READY", "version": VERSION, "storage": PAPER_DB_TYPE,
            "stored_trades": int(count), "automation": "EXTERNAL_SCHEDULER_REQUIRED",
            "mode": "DEMO_ONLY", "scan_monitoring": True}


@app.post("/api/paper/scan", dependencies=[Depends(_paper_auth)])
def paper_scan(req: PaperScanRequest):
    """DEMO observation only. Persist both fresh WAIT and stale scan diagnostics."""
    _paper_initialize()
    pair = req.pair.strip().upper()
    timeframe = req.timeframe.strip()
    if pair not in ALLOWED_PAIRS:
        raise HTTPException(status_code=400, detail="Unsupported Forex pair")
    if timeframe not in ("1min", "5min"):
        raise HTTPException(status_code=400, detail="Paper test supports 1min and 5min expiry")

    try:
        signal_data = get_signal(SignalRequest(pair=pair, timeframe=timeframe))
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        code = detail.get("code")
        if exc.status_code == 429 and code in ("API_RATE_LIMITED", "SCAN_BUSY"):
            status = "SCAN_BUSY" if code == "SCAN_BUSY" else "API_RATE_LIMITED"
            reason = str(detail.get("reason") or detail.get("message") or status)
            m1_diagnostics = detail.get("m1_diagnostics") or {}
            rejected_timing = {"candles": {"1min": m1_diagnostics}} if m1_diagnostics else None
            return _paper_record_scan_result(pair, timeframe, {
                "status": status,
                "stored": False,
                "pair": pair,
                "reason": reason,
                "strategy_version": VERSION,
                "retry_after_seconds": detail.get("retry_after_seconds"),
                "twelvedata_budget": _td_quota_snapshot(),
            }, reason=reason, timing=rejected_timing)
        raise
    timing = dict(signal_data["freshness"])
    # Capture raw decision even if paper execution later marks M1 stale.
    # Never count stale snapshots as actionable signals in summary.
    timing["decision_audit"] = _live_decision_audit(signal_data)
    timing["live_pipeline"] = signal_data.get("live_pipeline")
    candle_open = _paper_time(signal_data["candle_time"])
    available_at = candle_open + TIMEFRAMES["1min"]
    observed_at = _paper_now()  # repeat the safety check after get_signal returns
    age_seconds = float((observed_at - available_at).total_seconds())
    timing["signal_age_seconds"] = round(age_seconds, 3)
    timing["observed_at_utc"] = _paper_iso(observed_at)

    # This also applies to WAIT scans; otherwise the monitor hides stale data.
    if age_seconds < 0 or age_seconds > PAPER_MAX_SIGNAL_AGE_SECONDS or timing["status"] != "FRESH":
        timing["status"] = "STALE" if age_seconds >= 0 else "NOT_CLOSED"
        timing["reason"] = (
            f"M1 candle age {age_seconds:.1f}s (limit {PAPER_MAX_SIGNAL_AGE_SECONDS}s)"
            if age_seconds >= 0 else "M1 candle not closed at observation time"
        )
        return _paper_record_scan_result(pair, timeframe, {
            "status": "STALE_SKIPPED", "stored": False, "pair": pair,
            "reason": timing["reason"],
            "candle_time": signal_data["candle_time"],
            "raw_signal": signal_data["raw_signal_before_freshness_gate"],
            "strategy_version": VERSION,
        }, candle_time=signal_data["candle_time"], reason=timing["reason"], timing=timing)

    side = signal_data["signal"]
    if side not in ("CALL", "PUT"):
        return _paper_record_scan_result(pair, timeframe, {
            "status": "WAIT", "stored": False, "pair": pair,
            "reason": signal_data["reason"],
            "candle_time": signal_data["candle_time"],
            "strategy_version": VERSION,
        }, candle_time=signal_data["candle_time"], reason=signal_data["reason"], timing=timing)

    # Store only the observed SIGNAL; never infer a fill from the old candle close.
    # A real DEMO entry/exit must be recorded via the manual paper-trade APIs.
    score = signal_data.get("score") or {}
    trade_id = str(uuid.uuid4())
    with _paper_db() as conn:
        _paper_exec(conn, """
            INSERT INTO paper_trades (
                id,pair,side,expiry_timeframe,signal_candle_open_utc,
                signal_available_at_utc,observed_at_utc,signal_age_seconds,
                signal_close,signal_reason,score_call,score_put,
                strategy_version,analysis_json,status,updated_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (pair, side, expiry_timeframe, signal_candle_open_utc) DO NOTHING
        """, (
            trade_id, pair, side, timeframe, _paper_iso(candle_open),
            _paper_iso(available_at), _paper_iso(observed_at), round(age_seconds, 3),
            float(signal_data["price"]), str(signal_data["reason"]),
            int(score.get("CALL", 0)), int(score.get("PUT", 0)),
            VERSION, __import__("json").dumps(make_json_safe(signal_data.get("analysis", {})), ensure_ascii=False),
            "SIGNAL", _paper_iso(observed_at),
        ))
        row = _paper_exec(conn, """
            SELECT * FROM paper_trades WHERE pair=? AND side=? AND expiry_timeframe=? AND signal_candle_open_utc=?
        """, (pair, side, timeframe, _paper_iso(candle_open))).fetchone()
        result = _paper_public_trade(_paper_dict(row))
    return _paper_record_scan_result(pair, timeframe, {
        "status": "CAPTURED" if result["id"] == trade_id else "ALREADY_CAPTURED",
        "stored": True, "trade": result,
        "note": "Signal observation only. DEMO fill and exit must be recorded separately; no broker execution.",
    }, candle_time=signal_data["candle_time"], reason=signal_data["reason"], timing=timing)


@app.post("/api/paper/trades/{trade_id}/entry", dependencies=[Depends(_paper_auth)])
def paper_record_entry(trade_id: str, req: PaperEntryRequest):
    """Attach actual observed broker DEMO entry (never reconstruct from M1 OHLC)."""
    _paper_initialize()
    entry_at = _paper_time(_paper_iso(req.entry_at_utc))
    now = _paper_now()
    if entry_at > now + pd.Timedelta(seconds=10):
        raise HTTPException(status_code=422, detail="Entry timestamp cannot be in the future")
    with _paper_db() as conn:
        trade = _paper_fetch(conn, trade_id)
        if trade is None:
            raise HTTPException(status_code=404, detail="Paper signal not found")
        if trade["status"] != "SIGNAL":
            raise HTTPException(status_code=409, detail="Entry already recorded or signal voided")
        available_at = _paper_time(trade["signal_available_at_utc"])
        observed_at = _paper_time(trade["observed_at_utc"])
        if entry_at < available_at or entry_at < observed_at:
            raise HTTPException(status_code=422, detail="Entry must be at/after live signal observation")
        expiry_at = (
            _paper_time(_paper_iso(req.expiry_at_utc))
            if req.expiry_at_utc is not None
            else entry_at + TIMEFRAMES[trade["expiry_timeframe"]]
        )
        if expiry_at <= entry_at:
            raise HTTPException(status_code=422, detail="Expiry must be after entry")
        if expiry_at - entry_at > pd.Timedelta(minutes=15):
            raise HTTPException(status_code=422, detail="Expiry more than 15 minutes after entry is unsupported")
        pip = _pip_size_for_pair(trade["pair"])
        adverse_gap = (float(req.entry_price) - float(trade["signal_close"])) / pip
        if trade["side"] == "PUT":
            adverse_gap = -adverse_gap
        cursor = _paper_exec(conn, """
            UPDATE paper_trades SET
                status='OPEN',entry_at_utc=?,entry_price=?,expiry_at_utc=?,
                stake=?,payout_percent=?,broker=?,entry_adverse_gap_pips=?,
                notes=?,updated_at_utc=?
            WHERE id=? AND status='SIGNAL'
        """, (_paper_iso(entry_at), float(req.entry_price), _paper_iso(expiry_at),
              float(req.stake),float(req.payout_percent),req.broker.strip(),round(adverse_gap,5),
              req.notes,_paper_iso(now),trade_id))
        if cursor.rowcount != 1:
            raise HTTPException(status_code=409, detail="Entry changed concurrently")
        updated = _paper_fetch(conn, trade_id)
    return {"status": "OPEN", "trade": _paper_public_trade(updated),
            "disclaimer": "Manual DEMO fill; entry price is provided by the user, not Twelve Data."}


@app.post("/api/paper/trades/{trade_id}/exit", dependencies=[Depends(_paper_auth)])
def paper_record_exit(trade_id: str, req: PaperExitRequest):
    _paper_initialize()
    exit_at = _paper_time(_paper_iso(req.exit_at_utc))
    if exit_at > _paper_now() + pd.Timedelta(seconds=10):
        raise HTTPException(status_code=422, detail="Exit timestamp cannot be in the future")
    if req.broker_outcome is not None and req.broker_outcome not in ("WIN","LOSS","DRAW"):
        raise HTTPException(status_code=422, detail="broker_outcome must be WIN, LOSS or DRAW")
    with _paper_db() as conn:
        trade = _paper_fetch(conn, trade_id)
        if trade is None:
            raise HTTPException(status_code=404, detail="Paper signal not found")
        if trade["status"] != "OPEN":
            raise HTTPException(status_code=409, detail="Trade is not OPEN")
        entry_at = _paper_time(trade["entry_at_utc"])
        expiry_at = _paper_time(trade["expiry_at_utc"])
        if exit_at <= entry_at:
            raise HTTPException(status_code=422, detail="Exit must occur after entry")
        # Real broker expiry may differ slightly. Keep discrepancy visible.
        discrepancy_seconds = round(float((exit_at - expiry_at).total_seconds()),3)
        price_outcome = _expiry_outcome(trade["side"], float(trade["entry_price"]), float(req.exit_price))
        effective_outcome = req.broker_outcome or price_outcome
        stake = float(trade["stake"])
        payout = float(trade["payout_percent"])
        profit = round((stake * payout / 100 if effective_outcome == "WIN"
                        else -stake if effective_outcome == "LOSS" else 0.0), 6)
        notes = (trade.get("notes") or "")
        if req.notes:
            notes = (notes + " | " + req.notes).strip(" |")[:1000]
        cursor = _paper_exec(conn, """
            UPDATE paper_trades SET
                status='CLOSED',exit_at_utc=?,exit_price=?,price_outcome=?,
                broker_outcome=?,effective_outcome=?,net_profit=?,notes=?,updated_at_utc=?
            WHERE id=? AND status='OPEN'
        """, (_paper_iso(exit_at),float(req.exit_price),price_outcome,req.broker_outcome,
              effective_outcome,profit,notes,_paper_iso(_paper_now()),trade_id))
        if cursor.rowcount != 1:
            raise HTTPException(status_code=409, detail="Exit changed concurrently")
        updated = _paper_fetch(conn, trade_id)
    return {"status": "CLOSED", "trade": _paper_public_trade(updated),
            "expiry_timestamp_error_seconds": discrepancy_seconds,
            "expiry_timestamp_review_required": abs(discrepancy_seconds) > 5,
            "broker_vs_price_outcome_mismatch": bool(req.broker_outcome and req.broker_outcome != price_outcome)}


@app.post("/api/paper/trades/{trade_id}/void", dependencies=[Depends(_paper_auth)])
def paper_void_trade(trade_id: str, req: PaperVoidRequest):
    """Do not delete or rewrite an observation; retain a reasoned audit trail."""
    _paper_initialize()
    with _paper_db() as conn:
        row = _paper_fetch(conn, trade_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Paper trade not found")
        if row["status"] == "VOID":
            raise HTTPException(status_code=409, detail="Already voided")
        _paper_exec(conn, "UPDATE paper_trades SET status='VOID',void_reason=?,updated_at_utc=? WHERE id=? AND status!='VOID'",
                    (req.reason,_paper_iso(_paper_now()),trade_id))
    return {"status": "VOID", "id": trade_id, "reason": req.reason}


@app.get("/api/paper/trades", dependencies=[Depends(_paper_auth)])
def paper_list_trades(limit: int = Query(default=50, ge=1, le=500),
                      offset: int = Query(default=0, ge=0),
                      status: str | None = Query(default=None)):
    _paper_initialize()
    if status is not None and status not in ("SIGNAL","OPEN","CLOSED","VOID"):
        raise HTTPException(status_code=422, detail="Invalid status filter")
    with _paper_db() as conn:
        if status:
            total = _paper_exec(conn,"SELECT COUNT(*) AS n FROM paper_trades WHERE status=?",(status,)).fetchone()["n"]
            rows = _paper_exec(conn,"SELECT * FROM paper_trades WHERE status=? ORDER BY observed_at_utc DESC LIMIT ? OFFSET ?",
                               (status,limit,offset)).fetchall()
        else:
            total = _paper_exec(conn,"SELECT COUNT(*) AS n FROM paper_trades").fetchone()["n"]
            rows = _paper_exec(conn,"SELECT * FROM paper_trades ORDER BY observed_at_utc DESC LIMIT ? OFFSET ?",
                               (limit,offset)).fetchall()
    return {"total":int(total),"returned":len(rows),"trades":[_paper_public_trade(_paper_dict(row)) for row in rows]}


@app.get("/api/paper/trades/{trade_id}", dependencies=[Depends(_paper_auth)])
def paper_get_trade(trade_id: str):
    _paper_initialize()
    with _paper_db() as conn:
        row = _paper_fetch(conn,trade_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Paper trade not found")
    return {"trade": _paper_public_trade(row)}


@app.get("/api/paper/stats", dependencies=[Depends(_paper_auth)])
def paper_stats():
    _paper_initialize()
    with _paper_db() as conn:
        counts = _paper_exec(conn, "SELECT status, COUNT(*) AS n FROM paper_trades GROUP BY status").fetchall()
        closed = [_paper_dict(row) for row in _paper_exec(conn,
            "SELECT side,expiry_timeframe,effective_outcome,price_outcome,broker_outcome,net_profit,stake,entry_adverse_gap_pips FROM paper_trades WHERE status='CLOSED'").fetchall()]
    c = {row["status"]:int(row["n"]) for row in counts}
    wins = sum(x["effective_outcome"]=="WIN" for x in closed)
    losses = sum(x["effective_outcome"]=="LOSS" for x in closed)
    draws = sum(x["effective_outcome"]=="DRAW" for x in closed)
    total_stakes = sum(float(x["stake"]) for x in closed)
    net = sum(float(x["net_profit"]) for x in closed)
    mismatch = sum(bool(x["broker_outcome"] and x["broker_outcome"]!=x["price_outcome"]) for x in closed)
    per_side = {}
    for side in ("CALL","PUT"):
        group = [x for x in closed if x["side"]==side]
        per_side[side] = {"count":len(group),"wins":sum(x["effective_outcome"]=="WIN" for x in group),
                          "losses":sum(x["effective_outcome"]=="LOSS" for x in group),
                          "net_profit":round(sum(float(x["net_profit"]) for x in group), 4)}
    return {"version":VERSION,"mode":"MANUAL_DEMO_FORWARD_TEST","storage":PAPER_DB_TYPE,
            "total":sum(c.values()),"by_status":c,"closed":len(closed),
            "wins":wins,"losses":losses,"draws":draws,
            "win_rate_excluding_draws_percent":round(wins/(wins+losses)*100,2) if wins+losses else None,
            "total_stakes":round(total_stakes,4),"net_profit":round(net,4),
            "return_on_stakes_percent":round(net/total_stakes*100,2) if total_stakes else None,
            "broker_outcome_mismatches":mismatch,"by_direction":per_side,
            "warning":"Forward DEMO observations only; no trades are executed by this API. Small samples have high uncertainty."}


@app.get("/api/paper/export.csv", dependencies=[Depends(_paper_auth)])
def paper_export_csv():
    _paper_initialize()
    with _paper_db() as conn:
        rows = [_paper_public_trade(_paper_dict(row)) for row in _paper_exec(
            conn,"SELECT * FROM paper_trades ORDER BY observed_at_utc ASC").fetchall()]
    fields = ["id","pair","side","expiry_timeframe","signal_candle_open_utc",
              "signal_available_at_utc","observed_at_utc","signal_age_seconds",
              "signal_close","strategy_version","status","entry_at_utc","entry_price",
              "expiry_at_utc","stake","payout_percent","broker","exit_at_utc",
              "exit_price","price_outcome","broker_outcome","effective_outcome",
              "net_profit","entry_adverse_gap_pips","outcome_mismatch","void_reason","notes"]
    data = io.StringIO()
    writer = csv.DictWriter(data, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        # Neutralize spreadsheet-formula injection if a CSV is opened in Excel.
        safe_row = {k: ("'" + str(row.get(k)) if isinstance(row.get(k),str) and row.get(k).startswith(("=","+","-","@","\t","\r")) else row.get(k)) for k in fields}
        writer.writerow(safe_row)
    return Response(content="\ufeff"+data.getvalue(),media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition":"attachment; filename=smc_paper_trades.csv",
                             "Cache-Control":"no-store"})


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
