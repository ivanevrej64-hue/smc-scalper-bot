"""SMC 2.4.13: GitHub Actions runner that invokes one authenticated DEMO scan.

No broker orders. Do not put keys or URLs containing credentials in code.
The runner deliberately performs only ONE pair scan per invocation.
"""

import json
import os
import sys
from datetime import datetime, time, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


BOT_SCAN_URL = "https://smc-scalper-bot-1.onrender.com/api/paper/scan"
ALLOWED_RESPONSES = {"WAIT", "CAPTURED", "ALREADY_CAPTURED", "STALE_SKIPPED"}


def forex_session_open(now_utc: datetime) -> bool:
    """Sunday 17:00 to Friday 17:00 New York time, regardless of DST."""
    ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    day = ny.weekday()  # Monday=0, Sunday=6
    if day in (0, 1, 2, 3):
        return True
    if day == 4:
        return ny.time() < time(17)
    if day == 6:
        return ny.time() >= time(17)
    return False


def run_scan(now_utc=None, post_json=None) -> dict:
    now_utc = now_utc or datetime.now(timezone.utc)
    if not forex_session_open(now_utc):
        return {"status": "MARKET_CLOSED_SKIPPED", "ny_time": now_utc.astimezone(ZoneInfo("America/New_York")).isoformat()}

    key = os.getenv("PAPER_API_KEY", "")
    if len(key) < 16:
        raise RuntimeError("Missing PAPER_API_KEY GitHub Actions secret (min 16 characters)")

    payload = {"pair": "EUR/USD", "timeframe": "1min"}
    if post_json is not None:
        response = post_json(BOT_SCAN_URL, key, payload)
    else:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            BOT_SCAN_URL, data=body, method="POST",
            headers={"Content-Type": "application/json", "X-Paper-Key": key, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=150) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"API scan HTTP status {resp.status}")
                response = json.loads(resp.read(128000).decode("utf-8"))
        except HTTPError as exc:
            # Don't log the full HTTP body. It may include unexpected details.
            raise RuntimeError(f"API scan returned HTTP {exc.code}") from None
        except URLError:
            raise RuntimeError("API scan network error") from None

    status = response.get("status")
    if status not in ALLOWED_RESPONSES:
        raise RuntimeError("API scan returned an unexpected status; see server logs")
    # For WAIT, stored=False is expected: no trade exists. We need a separate
    # DB receipt to know the SCAN itself was persisted. Version 2.4.14 provides it.
    scan_log_id = response.get("scan_log_id")
    if response.get("scan_logged") is not True or not isinstance(scan_log_id, str) or len(scan_log_id) < 16:
        raise RuntimeError(
            "API did not confirm scan log persistence. Deploy server.py 2.4.14 first, "
            "then retry; check Render logs if the server returns HTTP 503."
        )
    result = {
        "status": status,
        "pair": response.get("pair", payload["pair"]),
        "stored": bool(response.get("stored", False)),
        "scan_logged": True,
        "scan_log_id": scan_log_id,
    }
    if status in ("CAPTURED", "ALREADY_CAPTURED"):
        trade = response.get("trade") or {}
        result.update({"signal": trade.get("side"), "candle_time": trade.get("signal_candle_open_utc")})
    if status == "STALE_SKIPPED":
        result.update({"signal_age_seconds": response.get("age_seconds")})
    return result


def main() -> int:
    try:
        result = run_scan()
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"Paper scan failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
