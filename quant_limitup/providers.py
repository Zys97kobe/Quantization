from __future__ import annotations

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import json
import re
from time import sleep

import pandas as pd
import requests


def fetch_akshare_daily_prices(
    out_file: Path,
    start_date: str,
    end_date: str | None = None,
    adjust: str = "qfq",
    limit_symbols: int | None = None,
    pause_seconds: float = 0.12,
) -> pd.DataFrame:
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AkShare is not installed. Install it with: python3 -m pip install akshare"
        ) from exc

    end_date = end_date or datetime.now().strftime("%Y%m%d")
    spot = ak.stock_zh_a_spot_em()
    spot = _normalize_spot(spot)
    if limit_symbols:
        spot = spot.head(limit_symbols)

    rows: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    for item in spot.itertuples(index=False):
        try:
            hist = ak.stock_zh_a_hist(
                symbol=item.raw_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            if hist.empty:
                continue
            rows.append(_normalize_hist(hist, item))
            sleep(pause_seconds)
        except Exception as exc:  # noqa: BLE001 - provider errors need collection.
            failures.append((item.symbol, str(exc)))

    if not rows:
        details = "; ".join(f"{symbol}: {err}" for symbol, err in failures[:5])
        raise RuntimeError(f"No AkShare data fetched. {details}")

    data = pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_file, index=False)

    if failures:
        failure_file = out_file.with_suffix(".failures.csv")
        pd.DataFrame(failures, columns=["symbol", "error"]).to_csv(failure_file, index=False)
    return data


def fetch_tushare_daily_prices(
    out_file: Path,
    start_date: str,
    end_date: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    try:
        import tushare as ts  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TuShare is not installed. Install it with: python3 -m pip install tushare"
        ) from exc

    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Missing TuShare token. Set TUSHARE_TOKEN, pass --token, or create config/tushare_token.txt.")

    end_date = end_date or datetime.now().strftime("%Y%m%d")
    pro = ts.pro_api(token)
    basic = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,market,list_date",
    )
    daily = pro.daily(start_date=start_date, end_date=end_date)
    daily_basic = pro.daily_basic(
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,turnover_rate,circ_mv",
    )
    if daily.empty:
        raise RuntimeError("TuShare returned no daily rows for the requested date range.")

    data = daily.merge(basic, on="ts_code", how="left").merge(
        daily_basic, on=["ts_code", "trade_date"], how="left"
    )
    data = _normalize_tushare(data)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_file, index=False)
    return data


def fetch_sina_daily_prices(
    out_file: Path,
    symbols_file: Path,
    days: int = 260,
    pause_seconds: float = 0.08,
    workers: int = 24,
) -> pd.DataFrame:
    pool = read_symbol_pool(symbols_file)
    rows: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    items = list(pool.itertuples(index=False))
    if workers <= 1:
        for item in items:
            try:
                frame = _fetch_one_sina(item.symbol, item.name, item.board, item.is_st, days)
                if not frame.empty:
                    rows.append(frame)
                sleep(pause_seconds)
            except Exception as exc:  # noqa: BLE001 - external provider errors are collected.
                failures.append((item.symbol, str(exc)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_fetch_one_sina, item.symbol, item.name, item.board, item.is_st, days): item
                for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    frame = future.result()
                    if not frame.empty:
                        rows.append(frame)
                except Exception as exc:  # noqa: BLE001
                    failures.append((item.symbol, str(exc)))

    if not rows:
        detail = "; ".join(f"{symbol}: {err}" for symbol, err in failures[:5])
        raise RuntimeError(f"No Sina data fetched. {detail}")

    data = pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_file, index=False)
    if failures:
        pd.DataFrame(failures, columns=["symbol", "error"]).to_csv(out_file.with_suffix(".failures.csv"), index=False)
    return data


def fetch_sina_candidate_daily_prices(candidates: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """Fetch recent daily bars for review candidates without writing market data files."""
    if candidates.empty:
        return pd.DataFrame()
    required = {"symbol", "name", "board", "is_st"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate pool missing columns: {sorted(missing)}")
    rows = []
    items = list(candidates.drop_duplicates("symbol").itertuples(index=False))
    with ThreadPoolExecutor(max_workers=min(len(items), 16)) as executor:
        futures = {
            executor.submit(_fetch_one_sina, item.symbol, item.name, item.board, item.is_st, days): item
            for item in items
        }
        for future in as_completed(futures):
            try:
                frame = future.result()
            except Exception:  # noqa: BLE001
                continue
            if not frame.empty:
                rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def fetch_sina_minute_bars(
    out_file: Path,
    symbols_file: Path,
    scale: int = 5,
    bars: int = 80,
    pause_seconds: float = 0.05,
    max_symbols: int | None = None,
    workers: int = 24,
) -> pd.DataFrame:
    pool = read_symbol_pool(symbols_file)
    if max_symbols:
        pool = pool.head(max_symbols)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if pool.empty:
        empty = pd.DataFrame(columns=["datetime", "symbol", "open", "high", "low", "close", "volume", "amount"])
        empty.to_csv(out_file, index=False)
        return empty
    rows: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    items = list(pool.itertuples(index=False))
    if workers <= 1:
        for item in items:
            try:
                frame = _fetch_one_sina_minute(item.symbol, scale=scale, bars=bars)
                if not frame.empty:
                    rows.append(frame)
                sleep(pause_seconds)
            except Exception as exc:  # noqa: BLE001
                failures.append((item.symbol, str(exc)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_one_sina_minute, item.symbol, scale, bars): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    frame = future.result()
                    if not frame.empty:
                        rows.append(frame)
                except Exception as exc:  # noqa: BLE001
                    failures.append((item.symbol, str(exc)))

    if not rows:
        detail = "; ".join(f"{symbol}: {err}" for symbol, err in failures[:5])
        if failures:
            pd.DataFrame(failures, columns=["symbol", "error"]).to_csv(out_file.with_suffix(".failures.csv"), index=False)
        raise RuntimeError(f"No Sina minute data fetched. {detail}")
    data = pd.concat(rows, ignore_index=True).sort_values(["symbol", "datetime"])
    data.to_csv(out_file, index=False)
    if failures:
        pd.DataFrame(failures, columns=["symbol", "error"]).to_csv(out_file.with_suffix(".failures.csv"), index=False)
    return data


def sina_minute_market_is_current(
    symbols_file: Path,
    expected_date: str | None = None,
    scale: int = 5,
    bars: int = 2,
    sample_size: int = 12,
) -> bool:
    """Probe diversified symbols without writing files or training a model."""
    pool = read_symbol_pool(symbols_file)
    if pool.empty:
        return False
    step = max(len(pool) // max(sample_size, 1), 1)
    sample = pool.iloc[::step].head(sample_size)
    expected = pd.to_datetime(expected_date).normalize() if expected_date else pd.Timestamp.now().normalize()
    with ThreadPoolExecutor(max_workers=min(len(sample), 12)) as executor:
        futures = [
            executor.submit(_fetch_one_sina_minute, item.symbol, scale, bars)
            for item in sample.itertuples(index=False)
        ]
        for future in as_completed(futures):
            try:
                frame = future.result()
            except Exception:  # noqa: BLE001
                continue
            if frame.empty or "datetime" not in frame.columns:
                continue
            dates = pd.to_datetime(frame["datetime"], errors="coerce").dropna().dt.normalize()
            if bool((dates == expected).any()):
                return True
    return False


def update_sina_stock_pool(out_file: Path, page_size: int = 100) -> pd.DataFrame:
    count_url = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeStockCount"
    list_url = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
    headers = {"User-Agent": "Mozilla/5.0"}
    count_response = requests.get(count_url, params={"node": "hs_a"}, timeout=20, headers=headers)
    count_response.raise_for_status()
    total = int(count_response.text.strip().strip('"'))
    pages = total // page_size + int(total % page_size > 0)

    rows: list[dict] = []
    for page in range(1, pages + 1):
        params = {
            "page": page,
            "num": page_size,
            "sort": "symbol",
            "asc": "1",
            "node": "hs_a",
            "symbol": "",
            "_s_r_a": "page",
        }
        response = requests.get(list_url, params=params, timeout=30, headers=headers)
        response.raise_for_status()
        payload = json.loads(response.text)
        for item in payload:
            symbol = _from_sina_symbol(str(item["symbol"]))
            name = str(item.get("name") or item.get("code") or symbol)
            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "board": _infer_tushare_board(symbol, ""),
                    "is_st": int("ST" in name.upper()),
                }
            )
        sleep(0.05)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["symbol"]).sort_values("symbol")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_file, index=False)
    return frame.reset_index(drop=True)


def read_symbol_pool(path: Path) -> pd.DataFrame:
    if not path.exists():
        write_default_symbol_pool(path)
    frame = pd.read_csv(path)
    required = {"symbol", "name"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Stock pool missing columns: {sorted(missing)}")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["name"] = frame["name"].astype(str)
    if "board" not in frame.columns:
        frame["board"] = frame["symbol"].map(lambda symbol: _infer_tushare_board(symbol, ""))
    if "is_st" not in frame.columns:
        frame["is_st"] = frame["name"].str.contains("ST", case=False, regex=False).astype(int)
    return frame[["symbol", "name", "board", "is_st"]]


def write_default_symbol_pool(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = [
        ("000001.SZ", "平安银行", "main", 0),
        ("000002.SZ", "万科A", "main", 0),
        ("000063.SZ", "中兴通讯", "main", 0),
        ("000333.SZ", "美的集团", "main", 0),
        ("000651.SZ", "格力电器", "main", 0),
        ("000725.SZ", "京东方A", "main", 0),
        ("002230.SZ", "科大讯飞", "main", 0),
        ("002415.SZ", "海康威视", "main", 0),
        ("002594.SZ", "比亚迪", "main", 0),
        ("300059.SZ", "东方财富", "chinext", 0),
        ("300274.SZ", "阳光电源", "chinext", 0),
        ("300750.SZ", "宁德时代", "chinext", 0),
        ("600000.SH", "浦发银行", "main", 0),
        ("600036.SH", "招商银行", "main", 0),
        ("600050.SH", "中国联通", "main", 0),
        ("600519.SH", "贵州茅台", "main", 0),
        ("600900.SH", "长江电力", "main", 0),
        ("601318.SH", "中国平安", "main", 0),
        ("601398.SH", "工商银行", "main", 0),
        ("601899.SH", "紫金矿业", "main", 0),
        ("603259.SH", "药明康德", "main", 0),
        ("688111.SH", "金山办公", "star", 0),
        ("688981.SH", "中芯国际", "star", 0),
    ]
    pd.DataFrame(defaults, columns=["symbol", "name", "board", "is_st"]).to_csv(path, index=False)


def _normalize_spot(spot: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "代码": "raw_code",
        "名称": "name",
        "总市值": "total_mkt_cap",
        "流通市值": "free_float_mkt_cap",
    }
    frame = spot.rename(columns=rename).copy()
    required = {"raw_code", "name"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"AkShare spot data missing columns: {sorted(missing)}")

    frame["raw_code"] = frame["raw_code"].astype(str).str.zfill(6)
    frame["symbol"] = frame["raw_code"].map(_to_exchange_symbol)
    frame["board"] = frame["raw_code"].map(_infer_board)
    frame["is_st"] = frame["name"].astype(str).str.contains("ST", case=False, regex=False).astype(int)
    if "free_float_mkt_cap" not in frame.columns:
        frame["free_float_mkt_cap"] = pd.NA
    return frame[["raw_code", "symbol", "name", "board", "is_st", "free_float_mkt_cap"]]


def _normalize_hist(hist: pd.DataFrame, item: object) -> pd.DataFrame:
    rename = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }
    frame = hist.rename(columns=rename).copy()
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"AkShare historical data missing columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame["symbol"] = item.symbol
    frame["name"] = item.name
    frame["board"] = item.board
    frame["is_st"] = item.is_st
    if "turnover" not in frame.columns:
        frame["turnover"] = 0.0
    frame["turnover"] = pd.to_numeric(frame["turnover"], errors="coerce").fillna(0.0) / 100
    free_float = pd.to_numeric(pd.Series([item.free_float_mkt_cap]), errors="coerce").iloc[0]
    if pd.isna(free_float) or free_float <= 0:
        free_float = frame["close"].astype(float) * frame["volume"].astype(float) / frame["turnover"].replace(0, pd.NA)
        free_float = float(pd.to_numeric(free_float, errors="coerce").median())
    frame["free_float_mkt_cap"] = free_float
    cols = [
        "date",
        "symbol",
        "name",
        "board",
        "is_st",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
        "free_float_mkt_cap",
    ]
    return frame[cols]


def _normalize_tushare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    data["symbol"] = data["ts_code"]
    data["board"] = data.apply(lambda row: _infer_tushare_board(row["symbol"], row.get("market")), axis=1)
    data["is_st"] = data["name"].astype(str).str.contains("ST", case=False, regex=False).astype(int)
    data["volume"] = pd.to_numeric(data["vol"], errors="coerce").fillna(0) * 100
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce").fillna(0) * 1000
    data["turnover"] = pd.to_numeric(data["turnover_rate"], errors="coerce").fillna(0) / 100
    data["free_float_mkt_cap"] = pd.to_numeric(data["circ_mv"], errors="coerce") * 10_000
    fallback_cap = pd.to_numeric(data["close"], errors="coerce") * data["volume"] / data["turnover"].replace(0, pd.NA)
    data["free_float_mkt_cap"] = data["free_float_mkt_cap"].fillna(fallback_cap).fillna(0)
    cols = [
        "date",
        "symbol",
        "name",
        "board",
        "is_st",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
        "free_float_mkt_cap",
    ]
    return data[cols].sort_values(["symbol", "date"]).reset_index(drop=True)


def _fetch_one_sina(symbol: str, name: str, board: str, is_st: int, days: int) -> pd.DataFrame:
    sina_symbol = _to_sina_symbol(symbol)
    url = "https://quotes.sina.cn/cn/api/jsonp.php/var%20_data=/CN_MarketData.getKLineData"
    params = {"symbol": sina_symbol, "scale": "240", "ma": "no", "datalen": str(days)}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    match = re.search(r"var _data=\((\[.*\])\)", response.text, flags=re.S)
    if not match:
        raise RuntimeError("Unexpected Sina response")
    payload = json.loads(match.group(1))
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    frame["date"] = frame["day"]
    frame["symbol"] = symbol
    frame["name"] = name
    frame["board"] = board
    frame["is_st"] = int(is_st)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    avg_price = (frame["open"] + frame["close"]) / 2
    frame["amount"] = frame["volume"] * avg_price
    frame["turnover"] = 0.0
    frame["free_float_mkt_cap"] = frame["close"] * frame["volume"].rolling(20, min_periods=1).mean() * 100
    cols = [
        "date",
        "symbol",
        "name",
        "board",
        "is_st",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
        "free_float_mkt_cap",
    ]
    return frame[cols].dropna().copy()


def _fetch_one_sina_minute(symbol: str, scale: int, bars: int) -> pd.DataFrame:
    sina_symbol = _to_sina_symbol(symbol)
    url = "https://quotes.sina.cn/cn/api/jsonp.php/var%20_data=/CN_MarketData.getKLineData"
    params = {"symbol": sina_symbol, "scale": str(scale), "ma": "no", "datalen": str(bars)}
    response = requests.get(url, params=params, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    match = re.search(r"var _data=\((\[.*\])\)", response.text, flags=re.S)
    if not match:
        raise RuntimeError("Unexpected Sina minute response")
    payload = json.loads(match.group(1))
    frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    dt_col = "day" if "day" in frame.columns else "datetime"
    frame["datetime"] = pd.to_datetime(frame[dt_col])
    frame["symbol"] = symbol
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    avg_price = (frame["open"] + frame["close"]) / 2
    frame["amount"] = frame["volume"] * avg_price
    return frame[["datetime", "symbol", "open", "high", "low", "close", "volume", "amount"]].dropna()


def _to_exchange_symbol(raw_code: str) -> str:
    if raw_code.startswith(("5", "6", "9")):
        return f"{raw_code}.SH"
    if raw_code.startswith(("8", "4")):
        return f"{raw_code}.BJ"
    return f"{raw_code}.SZ"


def _infer_board(raw_code: str) -> str:
    if raw_code.startswith("688"):
        return "star"
    if raw_code.startswith(("300", "301")):
        return "chinext"
    if raw_code.startswith(("8", "4")):
        return "bse"
    return "main"


def _infer_tushare_board(ts_code: str, market: object) -> str:
    code = str(ts_code).split(".")[0]
    market_text = "" if pd.isna(market) else str(market)
    if "科创" in market_text or code.startswith("688"):
        return "star"
    if "创业" in market_text or code.startswith(("300", "301")):
        return "chinext"
    if "北交" in market_text or ts_code.endswith(".BJ") or code.startswith(("8", "4")):
        return "bse"
    return "main"


def _to_sina_symbol(symbol: str) -> str:
    code, exchange = symbol.split(".")
    if exchange.upper() == "SH":
        return f"sh{code}"
    if exchange.upper() == "SZ":
        return f"sz{code}"
    if exchange.upper() == "BJ":
        return f"bj{code}"
    raise RuntimeError(f"Unsupported exchange for Sina symbol: {symbol}")


def _from_sina_symbol(symbol: str) -> str:
    prefix = symbol[:2].lower()
    code = symbol[2:]
    if prefix == "sh":
        return f"{code}.SH"
    if prefix == "sz":
        return f"{code}.SZ"
    if prefix == "bj":
        return f"{code}.BJ"
    raise RuntimeError(f"Unsupported Sina symbol: {symbol}")
