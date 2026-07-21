"""Exploratory event study for public semiconductor/AI information sources.

This module deliberately does not connect to the live or paper-trading executor.
It freezes public embedded-profile snapshots, extracts only explicit directional
language with deterministic rules, and tests subsequent adjusted-close returns.
The public profile feed can be incomplete or selected, so its results are a
research screen rather than production evidence.
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from research.market_data import fetch_yahoo_adjusted_close


PROFILE_URL = (
    "https://cdn.syndication.twimg.com/srv/timeline-profile/screen-name/{handle}"
)
DEFAULT_OUTPUT_DIR = Path("output/source-signals")
DEFAULT_BENCHMARKS = ("SOXX", "QQQ")
DEFAULT_HORIZONS = (5, 20, 60)
ROUND_TRIP_COST_BPS = 20.0
SOURCE_COOLDOWN_DAYS = 30
HOUSE_COOLDOWN_DAYS = 7


@dataclass(frozen=True)
class SourceSpec:
    handle: str
    house: str
    role: str
    directional_candidate: bool


SOURCES: Tuple[SourceSpec, ...] = (
    SourceSpec("aleabitoreddit", "Serenity", "idea", True),
    SourceSpec("jukan05", "Citrini", "idea", True),
    SourceSpec("zephyr_z9", "Citrini", "idea", True),
    SourceSpec("PhotonCap", "Photon Capital", "mechanism", True),
    SourceSpec("dnystedt", "independent", "news_validation", False),
    SourceSpec("SemiAnalysis_", "SemiAnalysis", "mechanism", False),
    SourceSpec("SKundojjala", "SemiAnalysis", "industry_validation", False),
    SourceSpec("jaygoldberg", "Digits to Dollars", "contrarian_validation", False),
    SourceSpec("FoolAllTheTime", "Fabricated Knowledge", "contrarian_validation", False),
    SourceSpec("mingchikuo", "TF International", "supply_chain_validation", False),
    SourceSpec("tphuang", "independent", "industry_validation", False),
)


# Company-name matching is intentionally limited to unambiguous, listed names.
# Cashtags remain the primary extraction path.
COMPANY_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "NVDA": ("NVIDIA",),
    "INTC": ("Intel",),
    "AMD": ("Advanced Micro Devices",),
    "AAPL": ("Apple",),
    "MSFT": ("Microsoft",),
    "GOOGL": ("Alphabet", "Google"),
    "AMZN": ("Amazon",),
    "META": ("Meta Platforms", "Facebook"),
    "AVGO": ("Broadcom",),
    "QCOM": ("Qualcomm",),
    "TSM": ("TSMC", "Taiwan Semiconductor"),
    "ASML": ("ASML",),
    "MU": ("Micron",),
    "MRVL": ("Marvell",),
    "ARM": ("Arm Holdings",),
    "SMCI": ("Super Micro",),
    "PLTR": ("Palantir",),
    "ORCL": ("Oracle",),
    "CRWV": ("CoreWeave",),
    "IBM": ("International Business Machines",),
    "DELL": ("Dell",),
    "HPE": ("Hewlett Packard Enterprise",),
    "VRT": ("Vertiv",),
    "ANET": ("Arista Networks",),
    "LRCX": ("Lam Research",),
    "AMAT": ("Applied Materials",),
    "KLAC": ("KLA Corporation",),
    "CDNS": ("Cadence Design",),
    "SNPS": ("Synopsys",),
    "WDC": ("Western Digital",),
    "STX": ("Seagate",),
    "TXN": ("Texas Instruments",),
    "ADI": ("Analog Devices",),
    "NXPI": ("NXP Semiconductors",),
    "MCHP": ("Microchip Technology",),
    "TSLA": ("Tesla",),
    "SNOW": ("Snowflake",),
    "NOW": ("ServiceNow",),
    "CRM": ("Salesforce",),
}

POSITIVE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("buy", r"\bbuy\s+(?:the\s+)?(?:stock|shares|\$[A-Z])"),
    ("long", r"\b(?:go|going|went)?\s*long\b"),
    ("bullish", r"\bbullish\b"),
    (
        "positive_rating",
        r"\b(?:overweight|outperform)\b|\bupgrade[sd]?\s+(?:to\s+)?(?:buy|outperform|overweight)\b",
    ),
    ("upside", r"\bupside\b"),
    ("beneficiary", r"\b(?:beneficiar(?:y|ies)|benefit(?:s|ed|ing)?)\b"),
    ("tailwind", r"\btailwinds?\b"),
    ("winner", r"\bwinners?\b"),
    ("positive", r"\bpositive\b"),
    ("valuation_positive", r"\b(?:undervalued|compelling|attractive)\b"),
)
NEGATIVE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("sell", r"\bsell\s+(?:the\s+)?(?:stock|shares|\$[A-Z])"),
    ("short", r"\b(?:go|going|went)?\s*short\b"),
    ("bearish", r"\bbearish\b"),
    (
        "negative_rating",
        r"\b(?:underweight|underperform)\b|\bdowngrade[sd]?\s+(?:to\s+)?(?:sell|underperform|underweight)\b",
    ),
    ("downside", r"\bdownside\b"),
    ("headwind", r"\bheadwinds?\b"),
    ("loser", r"\blosers?\b"),
    ("avoid", r"\bavoid(?:ed|ing)?\b"),
    ("negative", r"\bnegative\b"),
    ("valuation_negative", r"\b(?:overvalued|bubble)\b"),
)


def parse_embedded_profile(page_html: str) -> List[Dict[str, object]]:
    """Return normalized tweet records from one embedded-profile HTML page."""
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        page_html,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Embedded profile did not contain __NEXT_DATA__")
    payload = json.loads(html_module.unescape(match.group(1)))
    entries = (
        payload.get("props", {})
        .get("pageProps", {})
        .get("timeline", {})
        .get("entries", [])
    )
    records: List[Dict[str, object]] = []
    for entry in entries:
        tweet = entry.get("content", {}).get("tweet") or {}
        tweet_id = tweet.get("id_str") or tweet.get("rest_id")
        text = tweet.get("full_text")
        created_at = tweet.get("created_at")
        if not tweet_id or not text or not created_at:
            continue
        records.append(
            {
                "tweet_id": str(tweet_id),
                "created_at": str(created_at),
                "text": str(text),
                "conversation_id": str(tweet.get("conversation_id_str") or ""),
                "is_reply": bool(tweet.get("in_reply_to_status_id_str")),
                "is_retweet": str(text).startswith("RT @"),
            }
        )
    return records


def fetch_profile_snapshot(
    source: SourceSpec,
    session: Optional[requests.Session] = None,
    retries: int = 2,
) -> Dict[str, object]:
    """Fetch one public embedded profile and return a frozen JSON-ready record."""
    client = session or requests.Session()
    url = PROFILE_URL.format(handle=source.handle)
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = client.get(
                url,
                params={"dnt": "true"},
                headers={"User-Agent": "Mozilla/5.0 source-signal-research/1.0"},
                timeout=30,
            )
            if response.status_code == 429 and attempt < retries:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            posts = parse_embedded_profile(response.text)
            return {
                "source": asdict(source),
                "profile_url": f"https://x.com/{source.handle}",
                "retrieval_url": url,
                "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "sampling_note": (
                    "Public embedded-profile snapshot; completeness and selection "
                    "rules are not documented and may vary by account."
                ),
                "posts": posts,
            }
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not fetch @{source.handle}: {last_error}")


def collect_snapshots(
    output_dir: Path,
    sources: Sequence[SourceSpec] = SOURCES,
    refresh: bool = False,
    request_delay_seconds: float = 1.0,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    """Collect or reuse one snapshot per source without failing the full batch."""
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshots: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    session = requests.Session()
    for index, source in enumerate(sources):
        path = snapshot_dir / f"{source.handle}.json"
        if path.exists() and not refresh:
            snapshots.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        try:
            snapshot = fetch_profile_snapshot(source, session=session)
            path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            snapshots.append(snapshot)
        except RuntimeError as exc:
            failure = {"handle": source.handle, "error": str(exc)}
            if path.exists():
                snapshots.append(json.loads(path.read_text(encoding="utf-8")))
                failure["fallback"] = "used_cached_snapshot"
            failures.append(failure)
        if index < len(sources) - 1:
            time.sleep(request_delay_seconds)
    return snapshots, failures


def extract_tickers(text: str) -> List[str]:
    """Extract cashtags and conservative company-name aliases."""
    tickers = {
        match.upper()
        for match in re.findall(r"(?<!\w)\$([A-Za-z][A-Za-z0-9.\-]{0,9})\b", text)
    }
    for ticker, aliases in COMPANY_ALIASES.items():
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.I) for alias in aliases):
            tickers.add(ticker)
    return sorted(tickers)


def classify_direction(text: str) -> Tuple[int, str]:
    """Return +1, -1, or 0 only when directional language is unambiguous."""
    positive = [name for name, pattern in POSITIVE_PATTERNS if re.search(pattern, text, re.I)]
    negative = [name for name, pattern in NEGATIVE_PATTERNS if re.search(pattern, text, re.I)]
    if positive and not negative:
        return 1, ",".join(positive)
    if negative and not positive:
        return -1, ",".join(negative)
    if positive and negative:
        return 0, "mixed_language"
    return 0, "no_explicit_direction"


def classify_ticker_direction(text: str, ticker: str) -> Tuple[int, str]:
    """Classify only language close to this ticker's mention.

    This avoids turning narrative verbs such as "sold a company to Microsoft"
    or "buying chips from Broadcom" into stock recommendations.
    """
    aliases = COMPANY_ALIASES.get(ticker, ())
    mention_patterns = [rf"(?<!\w)\${re.escape(ticker)}\b"]
    mention_patterns.extend(
        rf"(?<!\w){re.escape(alias)}(?!\w)" for alias in aliases
    )
    mention = "(?:" + "|".join(mention_patterns) + ")"
    segments = re.split(r"(?:\n+|(?<=[.!?])\s+|;|&gt;)", text)
    directions: List[int] = []
    reasons: List[str] = []
    positive_words = (
        r"bullish|positive|upside|beneficiar(?:y|ies)|benefit(?:s|ed|ing)?|"
        r"overweight|outperform|winner"
    )
    negative_words = r"bearish|negative|downside|underweight|underperform|loser"
    for segment in segments:
        if not re.search(mention, segment, re.I):
            continue
        segment_tickers = extract_tickers(segment)
        if len(segment_tickers) > 1 and re.search(r"\bthan\b", segment, re.I):
            continue
        positive_sentiment = bool(
            re.search(rf"\b(?:{positive_words})\b.{{0,45}}{mention}", segment, re.I)
            or re.search(rf"{mention}.{{0,45}}\b(?:{positive_words})\b", segment, re.I)
        )
        negative_sentiment = bool(
            re.search(rf"\b(?:{negative_words})\b.{{0,45}}{mention}", segment, re.I)
            or re.search(rf"{mention}.{{0,45}}\b(?:{negative_words})\b", segment, re.I)
        )
        positive_trade = bool(
            re.search(
                rf"\b(?:buy|long)\s+(?:(?:the\s+)?(?:stock|shares)(?:\s+of)?\s+)?{mention}",
                segment,
                re.I,
            )
        )
        negative_trade = bool(
            re.search(
                rf"\b(?:sell|short|avoid)\s+(?:(?:the\s+)?(?:stock|shares)(?:\s+of)?\s+)?{mention}",
                segment,
                re.I,
            )
        )
        positive = positive_sentiment or positive_trade
        negative = negative_sentiment or negative_trade
        if positive and not negative:
            directions.append(1)
            reasons.append("ticker_local_positive")
        elif negative and not positive:
            directions.append(-1)
            reasons.append("ticker_local_negative")
    unique = set(directions)
    if unique == {1}:
        return 1, ",".join(sorted(set(reasons)))
    if unique == {-1}:
        return -1, ",".join(sorted(set(reasons)))
    if len(unique) > 1:
        return 0, "mixed_ticker_language"
    return 0, "no_ticker_local_direction"


def snapshots_to_posts(snapshots: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for snapshot in snapshots:
        source = snapshot["source"]
        assert isinstance(source, Mapping)
        for post in snapshot.get("posts", []):
            assert isinstance(post, Mapping)
            created_at = pd.to_datetime(post["created_at"], utc=True)
            text = str(post["text"])
            direction, reason = classify_direction(text)
            tickers = extract_tickers(text)
            rows.append(
                {
                    "handle": source["handle"],
                    "house": source["house"],
                    "role": source["role"],
                    "directional_candidate": bool(source["directional_candidate"]),
                    "tweet_id": post["tweet_id"],
                    "created_at_utc": created_at,
                    "text": text,
                    "is_reply": bool(post.get("is_reply")),
                    "is_retweet": bool(post.get("is_retweet")),
                    "direction": direction,
                    "direction_reason": reason,
                    "tickers": tickers,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["created_at_utc", "handle", "tweet_id"])


def build_events(posts: pd.DataFrame) -> pd.DataFrame:
    """Explode directional candidate posts and mark cooldown-independent events."""
    columns = [
        "handle",
        "house",
        "tweet_id",
        "created_at_utc",
        "ticker",
        "direction",
        "direction_reason",
        "is_reply",
        "is_retweet",
    ]
    if posts.empty:
        return pd.DataFrame(columns=columns + ["source_independent", "house_independent"])
    eligible = posts[
        posts["directional_candidate"] & posts["tickers"].map(bool)
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=columns + ["source_independent", "house_independent"])
    events = eligible.explode("tickers").rename(columns={"tickers": "ticker"})
    ticker_labels = events.apply(
        lambda row: classify_ticker_direction(str(row["text"]), str(row["ticker"])),
        axis=1,
    )
    events["direction"] = ticker_labels.map(lambda label: label[0])
    events["direction_reason"] = ticker_labels.map(lambda label: label[1])
    events = events[events["direction"].isin((-1, 1))]
    if events.empty:
        return pd.DataFrame(columns=columns + ["source_independent", "house_independent"])
    events = events[columns].sort_values("created_at_utc").reset_index(drop=True)
    events["source_independent"] = _cooldown_flags(
        events,
        group_columns=("handle", "ticker", "direction"),
        cooldown_days=SOURCE_COOLDOWN_DAYS,
    )
    events["house_independent"] = _cooldown_flags(
        events,
        group_columns=("house", "ticker", "direction"),
        cooldown_days=HOUSE_COOLDOWN_DAYS,
    )
    return events


def _cooldown_flags(
    events: pd.DataFrame,
    group_columns: Sequence[str],
    cooldown_days: int,
) -> pd.Series:
    flags = pd.Series(False, index=events.index, dtype=bool)
    last_dates: Dict[Tuple[object, ...], pd.Timestamp] = {}
    for index, row in events.sort_values("created_at_utc").iterrows():
        key = tuple(row[column] for column in group_columns)
        event_date = pd.Timestamp(row["created_at_utc"])
        last_date = last_dates.get(key)
        if last_date is None or event_date - last_date >= pd.Timedelta(days=cooldown_days):
            flags.loc[index] = True
            last_dates[key] = event_date
    return flags


def fetch_event_prices(
    events: pd.DataFrame,
    benchmarks: Sequence[str] = DEFAULT_BENCHMARKS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> Tuple[Dict[str, pd.Series], List[Dict[str, str]]]:
    if events.empty:
        return {}, []
    start = pd.Timestamp(events["created_at_utc"].min()).tz_convert(None).normalize()
    end = pd.Timestamp.utcnow().tz_localize(None).normalize() + pd.Timedelta(days=1)
    start -= pd.Timedelta(days=10)
    max_horizon = max(horizons)
    failures: List[Dict[str, str]] = []
    prices: Dict[str, pd.Series] = {}
    for ticker in sorted(set(events["ticker"]) | set(benchmarks)):
        try:
            series = fetch_yahoo_adjusted_close(ticker, start, end)
            if len(series) <= max_horizon:
                raise RuntimeError(f"only {len(series)} daily rows")
            prices[ticker] = series
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            failures.append({"ticker": ticker, "error": str(exc)})
    return prices, failures


def run_event_study(
    events: pd.DataFrame,
    prices: Mapping[str, pd.Series],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    cost_bps: float = ROUND_TRIP_COST_BPS,
) -> pd.DataFrame:
    """Calculate next-session-close event returns with no same-day entry."""
    rows: List[Dict[str, object]] = []
    cost_rate = cost_bps / 10_000.0
    for _, event in events.iterrows():
        ticker = str(event["ticker"])
        if ticker not in prices or any(name not in prices for name in DEFAULT_BENCHMARKS):
            continue
        event_date = pd.Timestamp(event["created_at_utc"]).tz_convert(None).normalize()
        stock = prices[ticker].dropna().sort_index()
        entry_candidates = stock.index[stock.index > event_date]
        if entry_candidates.empty:
            continue
        entry_date = entry_candidates[0]
        entry_position = stock.index.get_loc(entry_date)
        row = dict(event)
        row.update(
            {
                "entry_date": entry_date,
                "entry_price": float(stock.loc[entry_date]),
                "cost_bps": float(cost_bps),
            }
        )
        for horizon in horizons:
            exit_position = entry_position + horizon
            exit_date = stock.index[exit_position] if exit_position < len(stock) else pd.NaT
            exit_price = (
                float(stock.iloc[exit_position]) if exit_position < len(stock) else np.nan
            )
            stock_return = _forward_return(stock, entry_date, entry_position, horizon)
            row[f"exit_date_{horizon}d"] = exit_date
            row[f"exit_price_{horizon}d"] = exit_price
            row[f"stock_return_{horizon}d"] = stock_return
            for benchmark in DEFAULT_BENCHMARKS:
                benchmark_return = _return_between_dates(
                    prices[benchmark], entry_date, exit_date
                )
                row[f"{benchmark.lower()}_return_{horizon}d"] = benchmark_return
                if pd.isna(stock_return) or pd.isna(benchmark_return):
                    alpha = np.nan
                else:
                    alpha = int(event["direction"]) * (stock_return - benchmark_return)
                row[f"net_{benchmark.lower()}_alpha_{horizon}d"] = (
                    alpha - cost_rate if not pd.isna(alpha) else np.nan
                )
            row[f"net_signal_return_{horizon}d"] = (
                int(event["direction"]) * stock_return - cost_rate
                if not pd.isna(stock_return)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _forward_return(
    series: pd.Series,
    entry_date: pd.Timestamp,
    entry_position: int,
    horizon: int,
) -> float:
    exit_position = entry_position + horizon
    if exit_position >= len(series):
        return np.nan
    return float(series.iloc[exit_position] / series.loc[entry_date] - 1.0)


def _return_between_dates(
    series: pd.Series,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> float:
    if pd.isna(exit_date):
        return np.nan
    series = series.dropna().sort_index()
    entry_candidates = series.index[series.index >= entry_date]
    exit_candidates = series.index[series.index >= exit_date]
    if entry_candidates.empty or exit_candidates.empty:
        return np.nan
    benchmark_entry = entry_candidates[0]
    benchmark_exit = exit_candidates[0]
    return float(series.loc[benchmark_exit] / series.loc[benchmark_entry] - 1.0)


def summarize_results(
    results: pd.DataFrame,
    horizon: int = 20,
    seed: int = 20260720,
) -> pd.DataFrame:
    metric = f"net_soxx_alpha_{horizon}d"
    columns = [
        "handle",
        "events",
        "tickers",
        "span_days",
        "mean_net_soxx_alpha",
        "median_net_soxx_alpha",
        "win_rate",
        "bootstrap_mean_ci_low",
        "bootstrap_mean_ci_high",
        "max_ticker_share",
    ]
    if results.empty or metric not in results:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(seed)
    groups: List[Tuple[str, pd.DataFrame]] = [
        (handle, group[group["source_independent"]])
        for handle, group in results.groupby("handle")
    ]
    groups.append(("ALL_INDEPENDENT", results[results["house_independent"]]))
    for handle, group in groups:
        valid = group.dropna(subset=[metric]).copy()
        if valid.empty:
            continue
        values = valid[metric].to_numpy(dtype=float)
        samples = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
        ticker_share = valid["ticker"].value_counts(normalize=True).max()
        rows.append(
            {
                "handle": handle,
                "events": int(len(valid)),
                "tickers": int(valid["ticker"].nunique()),
                "span_days": int(
                    (valid["created_at_utc"].max() - valid["created_at_utc"].min()).days
                ),
                "mean_net_soxx_alpha": float(values.mean()),
                "median_net_soxx_alpha": float(np.median(values)),
                "win_rate": float((values > 0).mean()),
                "bootstrap_mean_ci_low": float(np.quantile(samples, 0.025)),
                "bootstrap_mean_ci_high": float(np.quantile(samples, 0.975)),
                "max_ticker_share": float(ticker_share),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_horizon_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    independent = (
        results[results["house_independent"]].copy() if not results.empty else results
    )
    for horizon in DEFAULT_HORIZONS:
        metric = f"net_soxx_alpha_{horizon}d"
        if metric not in independent:
            continue
        values = independent[metric].dropna().astype(float)
        if values.empty:
            continue
        rows.append(
            {
                "horizon_trading_days": horizon,
                "events": int(len(values)),
                "mean_net_soxx_alpha_0bps": float((values + 0.002).mean()),
                "mean_net_soxx_alpha_20bps": float(values.mean()),
                "mean_net_soxx_alpha_50bps": float((values - 0.003).mean()),
                "median_net_soxx_alpha_20bps": float(values.median()),
                "win_rate_20bps": float((values > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_coverage(
    snapshots: Sequence[Mapping[str, object]],
    posts: pd.DataFrame,
    events: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    snapshot_by_handle = {
        str(snapshot["source"]["handle"]): snapshot for snapshot in snapshots
    }
    rows: List[Dict[str, object]] = []
    for source in SOURCES:
        source_posts = posts[posts["handle"] == source.handle] if not posts.empty else posts
        source_events = events[events["handle"] == source.handle] if not events.empty else events
        source_results = results[results["handle"] == source.handle] if not results.empty else results
        snapshot = snapshot_by_handle.get(source.handle)
        rows.append(
            {
                **asdict(source),
                "snapshot_available": snapshot is not None,
                "retrieved_at_utc": snapshot.get("retrieved_at_utc") if snapshot else None,
                "snapshot_posts": int(len(source_posts)),
                "dated_min": (
                    source_posts["created_at_utc"].min().isoformat()
                    if not source_posts.empty
                    else None
                ),
                "dated_max": (
                    source_posts["created_at_utc"].max().isoformat()
                    if not source_posts.empty
                    else None
                ),
                "directional_events": int(len(source_events)),
                "priced_20d_events": int(
                    source_results["net_soxx_alpha_20d"].notna().sum()
                    if "net_soxx_alpha_20d" in source_results
                    else 0
                ),
                "sampling_completeness": "unknown",
            }
        )
    return pd.DataFrame(rows)


def build_gate(summary: pd.DataFrame) -> Dict[str, object]:
    row = summary[summary["handle"] == "ALL_INDEPENDENT"]
    if row.empty:
        metrics = {
            "events": 0,
            "tickers": 0,
            "span_days": 0,
            "median_net_soxx_alpha": None,
            "win_rate": None,
            "bootstrap_mean_ci_low": None,
            "max_ticker_share": None,
        }
    else:
        metrics = row.iloc[0].to_dict()
    checks = {
        "at_least_20_events": metrics["events"] >= 20,
        "at_least_4_tickers": metrics["tickers"] >= 4,
        "at_least_180_days": metrics["span_days"] >= 180,
        "positive_median_net_soxx_alpha": (
            metrics["median_net_soxx_alpha"] is not None
            and metrics["median_net_soxx_alpha"] > 0
        ),
        "win_rate_at_least_55pct": (
            metrics["win_rate"] is not None and metrics["win_rate"] >= 0.55
        ),
        "positive_bootstrap_mean_ci": (
            metrics["bootstrap_mean_ci_low"] is not None
            and metrics["bootstrap_mean_ci_low"] > 0
        ),
        "max_ticker_share_at_most_35pct": (
            metrics["max_ticker_share"] is not None
            and metrics["max_ticker_share"] <= 0.35
        ),
    }
    passed = all(checks.values())
    return {
        "status": "RESEARCH_ONLY_PASS" if passed else "INSUFFICIENT_DATA",
        "production_signal_enabled": False,
        "decision": (
            "Eligible for a separate paper overlay review; still not connected to execution."
            if passed
            else "Keep as a research/validation feed; do not alter allocator weights."
        ),
        "checks": checks,
        "metrics": metrics,
        "limitations": [
            "Public embedded-profile sampling completeness is unknown.",
            "Rule-based direction labels can miss context, sarcasm, and ticker-specific clauses.",
            "Source selection is retrospective and therefore subject to selection bias.",
            "Adjusted-close event returns omit borrow availability, taxes, and intraday fills.",
            "A positive event study would not establish compatibility with the macro allocator.",
        ],
    }


def save_outputs(
    output_dir: Path,
    snapshots: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, str]],
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    posts = snapshots_to_posts(snapshots)
    events = build_events(posts)
    prices, price_failures = fetch_event_prices(events)
    results = run_event_study(events, prices)
    summary = summarize_results(results)
    horizon_summary = build_horizon_summary(results)
    coverage = build_coverage(snapshots, posts, events, results)
    gate = build_gate(summary)

    posts_export = posts.copy()
    if not posts_export.empty:
        posts_export["tickers"] = posts_export["tickers"].map(json.dumps)
    posts_export.to_csv(output_dir / "posts.csv", index=False)
    events.to_csv(output_dir / "events.csv", index=False)
    results.to_csv(output_dir / "event_returns.csv", index=False)
    summary.to_csv(output_dir / "source_summary.csv", index=False)
    horizon_summary.to_csv(output_dir / "horizon_summary.csv", index=False)
    coverage.to_csv(output_dir / "coverage.csv", index=False)
    manifest = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": {
            "entry": "first adjusted close strictly after the post calendar date",
            "horizons_trading_days": list(DEFAULT_HORIZONS),
            "benchmarks": list(DEFAULT_BENCHMARKS),
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "source_cooldown_days": SOURCE_COOLDOWN_DAYS,
            "house_cooldown_days": HOUSE_COOLDOWN_DAYS,
        },
        "snapshot_failures": list(failures),
        "price_failures": price_failures,
        "gate": gate,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "posts": posts,
        "events": events,
        "results": results,
        "summary": summary,
        "horizon_summary": horizon_summary,
        "coverage": coverage,
        "manifest": manifest,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace cached public-profile snapshots before running the study.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    snapshots, failures = collect_snapshots(args.output_dir, refresh=args.refresh)
    artifacts = save_outputs(args.output_dir, snapshots, failures)
    gate = artifacts["manifest"]["gate"]
    print(f"Snapshots: {len(snapshots)}/{len(SOURCES)}")
    print(f"Directional events: {len(artifacts['events'])}")
    print(f"Priced events: {len(artifacts['results'])}")
    print(f"Gate: {gate['status']}")
    print(gate["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
