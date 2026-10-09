"""Accumulate minute bars without replacing earlier complete sessions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def minute_history_path(snapshot_file: Path) -> Path:
    return snapshot_file.with_name(f"{snapshot_file.stem}.history.csv")


def _prepare_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve defects within one fetch before merging overlapping fetches."""
    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data.dropna(subset=["datetime", "symbol"])
    duplicate = data.duplicated(["symbol", "datetime"], keep=False)
    if "minute_duplicate" in data:
        # CSV round trips may produce booleans, numbers, or strings.
        recorded = data["minute_duplicate"].astype(str).str.lower().isin(["true", "1", "1.0"])
        duplicate |= recorded
    data["minute_duplicate"] = duplicate
    return data


def update_minute_history(snapshot_file: Path, current: pd.DataFrame) -> None:
    archive = minute_history_path(snapshot_file)
    frames = [pd.read_csv(archive)] if archive.exists() else []
    # Preserve the preceding snapshot when upgrading an existing installation.
    if snapshot_file.exists():
        frames.append(pd.read_csv(snapshot_file))
    frames.append(current)
    history = pd.concat([_prepare_input(frame) for frame in frames], ignore_index=True)
    if history.empty:
        return
    # Repeated fetches correct the same bar; they must not multiply volume.
    # Each input has already recorded its own duplicates. A later clean fetch
    # can replace a marked bar, but merely archiving it cannot erase the defect.
    history = history.drop_duplicates(["symbol", "datetime"], keep="last")
    history = history.sort_values(["symbol", "datetime"])
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".csv.tmp")
    history.to_csv(temporary, index=False)
    temporary.replace(archive)


def read_minute_history(snapshot_file: Path) -> pd.DataFrame:
    archive = minute_history_path(snapshot_file)
    frames = [pd.read_csv(archive)] if archive.exists() else []
    if snapshot_file.exists():
        frames.append(pd.read_csv(snapshot_file))
    if not frames:
        return pd.DataFrame()
    frames = [_prepare_input(frame) for frame in frames]
    history = pd.concat(frames, ignore_index=True)
    # Only deduplicate overlapping archive/snapshot records. Within one input,
    # duplicate bars are retained so the feature completeness check rejects it.
    if len(frames) == 2:
        latest_keys = pd.MultiIndex.from_frame(frames[-1][["symbol", "datetime"]])
        old = history.iloc[:len(frames[0])]
        old_keys = pd.MultiIndex.from_frame(old[["symbol", "datetime"]])
        history = pd.concat([old[~old_keys.isin(latest_keys)], history.iloc[len(frames[0]):]])
    return history.sort_values(["symbol", "datetime"]).reset_index(drop=True)
