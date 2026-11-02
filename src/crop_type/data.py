"""Load annual multi-band reflectance tables and maintain field-level identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BANDS = ("Blue", "Green", "Red", "RedEdge1", "RedEdge2", "SWIR1", "SWIR2")
CLASSES = ("Corn", "Cotton", "Soy")
ID_COLS = ("Crop", "aid", "County")
CUTOFFS = ("06-01", "06-15", "07-01", "08-01", "09-01", "10-01", "11-01")
LABELS = {
    "Corn 1": "Corn", "Corn 2": "Corn",
    "Cotton 1": "Cotton", "Cotton 2": "Cotton",
    "Soy 1": "Soy", "Soy 2": "Soy",
}
DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y")


@dataclass(frozen=True)
class StudyYear:
    year: int
    frame: pd.DataFrame
    feature_dates: dict[str, pd.Timestamp]
    hashes: dict[str, str]

    @property
    def targets(self) -> np.ndarray:
        return self.frame["target"].map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)

    @property
    def groups(self) -> np.ndarray:
        return self.frame["group_id"].to_numpy()

    def features(self, cutoff: str) -> pd.DataFrame:
        if cutoff not in CUTOFFS:
            raise ValueError(f"Unknown cutoff {cutoff}; select from {CUTOFFS}")
        start, end = pd.Timestamp(f"{self.year}-04-01"), pd.Timestamp(f"{self.year}-{cutoff}")
        columns = [c for c, d in self.feature_dates.items() if start <= d <= end]
        columns.sort(key=lambda c: (BANDS.index(c.split("__", 1)[0]), self.feature_dates[c]))
        if not columns:
            raise ValueError(f"No observations for cutoff {cutoff}")
        return self.frame.loc[:, columns]


def _band_name(filename: str) -> str:
    name = re.sub(r"[^a-z0-9]", "", Path(filename).stem.lower())
    for token, band in (("rededge1", "RedEdge1"), ("rededge2", "RedEdge2"),
                        ("swir1", "SWIR1"), ("swir2", "SWIR2"),
                        ("blue", "Blue"), ("green", "Green"), ("red", "Red")):
        if token in name:
            return band
    raise ValueError(f"Unknown spectral band: {filename}")


def _date(header: str) -> pd.Timestamp:
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(str(header).strip(), fmt))
        except ValueError:
            continue
    raise ValueError(f"Invalid observation date: {header}")


def _load_csv(content: bytes, filename: str) -> tuple[str, pd.DataFrame, int]:
    band = _band_name(filename)
    frame = pd.read_csv(BytesIO(content), low_memory=False)
    absent = set(ID_COLS).difference(frame.columns)
    if absent:
        raise ValueError(f"{filename}: missing field identifiers: {sorted(absent)}")
    for name in ID_COLS:
        if frame[name].isna().any():
            raise ValueError(f"{filename}: missing values in {name}")
        frame[name] = frame[name].astype(str).str.strip()
        if frame[name].eq("").any():
            raise ValueError(f"{filename}: empty {name}")
    if frame.duplicated(list(ID_COLS)).any():
        raise ValueError(f"{filename}: duplicate Crop/aid/County records")
    features = [c for c in frame.columns if c not in ID_COLS]
    if not features:
        raise ValueError(f"{filename}: no dated columns")
    dates = [_date(c) for c in features]
    if len(set(dates)) != len(dates) or len({d.year for d in dates}) != 1:
        raise ValueError(f"{filename}: inconsistent or duplicate observation dates")
    renaming = dict(zip(features, (f"{band}__{d:%Y-%m-%d}" for d in dates)))
    frame.rename(columns=renaming, inplace=True)
    for col in renaming.values():
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return band, frame, dates[0].year


def read_study_year(data_zip: str | Path, year: int) -> StudyYear:
    """Read exactly seven matching annual bands from the released data archive."""
    data_zip = Path(data_zip)
    if not data_zip.is_file():
        raise FileNotFoundError(data_zip)
    parts, checksums = {}, {}
    with ZipFile(data_zip) as archive:
        for member in sorted(archive.namelist()):
            if not member.lower().endswith(".csv"):
                continue
            payload = archive.read(member)
            band, frame, observed_year = _load_csv(payload, member)
            if observed_year != year:
                continue
            if band in parts:
                raise ValueError(f"More than one {band} input for {year}")
            parts[band] = frame
            checksums[band] = sha256(payload).hexdigest()
    if set(parts) != set(BANDS):
        raise ValueError(f"Year {year}: expected {BANDS}; found {tuple(sorted(parts))}")
    # Match the original outer-merge operation, with explicit uniqueness checks.
    merged = parts[BANDS[0]].copy()
    for band in BANDS[1:]:
        merged = merged.merge(parts[band], on=list(ID_COLS), how="outer",
                              validate="one_to_one", sort=False)
    unknown = sorted(set(merged["Crop"]).difference(LABELS))
    if unknown:
        raise ValueError(f"Unrecognized crop identifiers: {unknown}")
    merged["target"] = merged["Crop"].map(LABELS)
    merged["group_id"] = merged[list(ID_COLS)].agg("|".join, axis=1)
    if merged["group_id"].duplicated().any():
        raise ValueError("Non-unique composite record identifier")
    merged.sort_values(["County", "aid", "Crop"], kind="mergesort", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    dates = {c: pd.Timestamp(c.split("__", 1)[1]) for c in merged.columns
             if "__" in c and c.split("__", 1)[0] in BANDS}
    if not dates or {d.year for d in dates.values()} != {year}:
        raise ValueError("Mixed study years or no observation dates")
    if set(merged["target"]) != set(CLASSES):
        raise ValueError("All three crop classes must be represented")
    return StudyYear(year, merged, dates, checksums)


def signature(frame: pd.DataFrame, indices: np.ndarray) -> str:
    identifiers = sorted(frame.iloc[indices][list(ID_COLS)].agg("|".join, axis=1).tolist())
    return sha256("\n".join(identifiers).encode("utf-8")).hexdigest()


def heldout_partition(study: StudyYear, seed: int = 1337,
                      reference: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Use a supplied fixed partition or generate the original first 5-fold split.

    Providing the released partition file preserves the membership of the
    reference test set when field identifiers have been relabeled.
    """
    frame = study.frame
    if reference is None:
        split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train, test = next(split.split(np.zeros(len(frame)), study.targets, study.groups))
    else:
        partition = pd.read_csv(reference, dtype={k: str for k in ID_COLS})
        required = set(ID_COLS) | {"partition"}
        if not required.issubset(partition):
            raise ValueError(f"Partition file must contain {sorted(required)}")
        if "year" in partition:
            partition = partition[partition["year"].astype(int) == study.year]
        partition = partition.copy()
        partition["key"] = partition[list(ID_COLS)].agg("|".join, axis=1)
        if partition["key"].duplicated().any():
            raise ValueError("Duplicate identifiers in fixed partition")
        mapping = partition.set_index("key")["partition"]
        if set(mapping.index) != set(study.groups) or set(mapping.values) != {"TRAIN", "TEST"}:
            raise ValueError("Fixed partition does not match the current dataset")
        selected = frame["group_id"].map(mapping)
        train = np.flatnonzero(selected.eq("TRAIN"))
        test = np.flatnonzero(selected.eq("TEST"))
    if set(study.groups[train]) & set(study.groups[test]):
        raise ValueError("Training and test record groups overlap")
    return train, test
