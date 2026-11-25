"""Dataset ingestion, provenance, and chronological splitters for DriftGuard-Kafka M4.

Implements leakage-free loaders for:
1. **SKAB** (`waico/SKAB` v0.9): semicolon-delimited multivariate sensor CSVs across
   `valve1/`, `valve2/`, `other/`, and `anomaly-free/`, preserving `anomaly` and
   `changepoint` labels as strictly separate ground-truth arrays.
2. **TSB-AD-M** (`TheDatumOrg/TSB-AD` multivariate suite): comma-delimited CSVs
   following the official `_tr_<train_end>_1st_<first_anom>.csv` chronological
   split protocol and `TSB_AD.utils.slidingWindows.find_length_rank` windowing.

Research invariants enforced:
- Unsupervised training (`x_train_normal`) and threshold calibration (`x_val_normal`)
  contain normal observations only (`y == 0`), sliced strictly from the chronological
  training prefix without ever reading test labels.
- Time-series rows are never shuffled or randomly split.
- Missing or empty dataset directories raise `DatasetNotAvailableError` rather than
  silently skipping benchmarks.
"""

from __future__ import annotations

import hashlib
import re
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from TSB_AD.utils.slidingWindows import find_length_rank

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SKAB_ARCHIVE_URL = "https://codeload.github.com/waico/SKAB/tar.gz/refs/heads/master"
TSB_AD_M_ZIP_URL = "https://www.thedatum.org/datasets/TSB-AD-M.zip"
TSB_AD_M_EVA_URL = (
    "https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/main/Datasets/File_List/TSB-AD-M-Eva.csv"
)

_TSB_TRAIN_RE = re.compile(r"_tr_(\d+)(?:_|$)")


class DatasetNotAvailableError(FileNotFoundError):
    """Raised when a required external benchmark dataset is missing or empty."""


class DatasetFormatError(ValueError):
    """Raised when a dataset file violates expected schema or split conventions."""


@dataclass(frozen=True)
class ChronologicalSplit:
    """Leakage-free chronological train/validation/test partition for one series."""

    experiment_id: str
    dataset_suite: str
    domain: str
    feature_names: tuple[str, ...]
    x_train_normal: FloatArray
    x_val_normal: FloatArray
    x_test: FloatArray
    y_test_anomaly: IntArray
    y_test_changepoint: IntArray | None
    sliding_window: int
    train_boundary_index: int
    dropped_train_anomalies: int
    is_anomaly_free: bool = False


@dataclass(frozen=True)
class SkabExperiment:
    """Raw ingested SKAB experiment preserving `anomaly` and `changepoint` separately."""

    experiment_id: str
    category: str
    path: Path
    feature_names: tuple[str, ...]
    timestamps: tuple[str, ...]
    features: FloatArray
    y_anomaly: IntArray
    y_changepoint: IntArray
    is_anomaly_free: bool


@dataclass(frozen=True)
class DatasetSuiteBundle:
    """Loaded dataset splits paired with reproducibility provenance metadata."""

    suite_name: str
    splits: list[ChronologicalSplit]
    provenance: dict[str, Any]


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(files: list[Path], base_dir: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(files):
        rel = file_path.relative_to(base_dir).as_posix()
        digest.update(f"{rel}:{file_path.stat().st_size}\n".encode())
    return digest.hexdigest()


def _download_https(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"only https:// dataset URLs are permitted, got {url!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "DriftGuard-Kafka-M4/1.0"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=300) as resp, destination.open("wb") as out:  # noqa: S310
        while block := resp.read(1024 * 1024):
            out.write(block)


def ensure_skab_downloaded(skab_root: Path) -> Path:
    """Download and extract SKAB into `skab_root` if not already present."""
    data_dir = _locate_skab_data_dir(skab_root, allow_missing=True)
    if data_dir is not None:
        return data_dir
    archive_path = skab_root / "skab-master.tar.gz"
    if not archive_path.exists():
        _download_https(SKAB_ARCHIVE_URL, archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (skab_root / member.name).resolve()
            if not str(member_path).startswith(str(skab_root.resolve())):
                raise DatasetFormatError(f"unsafe path in SKAB archive: {member.name}")
        tar.extractall(path=skab_root, filter="data")
    located = _locate_skab_data_dir(skab_root, allow_missing=False)
    if located is None:  # pragma: no cover - guarded by allow_missing=False
        raise DatasetNotAvailableError(f"SKAB data directory not found after extract: {skab_root}")
    return located


def ensure_tsb_ad_m_downloaded(tsb_root: Path) -> Path:
    """Download and extract TSB-AD-M and TSB-AD-M-Eva.csv into `tsb_root` if missing."""
    data_dir = _locate_tsb_data_dir(tsb_root, allow_missing=True)
    eva_path = tsb_root / "TSB-AD-M-Eva.csv"
    if not eva_path.exists():
        _download_https(TSB_AD_M_EVA_URL, eva_path)
    if data_dir is not None:
        return data_dir
    zip_path = tsb_root / "TSB-AD-M.zip"
    if not zip_path.exists():
        _download_https(TSB_AD_M_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            member_path = (tsb_root / info.filename).resolve()
            if not str(member_path).startswith(str(tsb_root.resolve())):
                raise DatasetFormatError(f"unsafe path in TSB-AD-M zip: {info.filename}")
        zf.extractall(path=tsb_root)  # noqa: S202 - validated member paths above
    located = _locate_tsb_data_dir(tsb_root, allow_missing=False)
    if located is None:  # pragma: no cover - guarded by allow_missing=False
        raise DatasetNotAvailableError(f"TSB-AD-M directory not found after extract: {tsb_root}")
    return located


def _locate_skab_data_dir(skab_root: Path, allow_missing: bool = False) -> Path | None:
    candidates = [
        skab_root / "SKAB-master" / "data",
        skab_root / "data",
        skab_root,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.rglob("*.csv")):
            return candidate
    if allow_missing:
        return None
    raise DatasetNotAvailableError(
        f"SKAB dataset not found under {skab_root}. "
        f"Download into data/external/skab or pass --download."
    )


def _locate_tsb_data_dir(tsb_root: Path, allow_missing: bool = False) -> Path | None:
    candidates = [
        tsb_root / "TSB-AD-M",
        tsb_root,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            csvs = [p for p in candidate.glob("*.csv") if p.name != "TSB-AD-M-Eva.csv"]
            if csvs:
                return candidate
    if allow_missing:
        return None
    raise DatasetNotAvailableError(
        f"TSB-AD-M dataset not found under {tsb_root}. "
        f"Download into data/external/tsb_ad_m or pass --download."
    )


def _estimate_sliding_window(features: FloatArray, default_window: int = 100) -> int:
    try:
        win = int(find_length_rank(features[:, 0].reshape(-1, 1), rank=1))
    except Exception:  # noqa: BLE001, S110 - fallback if short/constant synthetic fixture
        win = default_window
    return win if win > 0 else default_window


def parse_skab_csv(csv_path: Path, data_root: Path) -> SkabExperiment:
    """Parse one semicolon-delimited SKAB CSV while keeping labels separate."""
    df = pd.read_csv(csv_path, sep=";").dropna()
    if df.empty:
        raise DatasetFormatError(f"SKAB CSV is empty: {csv_path}")
    if "datetime" not in df.columns:
        raise DatasetFormatError(f"SKAB CSV missing 'datetime' column: {csv_path}")

    rel_id = csv_path.relative_to(data_root).as_posix()
    category = csv_path.parent.name if csv_path.parent != data_root else "root"
    is_anom_free = "anomaly-free" in rel_id or (
        "anomaly" not in df.columns and "changepoint" not in df.columns
    )

    if is_anom_free:
        sensor_cols = [c for c in df.columns if c not in {"datetime", "anomaly", "changepoint"}]
        y_anom = np.zeros(len(df), dtype=np.int64)
        y_cp = np.zeros(len(df), dtype=np.int64)
    else:
        if "anomaly" not in df.columns or "changepoint" not in df.columns:
            raise DatasetFormatError(
                f"SKAB fault CSV must contain both 'anomaly' and 'changepoint': {csv_path}"
            )
        sensor_cols = [c for c in df.columns if c not in {"datetime", "anomaly", "changepoint"}]
        y_anom = df["anomaly"].astype(float).astype(np.int64).to_numpy()
        y_cp = df["changepoint"].astype(float).astype(np.int64).to_numpy()

    if not sensor_cols:
        raise DatasetFormatError(f"SKAB CSV has no sensor feature columns: {csv_path}")

    features = df[sensor_cols].to_numpy(dtype=np.float64)
    timestamps = tuple(df["datetime"].astype(str).tolist())
    return SkabExperiment(
        experiment_id=rel_id,
        category=category,
        path=csv_path,
        feature_names=tuple(sensor_cols),
        timestamps=timestamps,
        features=features,
        y_anomaly=y_anom,
        y_changepoint=y_cp,
        is_anomaly_free=is_anom_free,
    )


def split_skab_experiment(
    exp: SkabExperiment,
    default_train_size: int = 400,
    val_fraction: float = 0.2,
) -> ChronologicalSplit:
    """Create a chronological normal-only train/val + test split for a SKAB experiment."""
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be in [0.05, 0.5]")
    n_rows = int(exp.features.shape[0])
    if n_rows < 20:
        raise DatasetFormatError(f"SKAB experiment {exp.experiment_id} too short ({n_rows} rows)")

    train_end = int(0.8 * n_rows) if exp.is_anomaly_free else min(default_train_size, n_rows // 2)

    prefix_x = exp.features[:train_end]
    prefix_y = exp.y_anomaly[:train_end]
    normal_mask = prefix_y == 0
    normal_x = prefix_x[normal_mask]
    dropped_anom = int(train_end - int(normal_x.shape[0]))

    if normal_x.shape[0] < 10:
        raise DatasetFormatError(
            f"SKAB experiment {exp.experiment_id} has fewer than 10 normal rows in prefix"
        )

    val_start = max(5, int(round(normal_x.shape[0] * (1.0 - val_fraction))))
    val_start = min(val_start, int(normal_x.shape[0]) - 2)
    x_train_normal = normal_x[:val_start]
    x_val_normal = normal_x[val_start:]

    x_test = exp.features[train_end:]
    y_test_anom = exp.y_anomaly[train_end:]
    y_test_cp = exp.y_changepoint[train_end:]

    sliding_window = _estimate_sliding_window(exp.features, default_window=60)
    return ChronologicalSplit(
        experiment_id=exp.experiment_id,
        dataset_suite="SKAB",
        domain=exp.category,
        feature_names=exp.feature_names,
        x_train_normal=x_train_normal,
        x_val_normal=x_val_normal,
        x_test=x_test,
        y_test_anomaly=y_test_anom,
        y_test_changepoint=y_test_cp,
        sliding_window=sliding_window,
        train_boundary_index=train_end,
        dropped_train_anomalies=dropped_anom,
        is_anomaly_free=exp.is_anomaly_free,
    )


def load_skab_suite(
    skab_root: Path,
    default_train_size: int = 400,
    val_fraction: float = 0.2,
) -> DatasetSuiteBundle:
    """Ingest full SKAB benchmark corpus and build experiment-level splits."""
    data_dir = _locate_skab_data_dir(skab_root, allow_missing=False)
    if data_dir is None:  # pragma: no cover
        raise DatasetNotAvailableError(f"SKAB data missing: {skab_root}")

    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        raise DatasetNotAvailableError(f"No SKAB CSV files found in {data_dir}")

    experiments = [parse_skab_csv(p, data_dir) for p in csv_files]
    splits = [
        split_skab_experiment(exp, default_train_size=default_train_size, val_fraction=val_fraction)
        for exp in experiments
    ]

    archive_path = skab_root / "skab-master.tar.gz"
    checksum = (
        compute_sha256(archive_path)
        if archive_path.exists()
        else _manifest_sha256(csv_files, data_dir)
    )
    checksum_source = archive_path.name if archive_path.exists() else "directory_manifest"

    total_rows = sum(int(e.features.shape[0]) for e in experiments)
    total_anomalies = sum(int(e.y_anomaly.sum()) for e in experiments)
    total_changepoints = sum(int(e.y_changepoint.sum()) for e in experiments)
    fault_exps = [e for e in experiments if not e.is_anomaly_free]
    af_exps = [e for e in experiments if e.is_anomaly_free]

    provenance: dict[str, Any] = {
        "dataset": "SKAB (Skoltech Anomaly Benchmark v0.9)",
        "source_url": SKAB_ARCHIVE_URL,
        "local_data_dir": str(data_dir),
        "checksum_sha256": checksum,
        "checksum_source": checksum_source,
        "total_csv_files": len(experiments),
        "fault_experiment_files": len(fault_exps),
        "anomaly_free_files": len(af_exps),
        "categories": sorted({e.category for e in experiments}),
        "feature_names": list(experiments[0].feature_names),
        "n_features": len(experiments[0].feature_names),
        "total_rows": total_rows,
        "total_anomaly_points": total_anomalies,
        "total_changepoint_points": total_changepoints,
        "labels_preserved_separately": ["anomaly", "changepoint"],
        "split_protocol": (
            f"Chronological per-experiment split (fault train_end={default_train_size}, "
            f"filtered strictly to anomaly==0, chronological val_fraction={val_fraction}; "
            f"evaluated on chronological suffix [train_end:])"
        ),
    }
    return DatasetSuiteBundle(suite_name="SKAB", splits=splits, provenance=provenance)


def parse_tsb_train_index(filename: str) -> tuple[str, int]:
    """Extract `(domain, train_end)` from an official TSB-AD-M filename."""
    stem = Path(filename).stem
    match = _TSB_TRAIN_RE.search(stem)
    if match is None:
        raise DatasetFormatError(
            f"TSB-AD-M filename {filename!r} does not encode '_tr_<train_end>_'"
        )
    parts = stem.split("_")
    domain = parts[1] if len(parts) >= 2 else "unknown"
    return domain, int(match.group(1))


def split_tsb_csv(
    csv_path: Path,
    val_fraction: float = 0.2,
) -> ChronologicalSplit:
    """Load one TSB-AD-M CSV and apply the official `_tr_` chronological split."""
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be in [0.05, 0.5]")
    domain, raw_train_end = parse_tsb_train_index(csv_path.name)

    df = pd.read_csv(csv_path).dropna()
    if df.empty or df.shape[1] < 2:
        raise DatasetFormatError(f"TSB-AD-M CSV invalid shape {df.shape}: {csv_path}")
    if "Label" not in df.columns:
        raise DatasetFormatError(f"TSB-AD-M CSV missing 'Label' column: {csv_path}")

    feature_cols = [c for c in df.columns if c != "Label"]
    features = df[feature_cols].to_numpy(dtype=np.float64)
    labels = df["Label"].astype(float).astype(np.int64).to_numpy()

    n_rows = int(features.shape[0])
    if raw_train_end <= 0 or raw_train_end >= n_rows:
        raise DatasetFormatError(
            f"TSB-AD-M train_end={raw_train_end} out of bounds for {n_rows} rows in {csv_path.name}"
        )

    prefix_x = features[:raw_train_end]
    prefix_y = labels[:raw_train_end]
    normal_mask = prefix_y == 0
    normal_x = prefix_x[normal_mask]
    dropped_anom = int(raw_train_end - int(normal_x.shape[0]))
    if normal_x.shape[0] < 10:
        raise DatasetFormatError(
            f"TSB-AD-M {csv_path.name} has fewer than 10 normal training observations"
        )

    val_start = max(5, int(round(normal_x.shape[0] * (1.0 - val_fraction))))
    val_start = min(val_start, int(normal_x.shape[0]) - 2)
    x_train_normal = normal_x[:val_start]
    x_val_normal = normal_x[val_start:]

    x_test = features[raw_train_end:]
    y_test = labels[raw_train_end:]
    sliding_window = _estimate_sliding_window(features, default_window=100)

    return ChronologicalSplit(
        experiment_id=csv_path.name,
        dataset_suite="TSB-AD-M",
        domain=domain,
        feature_names=tuple(str(c) for c in feature_cols),
        x_train_normal=x_train_normal,
        x_val_normal=x_val_normal,
        x_test=x_test,
        y_test_anomaly=y_test,
        y_test_changepoint=None,
        sliding_window=sliding_window,
        train_boundary_index=raw_train_end,
        dropped_train_anomalies=dropped_anom,
        is_anomaly_free=bool(int(y_test.sum()) == 0),
    )


def load_tsb_ad_m_suite(
    tsb_root: Path,
    val_fraction: float = 0.2,
    max_per_domain: int | None = None,
) -> DatasetSuiteBundle:
    """Ingest TSB-AD-M evaluation series using official `_tr_` protocol and Eva list."""
    data_dir = _locate_tsb_data_dir(tsb_root, allow_missing=False)
    if data_dir is None:  # pragma: no cover
        raise DatasetNotAvailableError(f"TSB-AD-M data missing: {tsb_root}")

    eva_csv = tsb_root / "TSB-AD-M-Eva.csv"
    if not eva_csv.exists() and (data_dir / "TSB-AD-M-Eva.csv").exists():
        eva_csv = data_dir / "TSB-AD-M-Eva.csv"

    if eva_csv.exists():
        eva_names = [str(x).strip() for x in pd.read_csv(eva_csv)["file_name"].tolist()]
        candidate_paths: list[Path] = []
        missing_eva: list[str] = []
        for fname in eva_names:
            fpath = data_dir / fname
            if fpath.exists():
                candidate_paths.append(fpath)
            else:
                missing_eva.append(fname)
        if missing_eva:
            raise DatasetNotAvailableError(
                f"TSB-AD-M missing {len(missing_eva)} official Eva files, e.g. {missing_eva[0]}"
            )
        eva_source = str(eva_csv)
    else:
        candidate_paths = sorted(p for p in data_dir.glob("*.csv") if p.name != "TSB-AD-M-Eva.csv")
        eva_source = "directory_glob"

    if not candidate_paths:
        raise DatasetNotAvailableError(f"No TSB-AD-M CSV files found in {data_dir}")

    available_domains: dict[str, int] = defaultdict(int)
    selected_paths: list[Path] = []
    domain_selected: dict[str, int] = defaultdict(int)

    for path in candidate_paths:
        domain, _ = parse_tsb_train_index(path.name)
        available_domains[domain] += 1
        if (
            max_per_domain is None
            or max_per_domain <= 0
            or domain_selected[domain] < max_per_domain
        ):
            selected_paths.append(path)
            domain_selected[domain] += 1

    splits = [split_tsb_csv(path, val_fraction=val_fraction) for path in selected_paths]

    zip_path = tsb_root / "TSB-AD-M.zip"
    checksum = (
        compute_sha256(zip_path)
        if zip_path.exists()
        else _manifest_sha256(selected_paths, data_dir)
    )
    checksum_source = zip_path.name if zip_path.exists() else "directory_manifest"

    total_eval_rows = sum(
        int(s.x_train_normal.shape[0] + s.x_val_normal.shape[0] + s.x_test.shape[0])
        + s.dropped_train_anomalies
        for s in splits
    )
    total_test_rows = sum(int(s.x_test.shape[0]) for s in splits)
    total_test_anom = sum(int(s.y_test_anomaly.sum()) for s in splits)

    selection_desc = (
        "full_eva_180"
        if (max_per_domain is None or max_per_domain <= 0)
        else f"domain_stratified_first_{max_per_domain}_per_domain"
    )

    provenance: dict[str, Any] = {
        "dataset": "TSB-AD-M (TheDatumOrg Multivariate Time-Series Anomaly Benchmark)",
        "source_url": TSB_AD_M_ZIP_URL,
        "eva_list_url": TSB_AD_M_EVA_URL,
        "eva_list_source": eva_source,
        "local_data_dir": str(data_dir),
        "checksum_sha256": checksum,
        "checksum_source": checksum_source,
        "available_eva_files": len(candidate_paths),
        "evaluated_files": len(splits),
        "available_domains": dict(sorted(available_domains.items())),
        "evaluated_domains": dict(sorted(domain_selected.items())),
        "domains_covered_count": len(domain_selected),
        "domains_available_count": len(available_domains),
        "selection_policy": selection_desc,
        "max_per_domain": max_per_domain,
        "evaluated_total_rows": total_eval_rows,
        "evaluated_test_rows": total_test_rows,
        "evaluated_test_anomalies": total_test_anom,
        "split_protocol": (
            f"Official _tr_<train_end> chronological prefix filtered to Label==0, "
            f"split chronologically (1 - {val_fraction:.2f}) train / {val_fraction:.2f} val; "
            f"evaluated on [train_end:] with official TSB_AD VUS-PR and ACF slidingWindow."
        ),
    }
    return DatasetSuiteBundle(suite_name="TSB-AD-M", splits=splits, provenance=provenance)
