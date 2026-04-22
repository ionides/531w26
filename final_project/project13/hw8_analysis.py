
from __future__ import annotations

import copy
import hashlib
import json
import shelve
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from pypomp import Pomp, ParTrans, RWSigma


# -----------------------------
# Configuration + public dataclasses
# -----------------------------
@dataclass(frozen=True)
class CacheConfig:
    enable_shelve_cache: bool = True
    clear_shelve_cache: bool = False
    require_existing_cache_hits: bool = False
    cache_version_token: str = "hw8_cache_v1"
    shelve_cache_directory: str = ".hw8_cache"
    shelve_cache_basename: str = "HW8"     # can be overridden in QMD

    csv_path: str = "FitbitHourly.csv"


@dataclass(frozen=True)
class AnalysisConfig:
    analysis_start_utc_iso: str
    analysis_end_utc_iso: str
    participant_data_lookback_hours: int = 48
    assumed_local_timezone: str = "America/New_York"
    participant_covariate_keys: list[str] = None

    estimated_free_params: list[str] = None

    initial_phi: float = 0.75
    initial_sigma: float = 0.30
    initial_k_fitbit: float = 200.0

    # Local IF2 controls
    local_mle_particles: int = 100
    local_mle_mif_iterations: int = 6
    local_mle_random_seed: int = 11
    local_mle_cooling_fraction: float = 0.50
    local_mle_rw_sd_scale: float = 0.02
    local_mle_evaluation_particles: int = 100
    local_mle_evaluation_pfilter_reps: int = 3

    # Global search controls
    global_search_random_seed: int = 37
    global_search_include_current_theta_as_start: bool = True
    global_search_jitter_scale: float = 0.10
    global_search_box_shrink_factor_per_stage: float = 1.00

    # Global start box
    global_start_box_phi_low: float = -0.95
    global_start_box_phi_high: float = 0.95
    global_start_box_sigma_low: float = 0.05
    global_start_box_sigma_high: float = 2.00
    global_start_box_k_fitbit_low: float = 1.0
    global_start_box_k_fitbit_high: float = 50.0

    # Stage configs (as dicts you already build in QMD)
    global_search_stage_controls: list[dict] = None


@dataclass
class ParticipantResult:
    participant_id: str
    hourly_data: "HourlyStepModelData"
    initial_params: dict[str, float]
    active_free_params: list[str]

    participant_summary: pd.DataFrame

    local_run: "StepPompRunResult"
    local_mle_summary: pd.DataFrame

    global_search_result: "GlobalSearchResult"
    global_best_summary: pd.DataFrame
    global_stage_summary: pd.DataFrame

    coefficient_table: pd.DataFrame


# -----------------------------
# Original structures you already had
# (kept mostly unchanged; only moved into module)
# -----------------------------
CSV_COLUMNS = [
    "PARTICIPANTIDENTIFIER",
    "STUDY_USER_ID",
    "time_utc",
    "date_utc",
    "hour_of_day_utc",
    "fitbitSteps",
    "fitbitHRAvg",
    "fitbitSleepMinutes",
    "moodScore",
    "messageSent",
]


def normalize_timestamp_to_utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _cache_key_default(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


class ShelveCache:
    """Small wrapper so cache path + CSV signature live in one object."""
    def __init__(self, cfg: CacheConfig):
        self.cfg = cfg
        self.csv_path = Path(cfg.csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"{cfg.csv_path} must be in the working directory.")
        self.csv_signature = {
            "csv_size_bytes": int(self.csv_path.stat().st_size),
            "csv_modified_ns": int(self.csv_path.stat().st_mtime_ns),
        }
        self.cache_dir = Path(cfg.shelve_cache_directory)
        self.cache_path = self.cache_dir / cfg.shelve_cache_basename
        if cfg.enable_shelve_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if cfg.clear_shelve_cache:
                with shelve.open(str(self.cache_path)) as db:
                    db.clear()

    def build_cache_key(self, cache_name: str, **cache_inputs) -> str:
        payload = {
            "cache_name": cache_name,
            "cache_version_token": self.cfg.cache_version_token,
            "csv_signature": self.csv_signature,
            "cache_inputs": cache_inputs,
        }
        encoded = json.dumps(payload, sort_keys=True, default=_cache_key_default).encode("utf-8")
        return f"{cache_name}:{hashlib.sha256(encoded).hexdigest()}"

    def get_or_compute(self, cache_name: str, cache_inputs: dict, compute_fn, *, prepare_for_store=None):
        cache_key = self.build_cache_key(cache_name, **cache_inputs)
        cache_load_error = None
        if self.cfg.enable_shelve_cache:
            with shelve.open(str(self.cache_path)) as db:
                if cache_key in db:
                    try:
                        return db[cache_key]
                    except Exception as exc:
                        cache_load_error = exc
                        try:
                            del db[cache_key]
                        except Exception:
                            pass
        if self.cfg.require_existing_cache_hits:
            if cache_load_error is not None:
                raise FileNotFoundError(
                    f"Unreadable shelve cache entry for {cache_name!r} in {self.cache_path} "
                    f"while require_existing_cache_hits=True."
                ) from cache_load_error
            raise FileNotFoundError(
                f"Missing required shelve cache entry for {cache_name!r} in {self.cache_path} "
                f"while require_existing_cache_hits=True."
            )
        value = compute_fn()
        stored_value = prepare_for_store(value) if prepare_for_store is not None else value
        if self.cfg.enable_shelve_cache:
            with shelve.open(str(self.cache_path)) as db:
                db[cache_key] = stored_value
        return stored_value


@dataclass(frozen=True)
class CovariateSpec:
    key: str
    label: str
    source_col: str
    beta_param: str
    description: str
    standardize: bool = True


COVARIATE_SPECS = {
    "trail_steps_24h": CovariateSpec(
        key="trail_steps_24h",
        label="Trailing 24h steps",
        source_col="cov_trail_steps_24h",
        beta_param="beta_steps24",
        description="Sum of Fitbit steps over the prior 24 hours.",
    ),
    "trail_sleep_minutes_24h": CovariateSpec(
        key="trail_sleep_minutes_24h",
        label="Trailing 24h sleep minutes",
        source_col="cov_trail_sleep_minutes_24h",
        beta_param="beta_sleep24",
        description="Sum of Fitbit sleep minutes over the prior 24 hours.",
    ),
    "trail_hr_mean_24h": CovariateSpec(
        key="trail_hr_mean_24h",
        label="Trailing 24h heart rate mean",
        source_col="cov_trail_hr_mean_24h",
        beta_param="beta_hr24",
        description="Average Fitbit heart rate over the prior 24 hours.",
    ),
    "trail_mood_mean_24h": CovariateSpec(
        key="trail_mood_mean_24h",
        label="Trailing 24h mood mean",
        source_col="cov_trail_mood_mean_24h",
        beta_param="beta_mood24",
        description="Average mood score over the prior 24 hours.",
    ),
    "send_lag_0_3h": CovariateSpec(
        key="send_lag_0_3h",
        label="Message sent 0 to 3 hours ago",
        source_col="cov_send_lag_0_3h",
        beta_param="eta_send_0_3h",
        description="Indicator that the most recent message was in the current hour or previous 3 hours.",
        standardize=False,
    ),
    "send_lag_4_11h": CovariateSpec(
        key="send_lag_4_11h",
        label="Message sent 4 to 11 hours ago",
        source_col="cov_send_lag_4_11h",
        beta_param="eta_send_4_11h",
        description="Indicator that the most recent message was 4 to 11 hours ago.",
        standardize=False,
    ),
    "send_lag_12_23h": CovariateSpec(
        key="send_lag_12_23h",
        label="Message sent 12 to 23 hours ago",
        source_col="cov_send_lag_12_23h",
        beta_param="eta_send_12_23h",
        description="Indicator that the most recent message was 12 to 23 hours ago.",
        standardize=False,
    ),
}


@dataclass
class HourlyStepModelData:
    participant_id: str
    study_user_id: str | None
    hourly_frame: pd.DataFrame
    covariate_keys: list[str]
    covariate_params: dict[str, str]
    covariate_standardization: dict[str, dict[str, float]]
    decision_window_frame: pd.DataFrame
    dropped_covariates: list[str]
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    assumed_timezone: str

    @property
    def model_covariate_columns(self) -> list[str]:
        return [COVARIATE_SPECS[key].source_col for key in self.covariate_keys]

    @property
    def ys(self) -> pd.DataFrame:
        return self.hourly_frame.set_index("time_index")[["fitbit_steps_obs"]]

    @property
    def covars(self) -> pd.DataFrame:
        columns = ["baseline_log_mean"] + self.model_covariate_columns
        return self.hourly_frame.set_index("time_index")[columns]


@dataclass
class StepPompFitResult:
    initial_params: dict[str, float]
    fitted_params: dict[str, float]
    loglik: float
    pomp_object: Pomp | None
    mif_summary: pd.DataFrame
    mif_traces: pd.DataFrame
    pfilter_summary: pd.DataFrame
    timing_frame: pd.DataFrame
    runtime_breakdown: pd.DataFrame


@dataclass
class StepPompRunResult:
    initial_params: dict[str, float]
    fitted_params: dict[str, float]
    loglik: float
    pomp_object: Pomp | None
    mif_summary: pd.DataFrame
    mif_traces: pd.DataFrame
    pfilter_summary: pd.DataFrame
    filtered_states: pd.DataFrame
    plot_frame: pd.DataFrame
    reward_frame: pd.DataFrame
    timing_frame: pd.DataFrame
    runtime_breakdown: pd.DataFrame


@dataclass(frozen=True)
class GlobalSearchStageConfig:
    name: str
    n_starts: int
    keep_top: int
    particles: int
    mif_iterations: int
    cooling_fraction: float
    rw_sd_scale: float
    evaluation_particles: int
    evaluation_pfilter_reps: int
    n_fresh_random_starts: int = 0
    n_jittered_restarts: int = 0


@dataclass
class GlobalSearchResult:
    best_fit: StepPompFitResult
    best_run: StepPompRunResult
    candidate_frame: pd.DataFrame
    stage_summary: pd.DataFrame
    start_box_frame: pd.DataFrame
    total_runtime_seconds: float


def _copy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy(deep=True)


def clone_hourly_step_model_data(data: HourlyStepModelData) -> HourlyStepModelData:
    return copy.deepcopy(data)


def strip_step_pomp_fit_result(result: StepPompFitResult) -> StepPompFitResult:
    return StepPompFitResult(
        initial_params=copy.deepcopy(result.initial_params),
        fitted_params=copy.deepcopy(result.fitted_params),
        loglik=float(result.loglik),
        pomp_object=None,
        mif_summary=_copy_frame(result.mif_summary),
        mif_traces=_copy_frame(result.mif_traces),
        pfilter_summary=_copy_frame(result.pfilter_summary),
        timing_frame=_copy_frame(result.timing_frame),
        runtime_breakdown=_copy_frame(result.runtime_breakdown),
    )


def strip_step_pomp_run_result(result: StepPompRunResult) -> StepPompRunResult:
    return StepPompRunResult(
        initial_params=copy.deepcopy(result.initial_params),
        fitted_params=copy.deepcopy(result.fitted_params),
        loglik=float(result.loglik),
        pomp_object=None,
        mif_summary=_copy_frame(result.mif_summary),
        mif_traces=_copy_frame(result.mif_traces),
        pfilter_summary=_copy_frame(result.pfilter_summary),
        filtered_states=_copy_frame(result.filtered_states),
        plot_frame=_copy_frame(result.plot_frame),
        reward_frame=_copy_frame(result.reward_frame),
        timing_frame=_copy_frame(result.timing_frame),
        runtime_breakdown=_copy_frame(result.runtime_breakdown),
    )


def strip_global_search_result(result: GlobalSearchResult) -> GlobalSearchResult:
    return GlobalSearchResult(
        best_fit=strip_step_pomp_fit_result(result.best_fit),
        best_run=strip_step_pomp_run_result(result.best_run),
        candidate_frame=_copy_frame(result.candidate_frame),
        stage_summary=_copy_frame(result.stage_summary),
        start_box_frame=_copy_frame(result.start_box_frame),
        total_runtime_seconds=float(result.total_runtime_seconds),
    )


# -----------------------------
# CSV loading + feature engineering
# -----------------------------
@lru_cache(maxsize=8)
def load_participant_hourly_from_csv(
    csv_path: str,
    participant_id: str,
    start_iso: str,
    end_iso: str,
    lookback_hours: int = 48,
) -> pd.DataFrame:
    csv_path = Path(csv_path)

    start_utc = pd.Timestamp(start_iso).tz_convert("UTC") - pd.Timedelta(hours=lookback_hours)
    end_utc = pd.Timestamp(end_iso).tz_convert("UTC")
    chunks = []

    for chunk in pd.read_csv(csv_path, usecols=CSV_COLUMNS, chunksize=250_000):
        chunk = chunk.loc[chunk["PARTICIPANTIDENTIFIER"] == participant_id].copy()
        if chunk.empty:
            continue

        chunk["time_utc"] = pd.to_datetime(chunk["time_utc"], utc=True, errors="coerce")
        chunk = chunk.loc[chunk["time_utc"].between(start_utc, end_utc, inclusive="both")]
        if chunk.empty:
            continue

        chunk["date_utc"] = chunk["time_utc"].dt.date
        chunk["hour_of_day_utc"] = chunk["time_utc"].dt.hour
        chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No hourly rows were found for {participant_id}.")

    return pd.concat(chunks, ignore_index=True).sort_values("time_utc").reset_index(drop=True)


def build_baseline_log_mean(step_proxy: pd.Series, local_hour: pd.Series, local_dayofweek: pd.Series) -> pd.Series:
    log_proxy = np.log(step_proxy.clip(lower=0.0) + 1.0)
    valid = log_proxy.replace([np.inf, -np.inf], np.nan).dropna()
    global_mean = float(valid.mean()) if not valid.empty else float(np.log(1.0))

    hour_effect = log_proxy.groupby(local_hour).mean().reindex(range(24), fill_value=np.nan).astype(float) - global_mean
    hour_effect = hour_effect.fillna(0.0)
    hour_effect = hour_effect - float(hour_effect.mean())

    hour_lookup = hour_effect.reindex(local_hour.astype(int)).to_numpy(dtype=float)
    residual = log_proxy.to_numpy(dtype=float) - (global_mean + hour_lookup)
    residual_series = pd.Series(residual, index=step_proxy.index)

    dow_effect = residual_series.groupby(local_dayofweek).mean().reindex(range(7), fill_value=np.nan).astype(float)
    dow_effect = dow_effect.fillna(0.0)
    dow_effect = dow_effect - float(dow_effect.mean())

    baseline = (
        global_mean
        + hour_effect.reindex(local_hour.astype(int)).to_numpy(dtype=float)
        + dow_effect.reindex(local_dayofweek.astype(int)).to_numpy(dtype=float)
    )
    return pd.Series(baseline, index=step_proxy.index, dtype="float64")


def add_hourly_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    step_proxy = pd.to_numeric(result["fitbitSteps"], errors="coerce").clip(lower=0.0)
    sleep_minutes = pd.to_numeric(result.get("fitbitSleepMinutes"), errors="coerce")
    hr_avg = pd.to_numeric(result.get("fitbitHRAvg"), errors="coerce")
    mood_score = pd.to_numeric(result.get("moodScore"), errors="coerce")

    hr_fill = float(hr_avg.dropna().median()) if hr_avg.notna().any() else 0.0
    mood_fill = float(mood_score.dropna().mean()) if mood_score.notna().any() else 0.0

    result["cov_trail_steps_24h"] = step_proxy.fillna(0.0).shift(1).rolling(24, min_periods=1).sum()
    result["cov_trail_sleep_minutes_24h"] = sleep_minutes.fillna(0.0).shift(1).rolling(24, min_periods=1).sum()
    result["cov_trail_hr_mean_24h"] = hr_avg.fillna(hr_fill).shift(1).rolling(24, min_periods=1).mean()
    result["cov_trail_mood_mean_24h"] = mood_score.fillna(mood_fill).shift(1).rolling(24, min_periods=1).mean()

    send_series = pd.to_numeric(result.get("messageSent", 0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    hours_since_send = np.full(len(result), np.inf, dtype=float)
    last_send_index = None
    for index, sent_flag in enumerate(send_series.to_numpy(dtype=float)):
        if sent_flag >= 0.5:
            last_send_index = index
            hours_since_send[index] = 0.0
        elif last_send_index is not None:
            hours_since_send[index] = float(index - last_send_index)

    result["cov_send_lag_0_3h"] = ((hours_since_send >= 0) & (hours_since_send <= 3)).astype(float)
    result["cov_send_lag_4_11h"] = ((hours_since_send >= 4) & (hours_since_send <= 11)).astype(float)
    result["cov_send_lag_12_23h"] = ((hours_since_send >= 12) & (hours_since_send <= 23)).astype(float)
    result["baseline_log_mean"] = build_baseline_log_mean(step_proxy, result["local_hour"], result["local_dayofweek"])
    return result


def build_decision_windows(frame: pd.DataFrame) -> pd.DataFrame:
    decision_rows = frame.loc[frame["local_hour"] == 15].copy()
    if decision_rows.empty:
        return pd.DataFrame(
            columns=[
                "decision_id",
                "decision_time_utc",
                "decision_local_time",
                "decision_local_date",
                "hours_in_window",
                "complete_window",
                "fitbit_observed_hours",
                "fitbit_coverage_fraction",
                "reward_window_status",
                "window_start_position",
                "window_end_position",
            ]
        )

    rows = []
    for decision_id, (_, row) in enumerate(decision_rows.iterrows(), start=1):
        start_pos = int(row.name)
        window = frame.iloc[start_pos : start_pos + 24].copy()
        hours_in_window = len(window)
        complete_window = hours_in_window == 24
        fitbit_observed = int(window["fitbit_observed"].sum())
        fitbit_coverage_fraction = fitbit_observed / hours_in_window if hours_in_window > 0 else 0.0

        if not complete_window:
            status = "truncated_range"
        elif fitbit_observed >= 24:
            status = "fully_observed"
        elif fitbit_coverage_fraction >= 0.5:
            status = "partially_observed"
        else:
            status = "heavily_imputed"

        rows.append(
            {
                "decision_id": decision_id,
                "decision_time_utc": row["time_utc"],
                "decision_local_time": row["local_time"],
                "decision_local_date": row["local_date"],
                "hours_in_window": hours_in_window,
                "complete_window": bool(complete_window),
                "fitbit_observed_hours": fitbit_observed,
                "fitbit_coverage_fraction": fitbit_coverage_fraction,
                "reward_window_status": status,
                "window_start_position": start_pos,
                "window_end_position": start_pos + hours_in_window - 1,
            }
        )

    return pd.DataFrame(rows)


def prepare_hourly_model_data(
    hourly_frame: pd.DataFrame,
    *,
    participant_id: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    covariate_keys: list[str],
    assumed_timezone: str,
) -> HourlyStepModelData:
    if hourly_frame.empty:
        raise ValueError("Selected participant/date range produced no hourly rows.")

    working = hourly_frame.copy()
    working["time_utc"] = pd.to_datetime(working["time_utc"], utc=True, errors="coerce")
    working = working.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)

    study_user_id = None
    values = working.get("STUDY_USER_ID", pd.Series([], dtype=object)).dropna().astype(str)
    study_user_id = values.iloc[0] if not values.empty else None

    working["local_time"] = working["time_utc"].dt.tz_convert(assumed_timezone)
    working["local_date"] = working["local_time"].dt.date
    working["local_hour"] = working["local_time"].dt.hour.astype(int)
    working["local_dayofweek"] = working["local_time"].dt.dayofweek.astype(int)
    working = add_hourly_covariates(working)

    selected = working.loc[(working["time_utc"] >= start_utc) & (working["time_utc"] <= end_utc)].copy()
    if selected.empty:
        raise ValueError("No hourly rows fall inside the requested UTC range.")

    selected["fitbit_steps_obs"] = pd.to_numeric(selected["fitbitSteps"], errors="coerce")
    selected["fitbit_observed"] = selected["fitbit_steps_obs"].notna().astype(int)
    if int(selected["fitbit_observed"].sum()) == 0:
        raise ValueError("No observed Fitbit step data are available in the selected range.")

    normalized_covariates = []
    dropped_covariates = []
    covariate_standardization = {}
    covariate_params = {}

    for key in covariate_keys:
        if key not in COVARIATE_SPECS:
            continue
        spec = COVARIATE_SPECS[key]
        values = pd.to_numeric(selected[spec.source_col], errors="coerce")

        if spec.standardize:
            mean = float(values.mean()) if values.notna().any() else 0.0
            std = float(values.std(ddof=0)) if values.notna().any() else 0.0
            if np.isclose(std, 0.0):
                standardized = values.fillna(mean) - mean
            else:
                standardized = (values.fillna(mean) - mean) / std
            if float(np.nanmax(np.abs(standardized.to_numpy(dtype=float)))) <= 1e-12:
                dropped_covariates.append(key)
                continue
            selected[spec.source_col] = standardized.astype(float)
            covariate_standardization[key] = {"mean": mean, "std": std}
        else:
            binary_values = values.fillna(0.0).astype(float)
            if float(binary_values.max()) <= 0.0:
                dropped_covariates.append(key)
                continue
            selected[spec.source_col] = binary_values
            covariate_standardization[key] = {"mean": 0.0, "std": 1.0}

        covariate_params[key] = spec.beta_param
        normalized_covariates.append(key)

    selected = selected.sort_values("time_utc").reset_index(drop=True)
    selected["time_index"] = np.arange(1, len(selected) + 1, dtype=float)
    decision_window_frame = build_decision_windows(selected)

    return HourlyStepModelData(
        participant_id=participant_id,
        study_user_id=study_user_id,
        hourly_frame=selected,
        covariate_keys=normalized_covariates,
        covariate_params=covariate_params,
        covariate_standardization=covariate_standardization,
        decision_window_frame=decision_window_frame,
        dropped_covariates=dropped_covariates,
        start_utc=start_utc,
        end_utc=end_utc,
        assumed_timezone=assumed_timezone,
    )


def build_step_pomp_default_params(hourly_data: HourlyStepModelData) -> dict[str, float]:
    defaults = {"phi": 0.75, "sigma": 0.30, "k_fitbit": 10}
    for cov_key in hourly_data.covariate_keys:
        defaults[COVARIATE_SPECS[cov_key].beta_param] = 0.0
    return defaults


def get_cached_participant_hourly(
    cache: ShelveCache,
    participant_id: str,
    *,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    lookback_hours: int,
) -> pd.DataFrame:
    cache_inputs = {
        "participant_id": participant_id,
        "analysis_start_utc_iso": start_utc.isoformat(),
        "analysis_end_utc_iso": end_utc.isoformat(),
        "lookback_hours": int(lookback_hours),
    }
    return cache.get_or_compute(
        "participant_hourly_frame",
        cache_inputs,
        lambda: load_participant_hourly_from_csv(
            cache.cfg.csv_path,
            participant_id,
            start_utc.isoformat(),
            end_utc.isoformat(),
            lookback_hours=lookback_hours,
        ),
        prepare_for_store=lambda frame: frame.copy(deep=True),
    )


def get_cached_hourly_model_data(
    cache: ShelveCache,
    participant_hourly: pd.DataFrame,
    *,
    participant_id: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    covariate_keys: list[str],
    assumed_timezone: str,
) -> HourlyStepModelData:
    cache_inputs = {
        "participant_id": participant_id,
        "analysis_start_utc_iso": start_utc.isoformat(),
        "analysis_end_utc_iso": end_utc.isoformat(),
        "covariate_keys": list(covariate_keys),
        "assumed_timezone": assumed_timezone,
    }
    return cache.get_or_compute(
        "participant_hourly_model_data",
        cache_inputs,
        lambda: prepare_hourly_model_data(
            participant_hourly,
            participant_id=participant_id,
            start_utc=start_utc,
            end_utc=end_utc,
            covariate_keys=covariate_keys,
            assumed_timezone=assumed_timezone,
        ),
        prepare_for_store=clone_hourly_step_model_data,
    )


# -----------------------------
# POMP / IF2 / search (your code, moved)
# (omitted comments for length; logic unchanged)
# -----------------------------
def _build_theta0(initial_params: dict[str, float], covariate_keys: list[str]) -> dict[str, float]:
    theta0 = {"phi": float(initial_params["phi"]), "sigma": float(initial_params["sigma"]), "k_fitbit": float(initial_params["k_fitbit"])}
    for cov_key in covariate_keys:
        theta0[COVARIATE_SPECS[cov_key].beta_param] = float(initial_params.get(COVARIATE_SPECS[cov_key].beta_param, 0.0))
    return theta0

def _softplus(value: jax.Array) -> jax.Array:
    return jnp.log1p(jnp.exp(-jnp.abs(value))) + jnp.maximum(value, 0)

def _inv_softplus(value: jax.Array) -> jax.Array:
    value = jnp.maximum(value, 1e-8)
    return jnp.where(value > 20.0, value, jnp.log(jnp.expm1(value)))

def _build_par_trans(theta0: dict[str, float]) -> ParTrans:
    positive_names = [name for name in theta0 if name in {"sigma", "k_fitbit"}]

    def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        result = dict(theta)
        for name in positive_names:
            result[name] = _inv_softplus(result[name])
        scaled = jnp.clip((result["phi"] + 1.0) / 2.0, 1e-6, 1.0 - 1e-6)
        result["phi"] = jnp.log(scaled / (1.0 - scaled))
        return result

    def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        result = dict(theta)
        for name in positive_names:
            result[name] = _softplus(result[name]) + 1e-6
        scaled = 1.0 / (1.0 + jnp.exp(-result["phi"]))
        result["phi"] = 2.0 * scaled - 1.0
        return result

    return ParTrans(to_est=to_est, from_est=from_est)

def _poisson_sample(key: jax.Array, mean: jax.Array) -> jax.Array:
    mean = jnp.maximum(mean, 1e-8)
    return jax.random.poisson(key, mean)

def _nb_sample_mean_disp(key: jax.Array, mean: jax.Array, disp: jax.Array) -> jax.Array:
    mean = jnp.maximum(mean, 1e-8)
    disp = jnp.maximum(disp, 1e-6)
    key_gamma, key_poisson = jax.random.split(key)
    rate = jax.random.gamma(key_gamma, disp) * (mean / disp)
    return jax.random.poisson(key_poisson, rate)

def _nb_logpmf_mean_disp(obs: jax.Array, mean: jax.Array, disp: jax.Array) -> jax.Array:
    obs = jnp.maximum(jnp.round(obs), 0.0)
    mean = jnp.maximum(mean, 1e-8)
    disp = jnp.maximum(disp, 1e-6)
    return (
        gammaln(obs + disp) - gammaln(disp) - gammaln(obs + 1.0)
        + disp * (jnp.log(disp) - jnp.log(disp + mean))
        + obs * (jnp.log(mean) - jnp.log(disp + mean))
    )

def _build_pomp_object(data: HourlyStepModelData, theta0: dict[str, float]) -> Pomp:
    cov_pairs = [(COVARIATE_SPECS[k].source_col, COVARIATE_SPECS[k].beta_param) for k in data.covariate_keys]
    par_trans = _build_par_trans(theta0)

    def rinit(theta_, key, covars, t0):
        return {"x": 0.0, "N": 1.0}

    def rproc(X_, theta_, key, covars, t, dt):
        key_x, key_n = jax.random.split(key)
        x_next = theta_["phi"] * X_["x"] + theta_["sigma"] * jax.random.normal(key_x)

        mean_term = covars.get("baseline_log_mean", 0.0)
        for covar_name, param_name in cov_pairs:
            mean_term = mean_term + theta_[param_name] * covars.get(covar_name, 0.0)

        log_lambda = jnp.clip(mean_term + x_next, -6.0, 12.0)
        lam = jnp.exp(log_lambda)
        latent_count = _poisson_sample(key_n, lam)
        return {"x": x_next, "N": latent_count}

    def dmeas(Y_, X_, theta_, covars, t):
        observed = Y_["fitbit_steps_obs"]
        mask = ~jnp.isnan(observed)
        return jnp.where(mask, _nb_logpmf_mean_disp(observed, X_["N"], theta_["k_fitbit"]), 0.0)

    def rmeas(X_, theta_, key, covars, t):
        draw = _nb_sample_mean_disp(key, X_["N"], theta_["k_fitbit"])
        return jnp.array([draw])

    ys = data.ys.copy()
    covars = data.covars.copy()
    ys.index = ys.index.astype(float)
    covars.index = covars.index.astype(float)

    return Pomp(
        ys=ys,
        theta=theta0,
        statenames=["x", "N"],
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        par_trans=par_trans,
        covars=covars,
        nstep=1,
        ydim=1,
    )

def _coerce_param_row(row: pd.Series) -> dict[str, float]:
    result = {}
    for key, value in row.items():
        if key in {"logLik", "se"}:
            continue
        try:
            result[str(key)] = float(value)
        except Exception:
            continue
    return result

def _extract_filtered_states(pomp: Pomp, data: HourlyStepModelData) -> pd.DataFrame:
    result = pomp.results_history.last()
    filter_mean = getattr(result, "filter_mean", None)
    if filter_mean is None:
        raise ValueError("Filtered means were not stored by the particle filter.")

    arr = np.asarray(filter_mean)
    state_mean = np.nanmean(arr[0], axis=0)
    state_q10 = np.nanquantile(arr[0], 0.1, axis=0)
    state_q90 = np.nanquantile(arr[0], 0.9, axis=0)

    frame = data.hourly_frame[["time_index", "time_utc", "local_time", "local_date", "local_hour", "local_dayofweek"]].copy()
    frame["x_filtered_mean"] = state_mean[:, 0]
    frame["N_filtered_mean"] = state_mean[:, 1]
    frame["x_filter_rep_q10"] = state_q10[:, 0]
    frame["x_filter_rep_q90"] = state_q90[:, 0]
    frame["N_filter_rep_q10"] = state_q10[:, 1]
    frame["N_filter_rep_q90"] = state_q90[:, 1]
    return frame

def _build_plot_frame(data: HourlyStepModelData, filtered_states: pd.DataFrame, fitted_params: dict[str, float]) -> pd.DataFrame:
    frame = data.hourly_frame.copy()
    frame = frame.merge(filtered_states, on=["time_index", "time_utc", "local_time", "local_date", "local_hour", "local_dayofweek"], how="left")
    frame["latent_fitbit_scale_steps"] = frame["N_filtered_mean"]

    linear_adjustment = np.zeros(len(frame), dtype=float)
    for cov_key in data.covariate_keys:
        spec = COVARIATE_SPECS[cov_key]
        linear_adjustment = linear_adjustment + float(fitted_params.get(spec.beta_param, 0.0)) * pd.to_numeric(
            frame[spec.source_col], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)

    frame["model_log_lambda"] = (
        pd.to_numeric(frame["baseline_log_mean"], errors="coerce").fillna(0.0)
        + pd.Series(linear_adjustment, index=frame.index)
        + pd.to_numeric(frame["x_filtered_mean"], errors="coerce").fillna(0.0)
    )
    frame["model_lambda"] = np.exp(np.clip(frame["model_log_lambda"], -6.0, 12.0))
    frame["fitbit_steps_obs_fitted"] = frame["N_filtered_mean"]
    frame["fitbit_hour_status"] = np.where(frame["fitbit_observed"].astype(int) == 1, "observed", "missing")
    return frame

def _build_reward_frame(data: HourlyStepModelData, plot_frame: pd.DataFrame) -> pd.DataFrame:
    if data.decision_window_frame.empty:
        return data.decision_window_frame.copy()

    rows = []
    for _, reward_row in data.decision_window_frame.iterrows():
        start_pos = int(reward_row["window_start_position"])
        end_pos = int(reward_row["window_end_position"])
        window = plot_frame.iloc[start_pos : end_pos + 1].copy()
        rows.append(
            {
                **reward_row.to_dict(),
                "latent_reward_24h": float(pd.to_numeric(window["N_filtered_mean"], errors="coerce").sum(min_count=1)),
                "fitbit_observed_reward_24h": float(pd.to_numeric(window["fitbit_steps_obs"], errors="coerce").sum(min_count=1)),
                "fitbit_fitted_reward_24h": float(pd.to_numeric(window["fitbit_steps_obs_fitted"], errors="coerce").sum(min_count=1)),
            }
        )
    return pd.DataFrame(rows)

def run_step_pomp_fit(
    data: HourlyStepModelData,
    *,
    initial_params: dict[str, float],
    free_params: list[str] | None,
    particles: int,
    mif_iterations: int,
    random_seed: int,
    cooling_fraction: float,
    rw_sd_scale: float,
    evaluation_particles: int | None = None,
    evaluation_pfilter_reps: int = 1,
) -> StepPompFitResult:
    fit_start = perf_counter()
    theta0 = _build_theta0(initial_params, data.covariate_keys)
    pomp = _build_pomp_object(data, theta0)
    free_param_names = theta0.keys() if free_params is None else [name for name in free_params if name in theta0]

    if free_param_names and int(mif_iterations) > 0:
        rw_sd = RWSigma({name: (float(rw_sd_scale) if name in free_param_names else 0.0) for name in theta0})
        mif_key = jax.random.key(int(random_seed))
        pomp.mif(J=int(particles), M=int(mif_iterations), rw_sd=rw_sd, a=float(cooling_fraction), key=mif_key)
        mif_summary = pomp.results(index=-1).copy()
        mif_traces = pomp.traces().copy()
    else:
        mif_summary = pd.DataFrame([{"method": "fixed_theta", **theta0}])
        mif_traces = pd.DataFrame()

    eval_particles = int(evaluation_particles or particles)
    eval_reps = max(1, int(evaluation_pfilter_reps))
    pfilter_key = jax.random.key(int(random_seed) + 1)
    pomp.pfilter(J=eval_particles, reps=eval_reps, key=pfilter_key, filter_mean=True)

    pfilter_summary = pomp.results(index=-1).copy()
    timing_frame = pomp.time().copy()
    fitted_params = _coerce_param_row(pfilter_summary.iloc[0] if not pfilter_summary.empty else pd.Series(theta0))
    for name, value in theta0.items():
        if name not in free_param_names:
            fitted_params[name] = float(value)

    loglik = float(pfilter_summary["logLik"].iloc[0]) if (not pfilter_summary.empty and "logLik" in pfilter_summary.columns) else float("nan")
    runtime_breakdown = pd.DataFrame([{"step": "fit_total", "seconds": float(perf_counter() - fit_start)}])

    return StepPompFitResult(
        initial_params={name: float(value) for name, value in theta0.items()},
        fitted_params=fitted_params,
        loglik=loglik,
        pomp_object=pomp,
        mif_summary=mif_summary,
        mif_traces=mif_traces,
        pfilter_summary=pfilter_summary,
        timing_frame=timing_frame,
        runtime_breakdown=runtime_breakdown,
    )

def run_step_pomp_if2(
    data: HourlyStepModelData,
    *,
    initial_params: dict[str, float],
    free_params: list[str] | None,
    particles: int,
    mif_iterations: int,
    random_seed: int,
    cooling_fraction: float,
    rw_sd_scale: float,
    evaluation_particles: int | None = None,
    evaluation_pfilter_reps: int = 1,
) -> StepPompRunResult:
    fit_result = run_step_pomp_fit(
        data,
        initial_params=initial_params,
        free_params=free_params,
        particles=particles,
        mif_iterations=mif_iterations,
        random_seed=random_seed,
        cooling_fraction=cooling_fraction,
        rw_sd_scale=rw_sd_scale,
        evaluation_particles=evaluation_particles,
        evaluation_pfilter_reps=evaluation_pfilter_reps,
    )
    filtered_states = _extract_filtered_states(fit_result.pomp_object, data)
    plot_frame = _build_plot_frame(data, filtered_states, fit_result.fitted_params)
    reward_frame = _build_reward_frame(data, plot_frame)

    return StepPompRunResult(
        initial_params=fit_result.initial_params,
        fitted_params=fit_result.fitted_params,
        loglik=fit_result.loglik,
        pomp_object=fit_result.pomp_object,
        mif_summary=fit_result.mif_summary,
        mif_traces=fit_result.mif_traces,
        pfilter_summary=fit_result.pfilter_summary,
        filtered_states=filtered_states,
        plot_frame=plot_frame,
        reward_frame=reward_frame,
        timing_frame=fit_result.timing_frame,
        runtime_breakdown=fit_result.runtime_breakdown,
    )


# -----------------------------
# Global search (same as your QMD)
# -----------------------------
def _coerce_free_params(free_params: list[str] | None, initial_params: dict[str, float]) -> list[str]:
    if free_params is None:
        return list(initial_params.keys())
    return [name for name in free_params if name in initial_params]

def _coerce_start_box(start_box: dict[str, tuple[float, float]] | None, free_params: list[str], initial_params: dict[str, float]) -> dict[str, tuple[float, float]]:
    result = {}
    start_box = start_box or {}
    for name in free_params:
        raw_bounds = start_box.get(name)
        if raw_bounds is None:
            value = float(initial_params[name])
            width = max(abs(value) * 0.25, 0.10)
            result[name] = (value - width, value + width)
            continue
        low, high = float(raw_bounds[0]), float(raw_bounds[1])
        if high < low:
            low, high = high, low
        if np.isclose(low, high):
            eps = max(abs(low) * 0.01, 1e-3)
            low, high = low - eps, high + eps
        result[name] = (low, high)
    return result

def _build_start_box_frame(start_box: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"parameter": n, "start_low": b[0], "start_high": b[1], "start_width": b[1]-b[0]} for n, b in sorted(start_box.items())])

def _stage_bounds(base_start_box: dict[str, tuple[float, float]], center_params: dict[str, float], *, stage_index: int, box_shrink_factor_per_stage: float) -> dict[str, tuple[float, float]]:
    shrink_factor = float(np.clip(box_shrink_factor_per_stage, 1e-3, 1.0)) ** int(max(stage_index, 0))
    result = {}
    for name, (base_low, base_high) in base_start_box.items():
        if shrink_factor >= 0.999999:
            result[name] = (float(base_low), float(base_high))
            continue
        width = float(base_high - base_low) * shrink_factor
        center = float(np.clip(center_params.get(name, 0.5 * (base_low + base_high)), base_low, base_high))
        low = max(float(base_low), center - 0.5 * width)
        high = min(float(base_high), center + 0.5 * width)
        if np.isclose(low, high):
            eps = max(abs(center) * 0.01, 1e-3)
            low = max(float(base_low), center - eps)
            high = min(float(base_high), center + eps)
        result[name] = (float(low), float(high))
    return result

def _sample_uniform_start(template_params: dict[str, float], free_params: list[str], bounds: dict[str, tuple[float, float]], rng: np.random.Generator) -> dict[str, float]:
    params = {n: float(v) for n, v in template_params.items()}
    for n in free_params:
        low, high = bounds[n]
        params[n] = float(rng.uniform(low, high))
    return params

def _sample_jittered_start(center_params: dict[str, float], template_params: dict[str, float], free_params: list[str], bounds: dict[str, tuple[float, float]], rng: np.random.Generator, *, jitter_scale: float) -> dict[str, float]:
    params = {n: float(v) for n, v in template_params.items()}
    for n in free_params:
        base_value = float(center_params.get(n, template_params[n]))
        low, high = bounds[n]
        width = max(float(high - low), 1e-6)
        jittered = base_value + rng.normal(scale=float(max(jitter_scale, 1e-6)) * width)
        params[n] = float(np.clip(jittered, low, high))
    return params

def _build_stage_candidates(*, stage_index: int, stage: GlobalSearchStageConfig, template_params: dict[str, float], free_params: list[str], bounds: dict[str, tuple[float, float]], previous_survivors: list[dict[str, object]], include_current_theta_as_start: bool, jitter_scale: float, rng: np.random.Generator) -> list[dict[str, object]]:
    n_starts = int(max(stage.n_starts, 1))
    candidates = []

    if stage_index == 0:
        if include_current_theta_as_start:
            candidates.append({"origin": "current_theta", "parent_candidate_id": None, "params": {n: float(v) for n, v in template_params.items()}})
        while len(candidates) < n_starts:
            candidates.append({"origin": "random_box", "parent_candidate_id": None, "params": _sample_uniform_start(template_params, free_params, bounds, rng)})
        return candidates

    survivors = previous_survivors[:n_starts]
    for s in survivors:
        candidates.append({"origin": "survivor", "parent_candidate_id": s["candidate_id"], "params": dict(s["fitted_params"])})

    remaining = n_starts - len(candidates)
    jitter_budget = min(int(max(stage.n_jittered_restarts, 0)), remaining)
    for _ in range(jitter_budget):
        anchor = previous_survivors[int(rng.integers(0, len(previous_survivors)))]
        candidates.append(
            {
                "origin": "jittered_survivor",
                "parent_candidate_id": anchor["candidate_id"],
                "params": _sample_jittered_start(anchor["fitted_params"], template_params, free_params, bounds, rng, jitter_scale=jitter_scale),
            }
        )

    remaining = n_starts - len(candidates)
    fresh_budget = min(int(max(stage.n_fresh_random_starts, 0)), remaining)
    for _ in range(fresh_budget):
        candidates.append({"origin": "fresh_random", "parent_candidate_id": None, "params": _sample_uniform_start(template_params, free_params, bounds, rng)})

    while len(candidates) < n_starts:
        candidates.append({"origin": "random_backfill", "parent_candidate_id": None, "params": _sample_uniform_start(template_params, free_params, bounds, rng)})

    return candidates


def run_multistage_step_pomp_search(
    data: HourlyStepModelData,
    *,
    initial_params: dict[str, float],
    free_params: list[str] | None,
    start_box: dict[str, tuple[float, float]] | None,
    stage_configs: list[GlobalSearchStageConfig],
    random_seed: int,
    include_current_theta_as_start: bool = True,
    jitter_scale: float = 0.10,
    box_shrink_factor_per_stage: float = 1.0,
) -> GlobalSearchResult:
    normalized_initial = {name: float(value) for name, value in initial_params.items()}
    normalized_free_params = _coerce_free_params(free_params, normalized_initial)
    if not normalized_free_params:
        raise ValueError("Select at least one parameter to estimate in the global MLE search.")
    if not stage_configs:
        raise ValueError("Provide at least one stage for the global search.")

    normalized_start_box = _coerce_start_box(start_box, normalized_free_params, normalized_initial)
    start_box_frame = _build_start_box_frame(normalized_start_box)

    previous_survivors = []
    current_center = dict(normalized_initial)
    best_fit = None
    best_candidate_row = None
    all_rows = []
    stage_summary_rows = []
    total_runtime_seconds = 0.0
    rng = np.random.default_rng(int(random_seed))

    for stage_index, stage in enumerate(stage_configs):
        stage_bounds = _stage_bounds(normalized_start_box, current_center, stage_index=stage_index, box_shrink_factor_per_stage=box_shrink_factor_per_stage)
        stage_candidates = _build_stage_candidates(
            stage_index=stage_index,
            stage=stage,
            template_params=normalized_initial,
            free_params=normalized_free_params,
            bounds=stage_bounds,
            previous_survivors=previous_survivors,
            include_current_theta_as_start=include_current_theta_as_start,
            jitter_scale=jitter_scale,
            rng=rng,
        )

        stage_rows = []
        for candidate_index, candidate in enumerate(stage_candidates, start=1):
            candidate_id = f"stage_{stage_index + 1:02d}_cand_{candidate_index:03d}"
            fit_seed = int(random_seed) + stage_index * 10_000 + candidate_index
            fit_start = perf_counter()
            fitted_params = {}
            fit_initial_params = dict(candidate["params"])
            loglik = float("nan")
            error_text = ""
            fit_succeeded = False
            fit_result = None

            try:
                fit_result = run_step_pomp_fit(
                    data,
                    initial_params=fit_initial_params,
                    free_params=normalized_free_params,
                    particles=stage.particles,
                    mif_iterations=stage.mif_iterations,
                    random_seed=fit_seed,
                    cooling_fraction=stage.cooling_fraction,
                    rw_sd_scale=stage.rw_sd_scale,
                    evaluation_particles=stage.evaluation_particles,
                    evaluation_pfilter_reps=stage.evaluation_pfilter_reps,
                )
                fit_succeeded = True
                loglik = float(fit_result.loglik)
                fitted_params = dict(fit_result.fitted_params)
                fit_initial_params = dict(fit_result.initial_params)
                if best_fit is None or (np.isfinite(loglik) and (not np.isfinite(best_fit.loglik) or loglik > float(best_fit.loglik))):
                    best_fit = fit_result
            except Exception as exc:
                error_text = str(exc)

            runtime_seconds = float(perf_counter() - fit_start)
            total_runtime_seconds += runtime_seconds

            row = {
                "stage_index": stage_index + 1,
                "stage_name": stage.name,
                "candidate_id": candidate_id,
                "parent_candidate_id": candidate["parent_candidate_id"],
                "origin": str(candidate["origin"]),
                "fit_seed": fit_seed,
                "fit_succeeded": fit_succeeded,
                "loglik": loglik,
                "runtime_seconds": runtime_seconds,
                "particles": stage.particles,
                "mif_iterations": stage.mif_iterations,
                "cooling_fraction": stage.cooling_fraction,
                "rw_sd_scale": stage.rw_sd_scale,
                "evaluation_particles": stage.evaluation_particles,
                "evaluation_pfilter_reps": stage.evaluation_pfilter_reps,
                "error": error_text,
            }
            for name in sorted(normalized_initial):
                row[f"start_{name}"] = float(fit_initial_params.get(name, normalized_initial[name]))
                row[f"fitted_{name}"] = float(fitted_params[name]) if name in fitted_params else np.nan
            stage_rows.append(row)

            if fit_succeeded and fit_result is not None:
                if best_candidate_row is None or (np.isfinite(loglik) and (not np.isfinite(float(best_candidate_row.get("loglik", np.nan))) or loglik > float(best_candidate_row["loglik"]))):
                    best_candidate_row = dict(row)

        stage_frame = pd.DataFrame(stage_rows).sort_values("loglik", ascending=False, na_position="last").reset_index(drop=True)
        stage_frame["stage_rank"] = np.arange(1, len(stage_frame) + 1, dtype=int)

        successful_frame = stage_frame.loc[stage_frame["fit_succeeded"]].copy()
        keep_count = min(int(stage.keep_top), len(successful_frame))
        kept_ids = set(successful_frame.head(keep_count)["candidate_id"].astype(str).tolist())
        stage_frame["kept_for_next_stage"] = stage_frame["candidate_id"].astype(str).isin(kept_ids)
        all_rows.extend(stage_frame.to_dict("records"))

        previous_survivors = [
            {
                "candidate_id": str(r["candidate_id"]),
                "fitted_params": {name: float(r[f"fitted_{name}"]) for name in normalized_initial if pd.notna(r.get(f"fitted_{name}"))},
            }
            for r in stage_frame.loc[stage_frame["kept_for_next_stage"]].to_dict("records")
        ]
        if previous_survivors:
            current_center = dict(previous_survivors[0]["fitted_params"])

        stage_summary_rows.append(
            {
                "stage_index": stage_index + 1,
                "stage_name": stage.name,
                "n_starts": stage.n_starts,
                "n_successful": int(successful_frame["fit_succeeded"].sum()) if "fit_succeeded" in successful_frame.columns else int(len(successful_frame)),
                "keep_top": stage.keep_top,
                "best_loglik": float(pd.to_numeric(stage_frame["loglik"], errors="coerce").max()) if stage_frame["loglik"].notna().any() else np.nan,
                "median_loglik": float(pd.to_numeric(successful_frame["loglik"], errors="coerce").median()) if not successful_frame.empty else np.nan,
                "runtime_seconds": float(pd.to_numeric(stage_frame["runtime_seconds"], errors="coerce").sum()),
                "fresh_random_starts": stage.n_fresh_random_starts,
                "jittered_restarts": stage.n_jittered_restarts,
                "stage_best_candidate_id": str(stage_frame.iloc[0]["candidate_id"]),
            }
        )

    if best_fit is None or best_candidate_row is None:
        raise ValueError("No successful IF2 fits were produced in the global search.")

    best_run = run_step_pomp_if2(
        data,
        initial_params=dict(best_fit.fitted_params),
        free_params=[],
        particles=int(best_candidate_row["evaluation_particles"]),
        mif_iterations=0,
        random_seed=int(best_candidate_row["fit_seed"]),
        cooling_fraction=float(best_candidate_row["cooling_fraction"]),
        rw_sd_scale=float(best_candidate_row["rw_sd_scale"]),
        evaluation_particles=int(best_candidate_row["evaluation_particles"]),
        evaluation_pfilter_reps=int(best_candidate_row["evaluation_pfilter_reps"]),
    )

    candidate_frame = pd.DataFrame(all_rows).sort_values(["loglik", "stage_index", "stage_rank"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
    stage_summary = pd.DataFrame(stage_summary_rows).sort_values("stage_index").reset_index(drop=True)

    return GlobalSearchResult(
        best_fit=best_fit,
        best_run=best_run,
        candidate_frame=candidate_frame,
        stage_summary=stage_summary,
        start_box_frame=start_box_frame,
        total_runtime_seconds=float(total_runtime_seconds),
    )


def get_cached_global_search_result(
    cache: ShelveCache,
    data: HourlyStepModelData,
    *,
    participant_id: str,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    initial_params: dict[str, float],
    free_params: list[str] | None,
    start_box: dict[str, tuple[float, float]] | None,
    stage_configs: list[GlobalSearchStageConfig],
    random_seed: int,
    include_current_theta_as_start: bool,
    jitter_scale: float,
    box_shrink_factor_per_stage: float,
) -> GlobalSearchResult:
    stage_config_snapshot = [
        {
            "name": s.name,
            "n_starts": int(s.n_starts),
            "keep_top": int(s.keep_top),
            "particles": int(s.particles),
            "mif_iterations": int(s.mif_iterations),
            "cooling_fraction": float(s.cooling_fraction),
            "rw_sd_scale": float(s.rw_sd_scale),
            "evaluation_particles": int(s.evaluation_particles),
            "evaluation_pfilter_reps": int(s.evaluation_pfilter_reps),
            "n_fresh_random_starts": int(s.n_fresh_random_starts),
            "n_jittered_restarts": int(s.n_jittered_restarts),
        }
        for s in stage_configs
    ]
    cache_inputs = {
        "participant_id": participant_id,
        "analysis_start_utc_iso": analysis_start.isoformat(),
        "analysis_end_utc_iso": analysis_end.isoformat(),
        "model_covariate_keys": list(data.covariate_keys),
        "initial_params": copy.deepcopy(initial_params),
        "free_params": list(free_params or []),
        "start_box": copy.deepcopy(start_box),
        "stage_configs": stage_config_snapshot,
        "random_seed": int(random_seed),
        "include_current_theta_as_start": bool(include_current_theta_as_start),
        "jitter_scale": float(jitter_scale),
        "box_shrink_factor_per_stage": float(box_shrink_factor_per_stage),
    }
    return cache.get_or_compute(
        "global_step_pomp_search_result",
        cache_inputs,
        lambda: run_multistage_step_pomp_search(
            data,
            initial_params=initial_params,
            free_params=free_params,
            start_box=start_box,
            stage_configs=stage_configs,
            random_seed=random_seed,
            include_current_theta_as_start=include_current_theta_as_start,
            jitter_scale=jitter_scale,
            box_shrink_factor_per_stage=box_shrink_factor_per_stage,
        ),
        prepare_for_store=strip_global_search_result,
    )


def get_cached_local_step_pomp_result(
    cache: ShelveCache,
    data: HourlyStepModelData,
    *,
    participant_id: str,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    initial_params: dict[str, float],
    free_params: list[str] | None,
    local_if2_controls: dict[str, float],
) -> StepPompRunResult:
    cache_inputs = {
        "participant_id": participant_id,
        "analysis_start_utc_iso": analysis_start.isoformat(),
        "analysis_end_utc_iso": analysis_end.isoformat(),
        "model_covariate_keys": list(data.covariate_keys),
        "initial_params": copy.deepcopy(initial_params),
        "free_params": list(free_params or []),
        "local_if2_controls": copy.deepcopy(local_if2_controls),
    }
    return cache.get_or_compute(
        "local_step_pomp_result",
        cache_inputs,
        lambda: run_step_pomp_if2(
            data,
            initial_params=initial_params,
            free_params=free_params,
            particles=local_if2_controls["particles"],
            mif_iterations=local_if2_controls["mif_iterations"],
            random_seed=local_if2_controls["random_seed"],
            cooling_fraction=local_if2_controls["cooling_fraction"],
            rw_sd_scale=local_if2_controls["rw_sd_scale"],
            evaluation_particles=local_if2_controls["evaluation_particles"],
            evaluation_pfilter_reps=local_if2_controls["evaluation_pfilter_reps"],
        ),
        prepare_for_store=strip_step_pomp_run_result,
    )


# -----------------------------
# Single-call "run one participant"
# -----------------------------
def run_participant(participant_id: str, *, cache_cfg: CacheConfig, cfg: AnalysisConfig) -> ParticipantResult:
    cache = ShelveCache(cache_cfg)

    analysis_start = normalize_timestamp_to_utc(cfg.analysis_start_utc_iso)
    analysis_end = normalize_timestamp_to_utc(cfg.analysis_end_utc_iso)
    covariate_keys = list(cfg.participant_covariate_keys or [])
    free_params = list(cfg.estimated_free_params or [])

    initial_param_overrides = {"phi": float(cfg.initial_phi), "sigma": float(cfg.initial_sigma), "k_fitbit": float(cfg.initial_k_fitbit)}
    local_if2_controls = {
        "particles": int(cfg.local_mle_particles),
        "mif_iterations": int(cfg.local_mle_mif_iterations),
        "random_seed": int(cfg.local_mle_random_seed),
        "cooling_fraction": float(cfg.local_mle_cooling_fraction),
        "rw_sd_scale": float(cfg.local_mle_rw_sd_scale),
        "evaluation_particles": int(cfg.local_mle_evaluation_particles),
        "evaluation_pfilter_reps": int(cfg.local_mle_evaluation_pfilter_reps),
    }
    global_search_start_box = {
        "phi": (float(cfg.global_start_box_phi_low), float(cfg.global_start_box_phi_high)),
        "sigma": (float(cfg.global_start_box_sigma_low), float(cfg.global_start_box_sigma_high)),
        "k_fitbit": (float(cfg.global_start_box_k_fitbit_low), float(cfg.global_start_box_k_fitbit_high)),
    }
    global_search_stage_configs = [GlobalSearchStageConfig(**d) for d in (cfg.global_search_stage_controls or [])]

    participant_hourly = get_cached_participant_hourly(
        cache,
        participant_id,
        start_utc=analysis_start,
        end_utc=analysis_end,
        lookback_hours=int(cfg.participant_data_lookback_hours),
    )
    hourly_data = get_cached_hourly_model_data(
        cache,
        participant_hourly,
        participant_id=participant_id,
        start_utc=analysis_start,
        end_utc=analysis_end,
        covariate_keys=covariate_keys,
        assumed_timezone=str(cfg.assumed_local_timezone),
    )

    initial_params = build_step_pomp_default_params(hourly_data)
    for k, v in initial_param_overrides.items():
        if k in initial_params:
            initial_params[k] = float(v)

    active_free_params = [p for p in free_params if p in initial_params]

    fitbit_missingness_percentage = 100.0 * (1.0 - float(hourly_data.hourly_frame["fitbit_observed"].mean()))
    participant_summary = pd.DataFrame(
        [{
            "Participant ID": participant_id,
            "Complete 24 hour reward windows": int(hourly_data.decision_window_frame["complete_window"].sum()),
            "Fitbit missingness percentage": float(fitbit_missingness_percentage),
        }]
    )

    local_run = get_cached_local_step_pomp_result(
        cache,
        hourly_data,
        participant_id=participant_id,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        initial_params=initial_params,
        free_params=active_free_params,
        local_if2_controls=local_if2_controls,
    )
    local_mle_summary = pd.DataFrame([{"Search": "Local MLE", "Participant ID": participant_id, "LogLik": local_run.loglik, **local_run.fitted_params}])

    global_search_result = get_cached_global_search_result(
        cache,
        hourly_data,
        participant_id=participant_id,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        initial_params=initial_params,
        free_params=active_free_params,
        start_box=global_search_start_box,
        stage_configs=global_search_stage_configs,
        random_seed=int(cfg.global_search_random_seed),
        include_current_theta_as_start=bool(cfg.global_search_include_current_theta_as_start),
        jitter_scale=float(cfg.global_search_jitter_scale),
        box_shrink_factor_per_stage=float(cfg.global_search_box_shrink_factor_per_stage),
    )

    global_best_fit = global_search_result.best_fit
    global_best_run = global_search_result.best_run
    global_best_summary = pd.DataFrame([{"Search": "Global MLE", "Participant ID": participant_id, "LogLik": global_best_run.loglik, **global_best_fit.fitted_params}])

    global_stage_summary = (
        global_search_result.stage_summary[
            ["stage_name", "n_starts", "n_successful", "keep_top", "best_loglik", "runtime_seconds"]
        ]
        .rename(
            columns={
                "stage_name": "Stage",
                "n_starts": "Starts",
                "n_successful": "Successful fits",
                "keep_top": "Keep top",
                "best_loglik": "Best logLik",
                "runtime_seconds": "Runtime (s)",
            }
        )
        .copy()
    )
    global_stage_summary.insert(0, "Participant ID", participant_id)
    global_stage_summary.insert(0, "Search", "Global")

    coefficient_rows = []
    for param_name in initial_params:
        local_value = local_run.fitted_params.get(param_name, initial_params[param_name])
        global_value = global_best_fit.fitted_params.get(param_name, initial_params[param_name])
        coefficient_rows.append(
            {"Participant ID": participant_id, "Parameter": param_name, "Initial value": initial_params[param_name], "Local MLE value": local_value, "Global MLE value": global_value}
        )
    coefficient_table = pd.DataFrame(coefficient_rows)

    return ParticipantResult(
        participant_id=participant_id,
        hourly_data=hourly_data,
        initial_params=initial_params,
        active_free_params=active_free_params,
        participant_summary=participant_summary,
        local_run=local_run,
        local_mle_summary=local_mle_summary,
        global_search_result=global_search_result,
        global_best_summary=global_best_summary,
        global_stage_summary=global_stage_summary,
        coefficient_table=coefficient_table,
    )
