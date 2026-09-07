#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tdcladder.qa import TDC_PRIMARY_ANCHOR, eligible_tdc_rows

TDC_ANCHOR_COL = TDC_PRIMARY_ANCHOR
TDC_ANCHOR_LABEL = "Long-history Tier 2 regression/MMF-RRP bank-only RU flow"
CORE_WEIGHT_FAMILIES = (
    "fixed_baseline",
    "fixed_conservative_duration",
    "fixed_cash_like_generous",
)
SIDECAR_WEIGHT_FAMILIES = (
    "price_convenience_yield",
    "turnover_duration_adjusted",
    "holder_revealed_first_pass",
    "holder_stock_context_z1",
)
MONTHLY_EVENT_FLAGS = (
    ("isolated_bill_shock", "isolated_large_positive_bill_shock"),
    ("bill_heavy", "bill_heavy_episode"),
    ("coupon_heavy", "coupon_heavy_episode"),
    ("wam_extension", "wam_extension_episode"),
    ("duration_gap_heavy", "duration_gap_heavy_episode"),
    ("reserve_drain", "reserve_drain_episode"),
    ("mmf_absorption", "mmf_absorption_episode"),
    ("on_rrp_drain", "on_rrp_drain_episode"),
    ("tga_refill", "tga_refill_episode"),
    ("high_rate_start", "high_rate_episode"),
    ("qt_start", "qt_episode"),
)
QUARTERLY_EVENT_FLAGS = (
    ("bill_heavy", "bill_heavy_episode_q"),
    ("coupon_heavy", "coupon_heavy_episode_q"),
    ("wam_extension", "wam_extension_episode_q"),
    ("duration_gap_heavy", "duration_gap_heavy_episode_q"),
    ("reserve_drain", "reserve_drain_episode_q"),
    ("mmf_absorption", "mmf_absorption_episode_q"),
    ("on_rrp_drain", "on_rrp_drain_episode_q"),
    ("tga_refill", "tga_refill_episode_q"),
)


def _corr(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 3:
        return None
    return float(frame["left"].corr(frame["right"]))


def _latest_value(frame: pd.DataFrame, col: str) -> float | None:
    vals = pd.to_numeric(frame.sort_values("quarter")[col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.iloc[-1])


def _add_unit_scaled_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    treasury_usd = [
        "raw_treasury_supply",
        "liquid_treasury_supply",
        "duration_gap_supply",
        "raw_treasury_supply_q",
        "liquid_treasury_supply_q",
        "duration_gap_supply_q",
        "d_raw_treasury_supply_q",
        "d_liquid_treasury_supply_q",
    ]
    usd_millions = [
        TDC_ANCHOR_COL,
        "tdc_tier2_interest_corrected_bank_only_ru_flow",
        "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow",
        "tdc_tier2_regression_mmf_rrp_prop_depository_institution_np_cu_ru_flow",
        "d_deposits",
        "d_domestic_deposits",
        "d_total_mmf_assets",
        "d_mmf_treasury_holdings",
        "d_mmf_repo_holdings",
        "d_mmf_on_rrp_exposure",
        "d_on_rrp",
        "d_reserves",
        "d_tga",
        "d_deposits_q",
        "d_domestic_deposits_q",
        "d_total_mmf_assets_q",
        "d_mmf_treasury_holdings_q",
        "d_on_rrp_q",
        "d_reserves_q",
        "d_tga_q",
    ]
    for col in treasury_usd:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_usd"] = values
            out[f"{col}_usd_billions"] = values / 1_000_000_000
    for col in usd_millions:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_usd_millions"] = values
            out[f"{col}_usd_billions"] = values / 1_000
    return out


def _standardize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return values * np.nan
    return (values - values.mean()) / std


def _fit_standardized_model(frame: pd.DataFrame, *, y_col: str, x_cols: list[str]) -> dict[str, object] | None:
    data = frame[[y_col, *x_cols]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) <= len(x_cols) + 1:
        return None

    y = _standardize(data[y_col])
    x = data[x_cols].apply(_standardize)
    model_data = pd.concat([y.rename(y_col), x], axis=1).dropna()
    if len(model_data) <= len(x_cols) + 1:
        return None

    y_arr = model_data[y_col].to_numpy(dtype=float)
    x_arr = model_data[x_cols].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(model_data)), x_arr])
    beta, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    fitted = design @ beta
    residual = y_arr - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else np.nan
    n_obs = int(len(model_data))
    k = len(x_cols)
    adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - k - 1) if n_obs > k + 1 else np.nan
    coefs = {col: float(value) for col, value in zip(x_cols, beta[1:], strict=True)}
    return {
        "n_obs": n_obs,
        "r2": r2,
        "adj_r2": adj_r2,
        "standardized_coefficients": ";".join(f"{key}={value:.6g}" for key, value in coefs.items()),
    }


def write_tdc_model_comparison_diagnostics(quarterly: pd.DataFrame, out_dir: Path) -> None:
    quarterly = eligible_tdc_rows(quarterly)
    required = {TDC_ANCHOR_COL, "raw_treasury_supply_q", "liquid_treasury_supply_q"}
    if not required.issubset(quarterly.columns):
        return

    group_cols = ["weight_family"]
    if "supply_basis" in quarterly.columns:
        group_cols.append("supply_basis")

    model_specs = [
        ("raw_supply_only", ["raw_supply_tdc_aligned"]),
        ("liquid_supply_only", ["liquid_supply_tdc_aligned"]),
        ("bill_share_only", ["bill_share_raw_q"]),
        ("weighted_maturity_only", ["raw_weighted_maturity_years_q"]),
        ("raw_supply_bill_share_wam", ["raw_supply_tdc_aligned", "bill_share_raw_q", "raw_weighted_maturity_years_q"]),
        ("liquid_supply_bill_share_wam", ["liquid_supply_tdc_aligned", "bill_share_raw_q", "liquid_weighted_maturity_years_q"]),
    ]

    rows = []
    for keys, g in quarterly.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_info = dict(zip(group_cols, key_values, strict=True))
        g = g.sort_values("quarter").copy()
        basis = group_info.get("supply_basis", "issuance_flow")
        if basis == "outstanding_stock":
            g["raw_supply_tdc_aligned"] = pd.to_numeric(g["d_raw_treasury_supply_q"], errors="coerce")
            g["liquid_supply_tdc_aligned"] = pd.to_numeric(g["d_liquid_treasury_supply_q"], errors="coerce")
            tdc_alignment = "stock_change_vs_tdc_flow"
        else:
            g["raw_supply_tdc_aligned"] = pd.to_numeric(g["raw_treasury_supply_q"], errors="coerce")
            g["liquid_supply_tdc_aligned"] = pd.to_numeric(g["liquid_treasury_supply_q"], errors="coerce")
            tdc_alignment = "issuance_flow_vs_tdc_flow"

        for model_name, x_cols in model_specs:
            if not set(x_cols).issubset(g.columns):
                continue
            fit = _fit_standardized_model(g, y_col=TDC_ANCHOR_COL, x_cols=x_cols)
            if fit is None:
                continue
            rows.append(
                {
                    **group_info,
                    "model_name": model_name,
                    "tdc_alignment": tdc_alignment,
                    "dependent_variable": TDC_ANCHOR_COL,
                    "dependent_variable_label": TDC_ANCHOR_LABEL,
                    "predictors": ",".join(x_cols),
                    "coefficient_scale": "standardized",
                    **fit,
                    "interpretation_guardrail": "Descriptive fit only; not a causal or structural model.",
                }
            )

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "tdc_model_comparison_diagnostics.csv", index=False)


def write_ladder_interpretation_table(quarterly: pd.DataFrame, out_dir: Path) -> None:
    quarterly = eligible_tdc_rows(quarterly)
    if quarterly.empty:
        return

    group_cols = ["weight_family"]
    if "supply_basis" in quarterly.columns:
        group_cols.append("supply_basis")

    rows = []
    for keys, g in quarterly.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values, strict=True))
        g = g.sort_values("quarter").copy()
        tdc = pd.to_numeric(g[TDC_ANCHOR_COL], errors="coerce")
        row["tdc_anchor"] = TDC_ANCHOR_COL
        row["tdc_anchor_label"] = TDC_ANCHOR_LABEL
        basis = row.get("supply_basis", "issuance_flow")

        if basis == "outstanding_stock":
            raw_compare = pd.to_numeric(g["d_raw_treasury_supply_q"], errors="coerce")
            liquid_compare = pd.to_numeric(g["d_liquid_treasury_supply_q"], errors="coerce")
            row["tdc_alignment"] = "stock_change_vs_tdc_flow"
        else:
            raw_compare = pd.to_numeric(g["raw_treasury_supply_q"], errors="coerce")
            liquid_compare = pd.to_numeric(g["liquid_treasury_supply_q"], errors="coerce")
            row["tdc_alignment"] = "issuance_flow_vs_tdc_flow"

        usable = g[["quarter", "raw_treasury_supply_q", "liquid_treasury_supply_q", TDC_ANCHOR_COL]].copy()
        usable[TDC_ANCHOR_COL] = tdc
        usable = usable.dropna()
        latest_sample = g.loc[tdc.notna()].copy()
        row["first_quarter"] = usable["quarter"].min().date().isoformat() if not usable.empty else ""
        row["last_quarter"] = usable["quarter"].max().date().isoformat() if not usable.empty else ""
        row["quarters_with_tdc"] = int(tdc.notna().sum())
        row["supply_aggregation"] = g["supply_aggregation"].dropna().iloc[0] if "supply_aggregation" in g.columns and g["supply_aggregation"].notna().any() else ""
        row["mean_liquidity_weight"] = float(pd.to_numeric(g["average_liquidity_weight_q"], errors="coerce").mean())
        row["mean_duration_gap_share"] = 1.0 - row["mean_liquidity_weight"]
        row["corr_raw_supply_with_tdc_anchor"] = _corr(raw_compare, tdc)
        row["corr_liquid_supply_with_tdc_anchor"] = _corr(liquid_compare, tdc)
        if "bill_share_raw_q" in g.columns:
            bill_share = pd.to_numeric(g["bill_share_raw_q"], errors="coerce")
            row["corr_bill_share_with_tdc_anchor"] = _corr(bill_share, tdc)
            row["corr_bill_share_with_liquid_supply"] = _corr(bill_share, liquid_compare)
            row["latest_bill_share"] = _latest_value(latest_sample, "bill_share_raw_q")
        else:
            row["corr_bill_share_with_tdc_anchor"] = None
            row["corr_bill_share_with_liquid_supply"] = None
            row["latest_bill_share"] = None
        if "raw_weighted_maturity_years_q" in g.columns:
            row["latest_weighted_maturity_years"] = _latest_value(latest_sample, "raw_weighted_maturity_years_q")
        row["latest_raw_supply_q_usd_billions"] = _latest_value(latest_sample, "raw_treasury_supply_q_usd_billions") or (
            (_latest_value(latest_sample, "raw_treasury_supply_q") or np.nan) / 1_000_000_000
        )
        row["latest_liquid_supply_q_usd_billions"] = _latest_value(latest_sample, "liquid_treasury_supply_q_usd_billions") or (
            (_latest_value(latest_sample, "liquid_treasury_supply_q") or np.nan) / 1_000_000_000
        )
        row["latest_tdc_anchor_flow_usd_millions"] = _latest_value(latest_sample, f"{TDC_ANCHOR_COL}_usd_millions") or _latest_value(latest_sample, TDC_ANCHOR_COL)
        row["latest_tdc_anchor_flow_usd_billions"] = _latest_value(latest_sample, f"{TDC_ANCHOR_COL}_usd_billions") or (
            (row["latest_tdc_anchor_flow_usd_millions"] if row["latest_tdc_anchor_flow_usd_millions"] is not None else np.nan) / 1_000
        )
        row["tdc_anchor_stored_unit"] = "USD millions"
        row["treasury_supply_stored_unit"] = "USD"
        row["interpretation_guardrail"] = "Liquidity weights are measurement assumptions, not structural facts."
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "ladder_interpretation_summary.csv", index=False)
    write_claim_hierarchy_tables(table, out_dir)
    write_latest_results_overview(table, out_dir)


def _merge_support(table: pd.DataFrame, out_dir: Path, sample_window: str = "tdc_sample") -> pd.DataFrame:
    support_path = out_dir.parent / "qa" / "family_support_matrix.csv"
    if table.empty or not support_path.exists():
        return table
    support = pd.read_csv(support_path)
    support = support[support["sample_window"].eq(sample_window)].copy()
    support_cols = [
        "weight_family",
        "supply_basis",
        "row_share_model_backed",
        "dollar_share_model_backed",
        "date_min",
        "date_max",
    ]
    support = support[[c for c in support_cols if c in support.columns]].drop_duplicates(["weight_family", "supply_basis"])
    return table.merge(support, on=["weight_family", "supply_basis"], how="left")


def write_claim_hierarchy_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return

    core = summary[
        summary["weight_family"].isin(CORE_WEIGHT_FAMILIES)
        & summary["supply_basis"].eq("issuance_flow")
    ].copy()
    if not core.empty:
        core["claim_role"] = "core_measurement_band"
        core["basis_role"] = "gross_accepted_auction_issuance_flow"
        core["claim_boundary"] = (
            "Headline composition measure; compare with Tier 2 descriptively, "
            "not as a replacement TDC estimator."
        )
        core = _merge_support(core, out_dir)
        core.to_csv(out_dir / "core_measurement_summary.csv", index=False)

    sidecar = summary[summary["weight_family"].isin(SIDECAR_WEIGHT_FAMILIES)].copy()
    if not sidecar.empty:
        role_map = {
            "price_convenience_yield": "price/convenience-yield sidecar",
            "turnover_duration_adjusted": "market-liquidity turnover sidecar",
            "holder_revealed_first_pass": "auction-buyer issuance sidecar",
            "holder_stock_context_z1": "aggregate Z.1 holder-context sidecar",
        }
        sidecar["claim_role"] = sidecar["weight_family"].map(role_map).fillna("exploratory_sidecar")
        sidecar["claim_boundary"] = (
            "Exploratory diagnostic assumption; do not present as structural moneyness or final-holder evidence."
        )
        sidecar = _merge_support(sidecar, out_dir)
        sidecar.to_csv(out_dir / "exploratory_sidecar_summary.csv", index=False)


def _fmt_trillion(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"${value / 1_000:.1f}T"


def _fmt_corr(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.2f}"


def _corr_label(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "not available"
    mag = abs(float(value))
    if mag < 0.20:
        strength = "weak"
    elif mag < 0.50:
        strength = "moderate"
    else:
        strength = "strong"
    direction = "positive" if value > 0 else "negative" if value < 0 else "zero"
    return f"{strength} {direction}"


def write_latest_results_overview(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return

    rows: list[dict[str, object]] = []
    for basis, g in summary.groupby("supply_basis", dropna=False):
        g = g.copy()
        core_g = g[g["weight_family"].isin(CORE_WEIGHT_FAMILIES)].copy()
        headline_g = core_g if str(basis) == "issuance_flow" and not core_g.empty else g
        latest_quarter = str(g["last_quarter"].dropna().max()) if "last_quarter" in g.columns else ""
        latest_raw = pd.to_numeric(headline_g["latest_raw_supply_q_usd_billions"], errors="coerce").dropna()
        latest_liquid = pd.to_numeric(headline_g["latest_liquid_supply_q_usd_billions"], errors="coerce").dropna()
        mean_weight = pd.to_numeric(headline_g["mean_liquidity_weight"], errors="coerce").dropna()
        corr = pd.to_numeric(headline_g["corr_liquid_supply_with_tdc_anchor"], errors="coerce").dropna()

        if str(basis) == "outstanding_stock":
            basis_label = "outstanding stock"
            supply_text = "Treasury stock outstanding at quarter end"
            comparison_text = "stock changes are compared with quarterly TDC flows"
        else:
            basis_label = "issuance flow"
            supply_text = "Gross accepted Treasury auction issuance during the quarter"
            comparison_text = "quarterly gross accepted auction issuance is compared with quarterly TDC flows"

        liquid_min = float(latest_liquid.min()) if not latest_liquid.empty else None
        liquid_max = float(latest_liquid.max()) if not latest_liquid.empty else None
        raw_value = float(latest_raw.iloc[0]) if not latest_raw.empty else None
        weight_min = float(mean_weight.min()) if not mean_weight.empty else None
        weight_max = float(mean_weight.max()) if not mean_weight.empty else None
        corr_median = float(corr.median()) if not corr.empty else None

        if liquid_min is not None and liquid_max is not None:
            liquid_range = f"{_fmt_trillion(liquid_min)} to {_fmt_trillion(liquid_max)}"
        else:
            liquid_range = ""
        if weight_min is not None and weight_max is not None:
            weight_range = f"{weight_min:.2f} to {weight_max:.2f}"
        else:
            weight_range = ""

        rows.append(
            {
                "supply_basis": basis,
                "basis_label": basis_label,
                "latest_quarter": latest_quarter,
                "raw_supply_plain_language": supply_text,
                "tdc_alignment_plain_language": comparison_text,
                "latest_raw_supply": _fmt_trillion(raw_value),
                "latest_liquid_supply_range": liquid_range,
                "mean_liquidity_weight_range": weight_range,
                "tdc_anchor": g["tdc_anchor"].dropna().iloc[0] if "tdc_anchor" in g.columns and g["tdc_anchor"].notna().any() else TDC_ANCHOR_COL,
                "tdc_anchor_label": g["tdc_anchor_label"].dropna().iloc[0] if "tdc_anchor_label" in g.columns and g["tdc_anchor_label"].notna().any() else TDC_ANCHOR_LABEL,
                "median_corr_liquid_supply_with_tdc_anchor": corr_median,
                "correlation_plain_language": _corr_label(corr_median),
                "headline_family_scope": (
                    "fixed baseline plus fixed sensitivity band"
                    if str(basis) == "issuance_flow" and not core_g.empty
                    else "active families"
                ),
                "interpretation_guardrail": "Liquidity weights are measurement assumptions, not structural facts.",
            }
        )

    overview = pd.DataFrame(rows)
    overview.to_csv(out_dir / "latest_results_overview.csv", index=False)

    lines = [
        "# Latest tdcladder Results Overview",
        "",
        "Liquidity weights are measurement assumptions and sensitivity surfaces, not structural facts.",
        f"TDC anchor: {TDC_ANCHOR_LABEL} (`{TDC_ANCHOR_COL}`).",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['basis_label'].title()}",
                "",
                f"- Latest TDC-aligned quarter: {row['latest_quarter']}",
                f"- Raw supply object: {row['raw_supply_plain_language']}.",
                f"- Latest raw supply: {row['latest_raw_supply']}.",
                f"- Headline family scope: {row['headline_family_scope']}.",
                f"- Latest liquidity-weighted supply range: {row['latest_liquid_supply_range']}.",
                f"- Mean liquidity-weight range: {row['mean_liquidity_weight_range']}.",
                f"- Liquid-supply correlation with the TDC anchor is {_corr_label(row['median_corr_liquid_supply_with_tdc_anchor'])} "
                f"(median r = {_fmt_corr(row['median_corr_liquid_supply_with_tdc_anchor'])}).",
                f"- Alignment note: {row['tdc_alignment_plain_language']}.",
                "",
            ]
        )
    (out_dir / "latest_results_overview.md").write_text("\n".join(lines), encoding="utf-8")


def _numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce")


def _flag_top(frame: pd.DataFrame, col: str, q: float = 0.75) -> pd.Series:
    values = _numeric(frame, col)
    threshold = values.dropna().quantile(q)
    if pd.isna(threshold):
        return pd.Series(False, index=frame.index)
    return values.ge(threshold).fillna(False)


def _flag_bottom_negative(frame: pd.DataFrame, col: str, q: float = 0.25) -> pd.Series:
    values = _numeric(frame, col)
    threshold = values.dropna().quantile(q)
    if pd.isna(threshold):
        return pd.Series(False, index=frame.index)
    return (values.le(threshold) & values.lt(0)).fillna(False)


def _boolish(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).ne(0)
    return values.astype("string").str.lower().isin(["true", "1", "yes", "y"])


def _month_number(value: pd.Timestamp) -> int:
    return int(value.year) * 12 + int(value.month)


def _quarter_number(value: pd.Timestamp) -> int:
    quarter = (int(value.month) - 1) // 3 + 1
    return int(value.year) * 4 + quarter


def _episode_start_dates(panel: pd.DataFrame, *, date_col: str, flag_col: str, min_gap: int, quarterly: bool = False) -> list[pd.Timestamp]:
    if flag_col not in panel.columns:
        return []
    flagged = panel.loc[_boolish(panel, flag_col), date_col].dropna().sort_values()
    starts: list[pd.Timestamp] = []
    number = _quarter_number if quarterly else _month_number
    last_num: int | None = None
    for value in flagged:
        date = pd.Timestamp(value)
        current = number(date)
        if last_num is None or current - last_num > min_gap:
            starts.append(date)
            last_num = current
    return starts


def _window_frame(
    panel: pd.DataFrame,
    *,
    date_col: str,
    event_type: str,
    event_date: pd.Timestamp,
    pre: int,
    post: int,
    quarterly: bool = False,
) -> pd.DataFrame:
    out = panel.copy()
    number = _quarter_number if quarterly else _month_number
    event_num = number(event_date)
    out["relative_period"] = out[date_col].map(lambda value: number(pd.Timestamp(value)) - event_num)
    out = out[out["relative_period"].between(-pre, post)].copy()
    out.insert(0, "episode_type", event_type)
    out.insert(1, "event_date", event_date)
    out.insert(2, "event_id", f"{event_type}:{event_date.date().isoformat()}")
    out.insert(4, "is_event_period", out["relative_period"].eq(0))
    out["has_full_window"] = bool(out["relative_period"].min() <= -pre and out["relative_period"].max() >= post)
    return out


def _event_window_summary(windows: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    numeric_cols = [
        col
        for col in windows.columns
        if col
        not in {
            "episode_type",
            "event_date",
            "event_id",
            date_col,
            "relative_period",
            "is_event_period",
            "has_full_window",
            "episode_design_note",
        }
        and pd.api.types.is_numeric_dtype(windows[col])
    ]
    grouped = windows.groupby(["episode_type", "relative_period"], dropna=False)
    rows = grouped[numeric_cols].mean(numeric_only=True).reset_index() if numeric_cols else grouped.size().reset_index(name="window_observations")
    counts = grouped.agg(
        events=("event_id", "nunique"),
        window_observations=("event_id", "size"),
        full_window_events=("has_full_window", "sum"),
    ).reset_index()
    if numeric_cols:
        rows = rows.merge(counts, on=["episode_type", "relative_period"], how="left")
    return rows.sort_values(["episode_type", "relative_period"]).reset_index(drop=True)


def _write_episode_overlap(panel: pd.DataFrame, *, flag_pairs: tuple[tuple[str, str], ...], out_path: Path) -> None:
    flags = [(name, col) for name, col in flag_pairs if col in panel.columns]
    if not flags:
        return
    data = pd.DataFrame({name: _boolish(panel, col) for name, col in flags})
    rows = []
    for left in data.columns:
        left_count = int(data[left].sum())
        for right in data.columns:
            overlap = int((data[left] & data[right]).sum())
            rows.append(
                {
                    "episode_type": left,
                    "overlap_episode_type": right,
                    "episode_count": left_count,
                    "overlap_count": overlap,
                    "overlap_share_within_episode": overlap / left_count if left_count else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_monthly_event_windows(panel: pd.DataFrame, out_dir: Path, *, pre: int = 3, post: int = 6) -> None:
    if panel.empty:
        return
    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["month"], errors="coerce")
    value_cols = [
        "month",
        "raw_treasury_supply_usd_billions",
        "liquid_treasury_supply_usd_billions",
        "duration_gap_supply_usd_billions",
        "duration_gap_share",
        "raw_bill_share",
        "raw_coupon_share",
        "raw_weighted_maturity_years",
        "liquid_weighted_maturity_years",
        "bill_supply_shock_z",
        "d_deposits_usd_billions",
        "d_domestic_deposits_usd_billions",
        "d_total_mmf_assets_usd_billions",
        "d_mmf_treasury_holdings_usd_billions",
        "d_mmf_repo_holdings_usd_billions",
        "d_mmf_on_rrp_exposure_usd_billions",
        "d_on_rrp_usd_billions",
        "d_reserves_usd_billions",
        "d_tga_usd_billions",
        "on_rrp_regime_fixed",
        "qe_qt_regime",
        "high_rate_regime",
    ]
    value_cols = [c for c in value_cols if c in panel.columns]

    windows = []
    catalog_rows = []
    for event_type, flag_col in MONTHLY_EVENT_FLAGS:
        starts = _episode_start_dates(panel, date_col="month", flag_col=flag_col, min_gap=post, quarterly=False)
        catalog_rows.append(
            {
                "episode_type": event_type,
                "flag_column": flag_col,
                "event_count": len(starts),
                "first_event": starts[0].date().isoformat() if starts else "",
                "last_event": starts[-1].date().isoformat() if starts else "",
                "window": f"-{pre}m..+{post}m",
                "event_start_rule": f"flagged month with no prior selected same-flag event within {post} months",
            }
        )
        for start in starts:
            windows.append(
                _window_frame(
                    panel[value_cols],
                    date_col="month",
                    event_type=event_type,
                    event_date=start,
                    pre=pre,
                    post=post,
                )
            )

    pd.DataFrame(catalog_rows).to_csv(out_dir / "monthly_episode_event_catalog.csv", index=False)
    if not windows:
        return
    window = pd.concat(windows, ignore_index=True).sort_values(["episode_type", "event_date", "relative_period"])
    change_cols = [c for c in window.columns if c.startswith("d_") and c.endswith("_usd_billions")]
    for col in change_cols:
        post_mask = window["relative_period"].ge(0)
        window[f"cumulative_post_{col}"] = np.nan
        window.loc[post_mask, f"cumulative_post_{col}"] = window.loc[post_mask].groupby("event_id")[col].cumsum()
    window["event_window_note"] = "Descriptive event window around mechanically selected episode starts; not a causal design."
    window.to_csv(out_dir / "monthly_episode_event_windows.csv", index=False)
    _event_window_summary(window, date_col="month").to_csv(out_dir / "monthly_episode_event_window_summary.csv", index=False)


def write_quarterly_event_windows(panel: pd.DataFrame, out_dir: Path, *, pre: int = 4, post: int = 4) -> None:
    if panel.empty:
        return
    panel = panel.copy()
    panel["quarter"] = pd.to_datetime(panel["quarter"], errors="coerce")
    value_cols = [
        "quarter",
        f"{TDC_ANCHOR_COL}_usd_billions",
        "raw_treasury_supply_q_usd_billions",
        "liquid_treasury_supply_q_usd_billions",
        "duration_gap_supply_q_usd_billions",
        "duration_gap_share_q",
        "bill_share_raw_q",
        "coupon_share_raw_q",
        "raw_weighted_maturity_years_q",
        "liquid_weighted_maturity_years_q",
        "d_deposits_q_usd_billions",
        "d_domestic_deposits_q_usd_billions",
        "d_total_mmf_assets_q_usd_billions",
        "d_mmf_treasury_holdings_q_usd_billions",
        "d_on_rrp_q_usd_billions",
        "d_reserves_q_usd_billions",
        "d_tga_q_usd_billions",
        "reserve_drain_pressure",
    ]
    value_cols = [c for c in value_cols if c in panel.columns]

    windows = []
    catalog_rows = []
    for event_type, flag_col in QUARTERLY_EVENT_FLAGS:
        starts = _episode_start_dates(panel, date_col="quarter", flag_col=flag_col, min_gap=post, quarterly=True)
        catalog_rows.append(
            {
                "episode_type": event_type,
                "flag_column": flag_col,
                "event_count": len(starts),
                "first_event": starts[0].date().isoformat() if starts else "",
                "last_event": starts[-1].date().isoformat() if starts else "",
                "window": f"-{pre}q..+{post}q",
                "event_start_rule": f"flagged quarter with no prior selected same-flag event within {post} quarters",
            }
        )
        for start in starts:
            windows.append(
                _window_frame(
                    panel[value_cols],
                    date_col="quarter",
                    event_type=event_type,
                    event_date=start,
                    pre=pre,
                    post=post,
                    quarterly=True,
                )
            )

    pd.DataFrame(catalog_rows).to_csv(out_dir / "quarterly_tier2_episode_event_catalog.csv", index=False)
    if not windows:
        return
    window = pd.concat(windows, ignore_index=True).sort_values(["episode_type", "event_date", "relative_period"])
    change_cols = [c for c in window.columns if c.startswith("d_") and c.endswith("_usd_billions")]
    for col in change_cols:
        post_mask = window["relative_period"].ge(0)
        window[f"cumulative_post_{col}"] = np.nan
        window.loc[post_mask, f"cumulative_post_{col}"] = window.loc[post_mask].groupby("event_id")[col].cumsum()
    window["event_window_note"] = "Descriptive quarterly event window around mechanically selected episode starts; not a causal design."
    window.to_csv(out_dir / "quarterly_tier2_episode_event_windows.csv", index=False)
    _event_window_summary(window, date_col="quarter").to_csv(out_dir / "quarterly_tier2_episode_event_window_summary.csv", index=False)


def write_monthly_episode_candidates(monthly: pd.DataFrame, out_dir: Path) -> None:
    if monthly.empty:
        return
    panel = monthly[
        monthly["weight_family"].eq("fixed_baseline")
        & monthly.get("supply_basis", pd.Series("issuance_flow", index=monthly.index)).eq("issuance_flow")
    ].copy()
    if panel.empty:
        return

    if "is_future_or_scheduled" in panel.columns:
        panel = panel[~_boolish(panel, "is_future_or_scheduled")].copy()
    panel = panel.sort_values("month").reset_index(drop=True)
    panel = _add_unit_scaled_columns(panel)
    panel["d_raw_weighted_maturity_years"] = _numeric(panel, "raw_weighted_maturity_years").diff()
    panel["duration_gap_share"] = _numeric(panel, "duration_gap_supply") / _numeric(panel, "raw_treasury_supply")

    bill_share = _numeric(panel, "raw_bill_share")
    panel["bill_heavy_episode"] = (_flag_top(panel, "raw_bill_share") | _boolish(panel, "large_positive_bill_shock")).astype(bool)
    panel["coupon_heavy_episode"] = (
        _flag_top(panel, "raw_coupon_share") & bill_share.le(bill_share.dropna().median())
    ).fillna(False)
    panel["wam_extension_episode"] = (
        (_flag_top(panel, "d_raw_weighted_maturity_years") & _numeric(panel, "d_raw_weighted_maturity_years").gt(0))
        | _flag_top(panel, "raw_weighted_maturity_years")
    ).fillna(False)
    panel["duration_gap_heavy_episode"] = _flag_top(panel, "duration_gap_share")
    panel["reserve_drain_episode"] = _flag_bottom_negative(panel, "d_reserves")
    panel["mmf_absorption_episode"] = (
        (_flag_top(panel, "d_mmf_treasury_holdings") & _numeric(panel, "d_mmf_treasury_holdings").gt(0))
        | (_flag_top(panel, "d_total_mmf_assets") & _numeric(panel, "d_total_mmf_assets").gt(0))
    ).fillna(False)
    panel["on_rrp_drain_episode"] = _flag_bottom_negative(panel, "d_on_rrp")
    panel["tga_refill_episode"] = (_flag_top(panel, "d_tga") & _numeric(panel, "d_tga").gt(0)).fillna(False)
    panel["high_rate_episode"] = panel.get("high_rate_regime", pd.Series("", index=panel.index)).astype("string").eq("high_rate")
    panel["qt_episode"] = panel.get("qe_qt_regime", pd.Series("", index=panel.index)).astype("string").eq("qt")
    panel["episode_flag_count"] = panel[
        [
            "bill_heavy_episode",
            "coupon_heavy_episode",
            "wam_extension_episode",
            "duration_gap_heavy_episode",
            "reserve_drain_episode",
            "mmf_absorption_episode",
            "on_rrp_drain_episode",
            "tga_refill_episode",
            "high_rate_episode",
            "qt_episode",
        ]
    ].sum(axis=1)
    panel["episode_design_note"] = (
        "Mechanical first-pass classifier for event-study design; thresholds are descriptive and pre-identification."
    )

    cols = [
        "month",
        "raw_treasury_supply",
        "raw_treasury_supply_usd_billions",
        "liquid_treasury_supply",
        "liquid_treasury_supply_usd_billions",
        "duration_gap_supply",
        "duration_gap_supply_usd_billions",
        "duration_gap_share",
        "raw_bill_share",
        "raw_coupon_share",
        "raw_weighted_maturity_years",
        "liquid_weighted_maturity_years",
        "d_raw_weighted_maturity_years",
        "bill_supply_shock_z",
        "large_positive_bill_shock",
        "isolated_large_positive_bill_shock",
        "bill_heavy_episode",
        "coupon_heavy_episode",
        "wam_extension_episode",
        "duration_gap_heavy_episode",
        "reserve_drain_episode",
        "mmf_absorption_episode",
        "on_rrp_drain_episode",
        "tga_refill_episode",
        "high_rate_episode",
        "qt_episode",
        "episode_flag_count",
        "d_deposits",
        "d_deposits_usd_billions",
        "d_domestic_deposits",
        "d_domestic_deposits_usd_billions",
        "d_total_mmf_assets",
        "d_total_mmf_assets_usd_billions",
        "d_mmf_treasury_holdings",
        "d_mmf_treasury_holdings_usd_billions",
        "d_mmf_repo_holdings",
        "d_mmf_repo_holdings_usd_billions",
        "d_mmf_on_rrp_exposure",
        "d_mmf_on_rrp_exposure_usd_billions",
        "d_on_rrp",
        "d_on_rrp_usd_billions",
        "d_reserves",
        "d_reserves_usd_billions",
        "d_tga",
        "d_tga_usd_billions",
        "on_rrp_regime_fixed",
        "qe_qt_regime",
        "high_rate_regime",
        "bill_iorb_spread",
        "sofr_iorb_spread",
        "episode_design_note",
    ]
    cols = [c for c in cols if c in panel.columns]
    out = panel[cols]
    out.to_csv(out_dir / "monthly_episode_candidates.csv", index=False)
    out.to_csv(out_dir / "monthly_episode_design.csv", index=False)
    _write_episode_overlap(out, flag_pairs=MONTHLY_EVENT_FLAGS, out_path=out_dir / "monthly_episode_overlap_matrix.csv")
    write_monthly_event_windows(out, out_dir)


def write_quarterly_episode_design(quarterly: pd.DataFrame, out_dir: Path) -> None:
    quarterly = eligible_tdc_rows(quarterly)
    if quarterly.empty:
        return
    basis = quarterly.get("supply_basis", pd.Series("issuance_flow", index=quarterly.index))
    panel = quarterly[
        quarterly["weight_family"].eq("fixed_baseline")
        & basis.eq("issuance_flow")
    ].copy()
    if panel.empty:
        return
    panel = panel.sort_values("quarter").reset_index(drop=True)
    panel = _add_unit_scaled_columns(panel)
    panel["d_raw_weighted_maturity_years_q"] = _numeric(panel, "raw_weighted_maturity_years_q").diff()
    panel["duration_gap_share_q"] = _numeric(panel, "duration_gap_supply_q") / _numeric(panel, "raw_treasury_supply_q")

    panel["bill_heavy_episode_q"] = _flag_top(panel, "bill_share_raw_q")
    panel["coupon_heavy_episode_q"] = _flag_top(panel, "coupon_share_raw_q") & _numeric(panel, "bill_share_raw_q").le(
        _numeric(panel, "bill_share_raw_q").dropna().median()
    )
    panel["wam_extension_episode_q"] = (
        (_flag_top(panel, "d_raw_weighted_maturity_years_q") & _numeric(panel, "d_raw_weighted_maturity_years_q").gt(0))
        | _flag_top(panel, "raw_weighted_maturity_years_q")
    ).fillna(False)
    panel["duration_gap_heavy_episode_q"] = _flag_top(panel, "duration_gap_share_q")
    panel["reserve_drain_episode_q"] = _flag_bottom_negative(panel, "d_reserves_q")
    panel["mmf_absorption_episode_q"] = (
        (_flag_top(panel, "d_mmf_treasury_holdings_q") & _numeric(panel, "d_mmf_treasury_holdings_q").gt(0))
        | (_flag_top(panel, "d_total_mmf_assets_q") & _numeric(panel, "d_total_mmf_assets_q").gt(0))
    ).fillna(False)
    panel["on_rrp_drain_episode_q"] = _flag_bottom_negative(panel, "d_on_rrp_q")
    panel["tga_refill_episode_q"] = (_flag_top(panel, "d_tga_q") & _numeric(panel, "d_tga_q").gt(0)).fillna(False)
    panel["episode_design_note"] = (
        "Quarterly Tier 2 overlay for episode design; descriptive classifier, not identification."
    )

    cols = [
        "quarter",
        TDC_ANCHOR_COL,
        f"{TDC_ANCHOR_COL}_usd_billions",
        "raw_treasury_supply_q",
        "raw_treasury_supply_q_usd_billions",
        "liquid_treasury_supply_q",
        "liquid_treasury_supply_q_usd_billions",
        "duration_gap_supply_q",
        "duration_gap_supply_q_usd_billions",
        "duration_gap_share_q",
        "bill_share_raw_q",
        "coupon_share_raw_q",
        "raw_weighted_maturity_years_q",
        "liquid_weighted_maturity_years_q",
        "d_raw_weighted_maturity_years_q",
        "d_deposits_q",
        "d_deposits_q_usd_billions",
        "d_domestic_deposits_q",
        "d_domestic_deposits_q_usd_billions",
        "d_total_mmf_assets_q",
        "d_total_mmf_assets_q_usd_billions",
        "d_mmf_treasury_holdings_q",
        "d_mmf_treasury_holdings_q_usd_billions",
        "d_on_rrp_q",
        "d_on_rrp_q_usd_billions",
        "d_reserves_q",
        "d_reserves_q_usd_billions",
        "d_tga_q",
        "d_tga_q_usd_billions",
        "reserve_drain_pressure",
        "bill_heavy_episode_q",
        "coupon_heavy_episode_q",
        "wam_extension_episode_q",
        "duration_gap_heavy_episode_q",
        "reserve_drain_episode_q",
        "mmf_absorption_episode_q",
        "on_rrp_drain_episode_q",
        "tga_refill_episode_q",
        "months_observed",
        "is_complete_quarter",
        "is_tdc_aligned",
        "episode_design_note",
    ]
    cols = [c for c in cols if c in panel.columns]
    out = panel[cols]
    out.to_csv(out_dir / "quarterly_tier2_episode_design.csv", index=False)
    out.to_csv(out_dir / "quarterly_tier2_episode_candidates.csv", index=False)
    _write_episode_overlap(out, flag_pairs=QUARTERLY_EVENT_FLAGS, out_path=out_dir / "quarterly_tier2_episode_overlap_matrix.csv")
    write_quarterly_event_windows(out, out_dir)


TDC_DERIVED_FILES = (
    "quarterly_tdc_comparison_correlations.csv",
    "ladder_interpretation_summary.csv",
    "core_measurement_summary.csv",
    "exploratory_sidecar_summary.csv",
    "latest_results_overview.csv",
    "latest_results_overview.md",
    "tdc_model_comparison_diagnostics.csv",
    "quarterly_tier2_episode_design.csv",
    "quarterly_tier2_episode_candidates.csv",
    "quarterly_tier2_episode_event_catalog.csv",
    "quarterly_tier2_episode_event_windows.csv",
    "quarterly_tier2_episode_event_window_summary.csv",
    "quarterly_tier2_episode_overlap_matrix.csv",
)


def write_output_tier_manifest(out_dir: Path, *, has_eligible_anchor: bool) -> None:
    rows = [
        {
            "path": "output/tables/core_measurement_summary.csv",
            "tier": "headline",
            "claim_status": "promote",
            "role": "Core measurement table: gross issuance-flow basis, fixed baseline plus fixed sensitivity band, Tier 2 anchor only.",
            "guardrail": "Descriptive composition measure; not a TDC estimator.",
        },
        {
            "path": "output/tables/latest_results_overview.md",
            "tier": "headline",
            "claim_status": "promote",
            "role": "Plain-language current result overview.",
            "guardrail": "Use with fixed-family headline scope for issuance flow.",
        },
        {
            "path": "output/tables/ladder_interpretation_summary.csv",
            "tier": "diagnostic",
            "claim_status": "use_with_care",
            "role": "Broad family and basis summary for interpretation and appendix checks.",
            "guardrail": "Do not pool all families as equal evidence in public prose.",
        },
        {
            "path": "output/tables/exploratory_sidecar_summary.csv",
            "tier": "sidecar",
            "claim_status": "appendix_only",
            "role": "Price, turnover, auction-buyer, and Z.1 holder-context variants.",
            "guardrail": "Diagnostic assumptions only; not structural moneyness or final-holder evidence.",
        },
        {
            "path": "output/tables/monthly_episode_design.csv",
            "tier": "episode_design",
            "claim_status": "research_design",
            "role": "Monthly fixed-baseline issuance-flow episode flags and absorption outcomes.",
            "guardrail": "Mechanical classifier; inspect overlaps before promotion.",
        },
        {
            "path": "output/tables/monthly_episode_event_window_summary.csv",
            "tier": "episode_design",
            "claim_status": "research_design",
            "role": "Average -3 to +6 month descriptive paths around monthly episode starts.",
            "guardrail": "Descriptive event windows, not causal event studies.",
        },
        {
            "path": "output/tables/quarterly_tier2_episode_design.csv",
            "tier": "episode_design",
            "claim_status": "research_design",
            "role": "Quarterly Tier 2 overlay episode flags.",
            "guardrail": "Low-frequency validation surface; not identification.",
        },
        {
            "path": "output/tables/quarterly_tier2_episode_event_window_summary.csv",
            "tier": "episode_design",
            "claim_status": "research_design",
            "role": "Average -4 to +4 quarter descriptive paths around quarterly episode starts.",
            "guardrail": "Descriptive event windows, not causal event studies.",
        },
        {
            "path": "output/tables/tdc_model_comparison_diagnostics.csv",
            "tier": "appendix",
            "claim_status": "appendix_only",
            "role": "Standardized descriptive model diagnostics.",
            "guardrail": "No causal interpretation; no headline R-squared claims.",
        },
        {
            "path": "output/qa/family_support_matrix.csv",
            "tier": "qa",
            "claim_status": "required_context",
            "role": "Row- and dollar-weighted support by family, basis, and sample window.",
            "guardrail": "Use before citing sidecar families.",
        },
        {
            "path": "output/qa/analysis_unit_assumptions.csv",
            "tier": "qa",
            "claim_status": "required_context",
            "role": "Unit-scale rules for Treasury dollars versus TDC/macro USD millions.",
            "guardrail": "Use scaled columns when comparing magnitudes.",
        },
    ]
    manifest = pd.DataFrame(rows)
    manifest["exists"] = manifest["path"].map(lambda rel: (out_dir.parent.parent / rel).exists())
    manifest.loc[~manifest["exists"], "claim_status"] = "unavailable"
    if not has_eligible_anchor:
        dependent = manifest["path"].map(lambda rel: Path(rel).name in TDC_DERIVED_FILES)
        manifest.loc[dependent, "claim_status"] = "blocked_missing_primary_anchor"
    manifest.to_csv(out_dir / "output_tiers.csv", index=False)


def write_release_readiness_report(out_dir: Path, *, has_eligible_anchor: bool) -> None:
    """Describe current-run artifacts; never grant a public release decision."""
    evidence = {
        "core_claim": ("core_measurement_summary.csv", "latest_results_overview.md"),
        "tdc_anchor": ("ladder_interpretation_summary.csv",),
        "episodes": ("quarterly_tier2_episode_design.csv",),
        "model_diagnostics": ("tdc_model_comparison_diagnostics.csv",),
    }
    rows = []
    for area, paths in evidence.items():
        available = all((out_dir / name).exists() for name in paths)
        status = "available_for_review" if available else "unavailable"
        if not has_eligible_anchor:
            status = "blocked_missing_primary_anchor"
        rows.append({"area": area, "status": status, "evidence": ";".join(paths)})
    pd.DataFrame(rows).to_csv(out_dir / "release_readiness_status.csv", index=False)
    lines = [
        "# tdcladder Analysis Readiness",
        "",
        "Current-run artifacts only; this report does not approve publication or empirical claims.",
        "TDC comparisons require a complete quarter with an aligned, observed primary anchor.",
        "Liquidity weights are measurement assumptions; comparisons remain descriptive.",
        "",
        *[f"- {row['area']}: {row['status']}" for row in rows],
    ]
    overview = out_dir / "latest_results_overview.md"
    if has_eligible_anchor and overview.exists():
        lines.extend(["", "## Latest Results Snapshot", "", overview.read_text(encoding="utf-8").strip()])
    (out_dir / "release_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(".")
    monthly_path = root / "data/clean/monthly_ladder_liqsub_panel.csv"
    quarterly_path = root / "data/clean/quarterly_tdc_comparison_panel.csv"
    if not monthly_path.exists():
        raise FileNotFoundError("Run `tdcladder build` before starter analysis.")

    monthly = pd.read_csv(monthly_path, parse_dates=["month"])
    monthly = _add_unit_scaled_columns(monthly)
    numeric_candidates = [
        "raw_treasury_supply",
        "liquid_treasury_supply",
        "duration_gap_supply",
        "deposits",
        "domestic_deposits",
        "total_mmf_assets",
        "mmf_treasury_holdings",
        "on_rrp",
        "reserves",
    ]
    keep = [c for c in numeric_candidates if c in monthly.columns]
    group_cols = ["weight_family"]
    if "supply_basis" in monthly.columns:
        group_cols.append("supply_basis")
    corr = monthly.groupby(group_cols)[keep].corr().reset_index()
    out_dir = root / "output/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove only the known generated TDC surfaces before evaluating this run.
    for name in (*TDC_DERIVED_FILES, "output_tiers.csv", "release_readiness_status.csv", "release_readiness_report.md"):
        (out_dir / name).unlink(missing_ok=True)
    corr.to_csv(out_dir / "monthly_liquidity_measure_correlations.csv", index=False)
    write_monthly_episode_candidates(monthly, out_dir)

    quarterly = pd.read_csv(quarterly_path, parse_dates=["quarter"]) if quarterly_path.exists() else pd.DataFrame()
    quarterly = eligible_tdc_rows(quarterly)
    if quarterly.empty:
        write_output_tier_manifest(out_dir, has_eligible_anchor=False)
        write_release_readiness_report(out_dir, has_eligible_anchor=False)
        return 1
    else:
        quarterly = _add_unit_scaled_columns(quarterly)
        stock = quarterly["supply_basis"].eq("outstanding_stock") if "supply_basis" in quarterly.columns else pd.Series(False, index=quarterly.index)
        quarterly["raw_supply_tdc_aligned"] = pd.to_numeric(quarterly["raw_treasury_supply_q"], errors="coerce")
        quarterly["liquid_supply_tdc_aligned"] = pd.to_numeric(quarterly["liquid_treasury_supply_q"], errors="coerce")
        if "d_raw_treasury_supply_q" in quarterly.columns:
            quarterly.loc[stock, "raw_supply_tdc_aligned"] = pd.to_numeric(quarterly.loc[stock, "d_raw_treasury_supply_q"], errors="coerce")
        if "d_liquid_treasury_supply_q" in quarterly.columns:
            quarterly.loc[stock, "liquid_supply_tdc_aligned"] = pd.to_numeric(quarterly.loc[stock, "d_liquid_treasury_supply_q"], errors="coerce")
        qcols = [c for c in [
            "raw_supply_tdc_aligned",
            "liquid_supply_tdc_aligned",
            TDC_ANCHOR_COL,
            "tdc_bank_only_qoq",
            "total_deposits_bank_qoq",
            "bill_share_raw_q",
            "raw_weighted_maturity_years_q",
        ] if c in quarterly.columns]
        qgroup_cols = ["weight_family"]
        if "supply_basis" in quarterly.columns:
            qgroup_cols.append("supply_basis")
        quarterly.groupby(qgroup_cols)[qcols].corr().reset_index().to_csv(
            out_dir / "quarterly_tdc_comparison_correlations.csv", index=False
        )
        write_ladder_interpretation_table(quarterly, out_dir)
        write_tdc_model_comparison_diagnostics(quarterly, out_dir)
        write_quarterly_episode_design(quarterly, out_dir)
        write_output_tier_manifest(out_dir, has_eligible_anchor=True)
        write_release_readiness_report(out_dir, has_eligible_anchor=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Wrote correlation tables. Install the analysis extra for figures: pip install -e '.[analysis]'")
        return 0

    fig_dir = root / "output/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    baseline = monthly[monthly["weight_family"].eq("fixed_baseline")].sort_values("month")
    if "supply_basis" not in baseline.columns:
        baseline["supply_basis"] = "issuance_flow"
    for basis, plot_frame in baseline.groupby("supply_basis", dropna=False):
        if plot_frame.empty:
            continue
        ax = plot_frame.plot(x="month", y=["raw_treasury_supply", "liquid_treasury_supply"])
        ax.set_title(f"Raw vs liquidity-weighted Treasury supply ({basis}, fixed baseline)")
        ax.set_ylabel("Supply amount, USD")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(fig_dir / f"raw_vs_liquid_supply_fixed_baseline_{basis}.png", dpi=150)
        plt.close(fig)
    print("Wrote starter analysis tables and figures under output/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
