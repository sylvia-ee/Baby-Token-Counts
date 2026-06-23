import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COUNTS_PATH = PROJECT_ROOT / "data" / "counts_by_age.csv"

# (canonical label, raw count column, log-transformed column) — the canonical
# label is used for color/sort identity so it stays stable when the log toggle flips
METRIC_DEFS = [
    ("Total alt inclusive count", "total_alt_incl_count", "total_alt_incl_logcount"),
    ("Total alt exclusive count", "total_alt_excl_count", "total_alt_excl_logcount"),
    ("Base inclusive count", "base_incl_count", "base_incl_logcount"),
    ("Base exclusive count", "base_excl_count", "base_excl_logcount"),
]
MAX_SLOTS = len(METRIC_DEFS)
METRIC_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

# sort-only fields (not plotted as bars) — same (canonical label, raw col, log col) shape
DIFF_DEFS = [
    ("Alt diff", "alt_diff", "alt_logdiff"),
    ("Base diff", "base_diff", "base_logdiff"),
    ("Alt-base exclusive diff", "alt_base_excl_diff", "alt_base_excl_logdiff"),
    ("Alt-base inclusive diff", "alt_base_incl_diff", "alt_base_incl_logdiff"),
]
SORTABLE_DEFS = METRIC_DEFS + DIFF_DEFS


@st.cache_data
def load_counts():
    return pd.read_csv(COUNTS_PATH)


def standardized_beta(x, y):
    """Correlations for log frequency vs. production (MCDI). Betas have been standardized,
    so interpret betas as comparable. Note that betas reflect cumulative input up to that age."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_std, y_std = x.std(), y.std()
    if x_std == 0 or y_std == 0:
        return float("nan")
    beta, _intercept = np.polyfit((x - x.mean()) / x_std, (y - y.mean()) / y_std, 1)
    return beta


def numeric_range_filter(df, col, label, default_min=None):
    if df.empty:
        st.caption(f"{label}: no rows to filter")
        return df

    lo, hi = df[col].min(), df[col].max()
    if lo == hi:
        st.caption(f"{label}: every row is {lo:g} (no range to filter)")
        return df

    cast = int if pd.api.types.is_integer_dtype(df[col]) else float
    lo, hi = cast(lo), cast(hi)
    default_min_value = lo if default_min is None else max(lo, min(cast(default_min), hi))

    st.caption(label)
    min_col, max_col = st.columns(2)
    with min_col:
        selected_min = st.number_input(
            "min", min_value=lo, max_value=hi, value=default_min_value, key=f"{col}_min"
        )
    with max_col:
        selected_max = st.number_input("max", min_value=lo, max_value=hi, value=hi, key=f"{col}_max")

    if selected_min > selected_max:
        st.error(f"{label}: min cannot exceed max")
        selected_min, selected_max = selected_max, selected_min

    return df[df[col].between(selected_min, selected_max)]


def _set_all_checkboxes(key_prefix, options, value):
    for opt in options:
        st.session_state[f"{key_prefix}_{opt}"] = value


def checkbox_list_filter(df, col, label, container, n_cols=2, searchable=False):
    options = sorted(df[col].dropna().unique().tolist())
    key_prefix = f"chk_{col}"

    container.caption(label)
    select_col, clear_col = container.columns(2)
    select_col.button("Select all", key=f"{key_prefix}_select_all", on_click=_set_all_checkboxes,
                       args=(key_prefix, options, True))
    clear_col.button("Clear all", key=f"{key_prefix}_clear_all", on_click=_set_all_checkboxes,
                      args=(key_prefix, options, False))

    # every option gets a stored state up front, so search can hide checkboxes
    # without losing their selection
    for opt in options:
        st.session_state.setdefault(f"{key_prefix}_{opt}", True)

    display_options = options
    if searchable:
        query = container.text_input(f"Search {label.lower()}", key=f"{key_prefix}_search")
        if query:
            display_options = [opt for opt in options if query.lower() in str(opt).lower()]

    cols = container.columns(n_cols)
    for i, opt in enumerate(display_options):
        with cols[i % n_cols]:
            st.checkbox(str(opt), key=f"{key_prefix}_{opt}")

    selected = [opt for opt in options if st.session_state.get(f"{key_prefix}_{opt}", True)]
    return df[df[col].isin(selected)] if selected else df.iloc[0:0]


def page_beta_trend():
    st.header("Standardized β by age cutoff")
    st.caption(
       "Correlations for log frequency vs. production (MCDI). Betas have been standardized,"
    "so interpret betas as comparable. Note that betas reflect cumulative input up to that age"
    )

    beta_trend_rows = []
    for age in available_ages:
        age_slice = counts_by_age_df[counts_by_age_df["age at count"] == age]
        age_base_df = (
            age_slice[["mcdi", "avg_production"] + [log_col for _, _, log_col in METRIC_DEFS]]
            .drop_duplicates(subset="mcdi")
        )
        for name, _, log_col in METRIC_DEFS:
            beta = standardized_beta(age_base_df[log_col], age_base_df["avg_production"])
            beta_trend_rows.append({"age": age, "metric": name, "beta": beta})
    beta_trend_df = pd.DataFrame(beta_trend_rows)

    beta_trend_chart = (
        alt.Chart(beta_trend_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("age:Q", title="Age cutoff (months)"),
            y=alt.Y("beta:Q", title="Standardized β"),
            color=alt.Color(
                "metric:N",
                title="Count type",
                scale=alt.Scale(domain=[name for name, _, _ in METRIC_DEFS], range=METRIC_COLORS),
            ),
            tooltip=["age", "metric", "beta"],
        )
        .properties(width=800, height=350)
    )
    st.altair_chart(beta_trend_chart, width="content")


def page_scatter():
    st.header("Avg. production vs. frequency by counting method")
    st.caption("See settings on sidebar, this is age segregated. So only shows one age at a time.")

    if counts_df.empty:
        st.warning("No data available.")
        return

    scatter_base_df = (
        counts_df[["mcdi", "avg_production"] + [log_col for _, _, log_col in METRIC_DEFS]]
        .drop_duplicates(subset="mcdi")
    )

    toggle_cols = st.columns(len(METRIC_DEFS))
    visible_scatter_metrics = []
    for (name, _, log_col), toggle_col in zip(METRIC_DEFS, toggle_cols):
        with toggle_col:
            if st.checkbox(name, value=True, key=f"scatter_show_{log_col}"):
                visible_scatter_metrics.append((name, log_col))

    if not visible_scatter_metrics:
        st.info("Toggle on at least one count type to plot.")
        return

    scatter_rows = []
    betas = []
    for name, log_col in visible_scatter_metrics:
        sub = scatter_base_df[["mcdi", "avg_production", log_col]].rename(
            columns={log_col: "frequency"}
        )
        sub["metric"] = name
        scatter_rows.append(sub)

        beta = standardized_beta(sub["frequency"], sub["avg_production"])
        betas.append((name, beta))

    scatter_df = pd.concat(scatter_rows, ignore_index=True)
    scatter_color_scale = alt.Scale(
        domain=[name for name, _, _ in METRIC_DEFS], range=METRIC_COLORS
    )

    points = (
        alt.Chart(scatter_df)
        .mark_circle(size=45, opacity=0.6)
        .encode(
            x=alt.X("frequency:Q", title="Log frequency"),
            y=alt.Y("avg_production:Q", title="Avg. production"),
            color=alt.Color("metric:N", title="Count type", scale=scatter_color_scale),
            tooltip=["mcdi", "metric", "frequency", "avg_production"],
        )
    )
    # drop the inherited per-point tooltip on the line layer so it never
    # shadows a point's mcdi-word tooltip where the two overlap
    regression_lines = (
        points.transform_regression("frequency", "avg_production", groupby=["metric"])
        .mark_line()
        .encode(tooltip=alt.value(None))
    )

    beta_col, chart_col = st.columns([1, 4])
    with beta_col:
        st.caption("**Standardized β**")
        st.caption("  \n".join(f"{name}: β = {beta:.6g}" for name, beta in betas))
    with chart_col:
        st.altair_chart(
            (points + regression_lines).properties(width=700, height=450).interactive(),
            width="content",
        )


def page_bar_plot():
    st.header("Summary statistics by MCDI word")
    st.caption("See settings on sidebar, this is age segregated. So only shows one age at a time. Also note" \
    "           that if you choose to apply log counts, that sorting by any diff will be the log diff.")

    st.sidebar.subheader("Bar plot filters")
    filtered_df = counts_df.copy()
    filtered_df = checkbox_list_filter(filtered_df, "category", "Category", st.sidebar, n_cols=2)
    filtered_df = checkbox_list_filter(filtered_df, "grammar", "Grammar", st.sidebar, n_cols=1)

    st.sidebar.caption("Count ranges")
    for col, label, default_min in [
        ("total_alt_incl_count", "Total alt inclusive count", 1),
        ("total_alt_excl_count", "Total alt exclusive count", 1),
        ("base_incl_count", "Base inclusive count", 1),
        ("base_excl_count", "Base exclusive count", 1),
        ("avg_production", "Avg. production", None),
    ]:
        with st.sidebar:
            filtered_df = numeric_range_filter(filtered_df, col, label, default_min=default_min)

    with st.expander(f"Filtered table ({len(filtered_df)} rows)"):
        st.dataframe(filtered_df, width="stretch")

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    use_log = st.checkbox("Use log-transformed counts", value=False)

    active_col_for = {name: (log_col if use_log else raw_col) for name, raw_col, log_col in SORTABLE_DEFS}
    canonical_for_col = {}
    for name, raw_col, log_col in SORTABLE_DEFS:
        canonical_for_col[raw_col] = name
        canonical_for_col[log_col] = name
    metric_names = [name for name, _, _ in METRIC_DEFS]
    active_cols = [active_col_for[name] for name in metric_names]
    sortable_cols = [active_col_for[name] for name, _, _ in SORTABLE_DEFS]

    # the active count/diff columns are already mcdi-level constants computed in
    # childes_preprocess.py, so just dedupe to one row per mcdi
    summary_df = filtered_df[["mcdi"] + sortable_cols].drop_duplicates(subset="mcdi").reset_index(drop=True)

    sort_field_map = {"MCDI word (A-Z)": None}
    for name, _, _ in SORTABLE_DEFS:
        sort_field_map[name] = active_col_for[name]

    sort_options = list(sort_field_map.keys())
    default_sort_label = "Total alt exclusive count"
    sort_col1, sort_col2, sort_col3 = st.columns([2, 1, 1])
    with sort_col1:
        sort_label = st.selectbox(
            "Sort groups by", sort_options, index=sort_options.index(default_sort_label)
        )
    with sort_col2:
        descending = st.checkbox(
            "Descending", value=sort_label != "MCDI word (A-Z)", key="sort_descending"
        )
    with sort_col3:
        words_per_panel = st.number_input("Words per panel", min_value=5, max_value=100, value=20, step=5)

    sort_field = sort_field_map[sort_label]
    if sort_field is None:
        mcdi_order = sorted(summary_df["mcdi"].tolist(), reverse=descending)
    else:
        mcdi_order = summary_df.sort_values(sort_field, ascending=not descending)["mcdi"].tolist()

    # collapse identical values into a single bar per mcdi word, ranked highest to lowest.
    # "metric" is the canonical (toggle-independent) name of the highest-priority count
    # type sharing that value, so color stays tied to a specific count type even when
    # collapsed or when switching between raw and log-transformed counts
    bar_rows = []
    for _, row in summary_df.iterrows():
        value_to_metrics = {}
        for col in active_cols:
            value_to_metrics.setdefault(row[col], []).append(canonical_for_col[col])
        ranked = sorted(value_to_metrics.items(), key=lambda kv: kv[0], reverse=True)
        for slot, (value, metrics) in enumerate(ranked, start=1):
            bar_rows.append({
                "mcdi": row["mcdi"],
                "slot": str(slot),
                "value": value,
                "metric": metrics[0],
                "metrics": ", ".join(metrics),
            })
    bars_df = pd.DataFrame(bar_rows)

    n_words = summary_df["mcdi"].nunique()
    y_max = bars_df["value"].max()
    n_panels = max(1, math.ceil(n_words / words_per_panel))
    y_title = "Log frequency" if use_log else "Frequency"

    for panel_idx in range(n_panels):
        panel_words = mcdi_order[panel_idx * words_per_panel:(panel_idx + 1) * words_per_panel]
        panel_bars_df = bars_df[bars_df["mcdi"].isin(panel_words)]

        chart = (
            alt.Chart(panel_bars_df)
            .mark_bar()
            .encode(
                x=alt.X("mcdi:N", title="MCDI word", sort=panel_words),
                xOffset=alt.XOffset(
                    "slot:N", scale=alt.Scale(domain=[str(i) for i in range(1, MAX_SLOTS + 1)])
                ),
                y=alt.Y("value:Q", title=y_title, scale=alt.Scale(domain=[0, y_max * 1.05])),
                color=alt.Color(
                    "metric:N",
                    title="Count type",
                    scale=alt.Scale(domain=metric_names, range=METRIC_COLORS),
                ),
                tooltip=["mcdi", "metrics", "value"],
            )
            .properties(width=max(700, 55 * len(panel_words)), height=450)
        )
        st.caption(
            f"Panel {panel_idx + 1} of {n_panels} — words "
            f"{panel_idx * words_per_panel + 1}-{panel_idx * words_per_panel + len(panel_words)} of {n_words}"
        )
        st.altair_chart(chart, width="content")


st.set_page_config(page_title="CHILDES-MCDI Counts", layout="wide")
st.title("MCDI-CHILDES Counts")

INTRO_CAPTION_LINES = [
    f"Source: {COUNTS_PATH}",
    "Each MCDI word here is displayed with four kinds of counts:",
    "• Total alt inclusive count: counts a word and all its possible forms, double-counting compound words",
    "• Total alt exclusive count: counts a word and all its possible forms, without double-counting compound words",
    "• Base inclusive count: counts only the canonical form of each word, double-counting compounds",
    "• Base exclusive count: counts only the canonical form of each word, without double-counting compounds",
    "The \"Use log-transformed counts\" toggle (on the bar plot page) switches all of the above to their log-transformed values.",
    "Four difference metrics are also available for sorting on the bar plot page:",
    "• Alt diff: total alt inclusive − total alt exclusive",
    "• Base diff: base inclusive − base exclusive",
    "• Alt-base exclusive diff: total alt exclusive − base exclusive",
    "• Alt-base inclusive diff: total alt inclusive − base inclusive",
    "Toggling the log-transform option switches these diffs to their log-difference versions too.",
]
st.caption("  \n".join(INTRO_CAPTION_LINES))

counts_by_age_df = load_counts()

available_ages = sorted(counts_by_age_df["age at count"].unique().tolist())
default_age = 24 if 24 in available_ages else available_ages[len(available_ages) // 2]

pg = st.navigation([
    st.Page(page_beta_trend, title="β by age", default=True),
    st.Page(page_scatter, title="Production vs. frequency"),
    st.Page(page_bar_plot, title="Summary bar plot"),
])

st.sidebar.header("Global filters")
st.sidebar.caption("Apply on every page.")
selected_age = st.sidebar.selectbox(
    "Age cutoff (months)", available_ages, index=available_ages.index(default_age)
)
counts_df = counts_by_age_df[counts_by_age_df["age at count"] == selected_age].drop(columns="age at count")

with st.sidebar.expander("MCDI word filter", expanded=False):
    counts_df = checkbox_list_filter(counts_df, "mcdi", "MCDI word", st, n_cols=2, searchable=True)

pg.run()
