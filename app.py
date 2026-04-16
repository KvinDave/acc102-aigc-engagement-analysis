import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Red Note AIGC Engagement Lab", page_icon="📈", layout="wide")


def parse_count(val):
    val = str(val).strip()
    if val.lower() in {"nan", "none", ""}:
        return np.nan
    if "万" in val:
        try:
            return float(val.replace("万", "")) * 10000
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def count_tags(x):
    if pd.isna(x):
        return 0
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return 0
    return len([t for t in s.split(",") if str(t).strip()])


@st.cache_data
def load_data():
    df = pd.read_csv("Posts.csv", encoding="gbk")
    df["liked_count_n"] = df["liked_count"].apply(parse_count)
    df["collected_count_n"] = df["collected_count"].apply(parse_count)
    df["comment_count_n"] = pd.to_numeric(df["comment_count"], errors="coerce")
    df["share_count_n"] = pd.to_numeric(df["share_count"], errors="coerce")
    df["engagement"] = df[["liked_count_n", "collected_count_n", "comment_count_n", "share_count_n"]].fillna(0).sum(axis=1)
    df["post_date"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    df["year"] = df["post_date"].dt.year
    df["month"] = df["post_date"].dt.month
    df["weekday"] = df["post_date"].dt.day_name()
    df["hour"] = df["post_date"].dt.hour
    df["tag_count"] = df["tag_list"].apply(count_tags)
    df["title_len"] = df["title"].fillna("").astype(str).str.len()
    df["desc_len"] = df["desc"].fillna("").astype(str).str.len()
    df["is_video"] = (df["type"] == "video").astype(int)
    df["log_engagement"] = np.log1p(df["engagement"])
    return df


def recommendation_text(df):
    if df.empty:
        return "No data available for the current filter selection."
    lines = []
    cat = df.groupby("feature")["engagement"].median().sort_values(ascending=False)
    top_cat = cat.index[0].replace("ai-", "").capitalize()
    lines.append(f"Focus on **{top_cat}**-style content first because it currently has the strongest typical engagement.")
    type_med = df.groupby("type")["engagement"].median()
    if {"video", "normal"}.issubset(set(type_med.index)) and type_med["normal"] > 0:
        lines.append(f"Where possible, convert ideas into **video posts** because they deliver about **{type_med['video']/type_med['normal']:.2f}×** the engagement of normal posts in the filtered sample.")
    tag_bins = pd.cut(df["tag_count"], bins=[-0.1,2,4,6,8,100], labels=["0–2","3–4","5–6","7–8","9+"])
    peak = df.groupby(tag_bins, observed=True)["engagement"].median().sort_values(ascending=False)
    if not peak.empty:
        lines.append(f"Aim for roughly **{peak.index[0]} tags**, which is the current sweet spot in this selection.")
    return "\n\n".join([f"- {x}" for x in lines])


df = load_data()

st.title("📈 Red Note AIGC Engagement Lab")
st.markdown("A distinction-level dashboard for exploring what drives engagement in Xiaohongshu AIGC posts.")

with st.sidebar:
    st.header("Filters")
    categories = sorted(df["feature"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Category",
        categories,
        default=categories,
        format_func=lambda x: x.replace("ai-", "").capitalize(),
    )
    selected_types = st.multiselect("Post type", ["normal", "video"], default=["normal", "video"])
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    selected_years = st.slider("Year range", year_min, year_max, (year_min, year_max))
    max_tags = int(df["tag_count"].max())
    selected_tag = st.slider("Tag count range", 0, max_tags, (0, max_tags))
    st.caption("Tip: narrow the sample to compare niches, years, or post formats.")

fdf = df[
    df["feature"].isin(selected_categories)
    & df["type"].isin(selected_types)
    & df["year"].between(selected_years[0], selected_years[1])
    & df["tag_count"].between(selected_tag[0], selected_tag[1])
].copy()

if fdf.empty:
    st.warning("The current filters return zero rows. Broaden the selection to continue.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Posts", f"{len(fdf):,}")
k2.metric("Median engagement", f"{fdf['engagement'].median():,.0f}")
k3.metric("Median likes", f"{fdf['liked_count_n'].median():,.0f}")
k4.metric("Median saves", f"{fdf['collected_count_n'].median():,.0f}")
k5.metric("Video share", f"{(fdf['type'].eq('video').mean()*100):.1f}%")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Executive summary", "Drivers", "Timing", "Evidence", "Data"])

with tab1:
    st.subheader("What should a creator do?")
    st.markdown(recommendation_text(fdf))

    col1, col2 = st.columns(2)
    with col1:
        cat = fdf.groupby("feature")["engagement"].median().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bars = ax.barh([c.replace("ai-", "").capitalize() for c in cat.index], cat.values)
        ax.set_title("Median engagement by category")
        ax.set_xlabel("Median engagement")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, cat.values):
            ax.text(val + max(cat.values) * 0.01, bar.get_y() + bar.get_height()/2, f"{val:,.0f}", va="center", fontsize=8)
        st.pyplot(fig)
    with col2:
        med = fdf.groupby("type")["engagement"].median().reindex(["normal", "video"])
        fig, ax = plt.subplots(figsize=(6, 4.8))
        ax.bar(["Normal", "Video"], med.values)
        ax.set_title("Median engagement by post type")
        ax.set_ylabel("Median engagement")
        ax.spines[["top", "right"]].set_visible(False)
        for i, val in enumerate(med.values):
            ax.text(i, val + max(med.values) * 0.03, f"{val:,.0f}", ha="center")
        st.pyplot(fig)

with tab2:
    left, right = st.columns(2)
    with left:
        tag_bins = pd.cut(fdf["tag_count"], bins=[-0.1, 2, 4, 6, 8, 100], labels=["0–2", "3–4", "5–6", "7–8", "9+"])
        tag_med = fdf.groupby(tag_bins, observed=True)["engagement"].median()
        tag_n = fdf.groupby(tag_bins, observed=True)["engagement"].count()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(tag_med.index.astype(str), tag_med.values)
        ax.set_title("Tag-count sweet spot")
        ax.set_xlabel("Tag-count bucket")
        ax.set_ylabel("Median engagement")
        ax.spines[["top", "right"]].set_visible(False)
        for i, (v, n) in enumerate(zip(tag_med.values, tag_n.values)):
            ax.text(i, v + max(tag_med.values) * 0.03, f"{v:,.0f}\n(n={n})", ha="center", fontsize=8)
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sample = fdf[["title_len", "engagement"]].dropna().copy()
        if len(sample) > 500:
            sample = sample.sample(500, random_state=42)
        ax.scatter(sample["title_len"], sample["engagement"], alpha=0.35)
        z = np.polyfit(sample["title_len"], np.log1p(sample["engagement"]), 1)
        xs = np.linspace(sample["title_len"].min(), sample["title_len"].max(), 100)
        ax2_y = np.expm1(z[0] * xs + z[1])
        ax.plot(xs, ax2_y)
        ax.set_title("Title length vs engagement")
        ax.set_xlabel("Title length")
        ax.set_ylabel("Engagement")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.caption("The fitted curve is descriptive, not causal.")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        year_med = fdf.groupby("year")["engagement"].median().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(year_med.index, year_med.values, marker="o", linewidth=2)
        ax.fill_between(year_med.index, year_med.values, alpha=0.15)
        ax.set_title("Median engagement over time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Median engagement")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
    with c2:
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = fdf.pivot_table(index="weekday", columns="hour", values="engagement", aggfunc="median").reindex(weekday_order)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(pivot.fillna(0).values, aspect="auto")
        ax.set_title("Median engagement heatmap: weekday × hour")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Weekday")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=0)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

with tab4:
    st.subheader("Statistical evidence")
    c1, c2 = st.columns(2)
    video = fdf.loc[fdf["type"] == "video", "engagement"].dropna()
    normal = fdf.loc[fdf["type"] == "normal", "engagement"].dropna()
    if len(video) > 0 and len(normal) > 0:
        mann = stats.mannwhitneyu(video, normal, alternative="two-sided")
        c1.metric("Mann–Whitney U p-value", f"{mann.pvalue:.4g}")
    groups = [g["engagement"].dropna().values for _, g in fdf.groupby("feature") if len(g) > 0]
    if len(groups) >= 2:
        kw = stats.kruskal(*groups)
        c2.metric("Kruskal–Wallis p-value", f"{kw.pvalue:.4g}")

    corr_cols = ["engagement", "liked_count_n", "collected_count_n", "comment_count_n", "share_count_n", "tag_count", "title_len", "desc_len"]
    corr = fdf[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation matrix")
    st.pyplot(fig)

with tab5:
    st.subheader("Filtered benchmark table")
    summary = fdf.groupby("feature").agg(
        posts=("engagement", "count"),
        median_engagement=("engagement", "median"),
        mean_engagement=("engagement", "mean"),
        median_likes=("liked_count_n", "median"),
        median_saves=("collected_count_n", "median"),
        pct_video=("is_video", "mean"),
        median_tags=("tag_count", "median"),
    ).round(2)
    summary.index = [x.replace("ai-", "").capitalize() for x in summary.index]
    summary["pct_video"] = (summary["pct_video"] * 100).round(1)
    st.dataframe(summary.sort_values("median_engagement", ascending=False), use_container_width=True)
    st.download_button("Download filtered data as CSV", fdf.to_csv(index=False).encode("utf-8-sig"), file_name="filtered_posts.csv", mime="text/csv")

st.caption("Built for ACC102 using Python, Streamlit, and statistical testing. Replace placeholder links in the README before submission.")
