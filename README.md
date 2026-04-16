# ACC102 Mini Assignment — 
## Analysis of Factors Affecting Engagement with AIGC-related Notes on Xiaohongshu (Red Note)

**Track:** Interactive Tool / Streamlit submission  
**Live app:** `ADD_YOUR_STREAMLIT_LINK_HERE`  
**GitHub repository:** `ADD_YOUR_GITHUB_REPO_LINK_HERE`  
**Notebook:** [notebook.ipynb](./notebook.ipynb)  
**Reflection:** [reflection.md](./reflection.md)

---

## 1. Problem and user
This project investigates which factors are most strongly associated with higher engagement for AIGC-related Xiaohongshu posts. The intended users are creators, marketers, and students who want evidence-based guidance on what kind of AIGC content performs best on Red Note.

## 2. Data
- **Dataset:** Xiaohongshu AIGC Comments and Posts Dataset
- **Core file used:** `Posts.csv`
- **Rows:** 1,582
- **Columns:** 37
- **Coverage:** 8 AIGC categories from 2021–2024
- **Encoding:** GBK
- **Engagement variables:** likes, saves, comments, shares

### Key engineered features
- `engagement` = likes + saves + comments + shares
- `tag_count` = number of hashtags / tags attached to the post
- `title_len` and `desc_len` = text-length proxies
- `year`, `month`, `weekday`, `hour` = posting time controls
- `log_engagement` = log-transformed engagement for robust modelling

## 3. Methods
The workflow combines data cleaning, feature engineering, descriptive analytics, statistical testing, and a lightweight predictive benchmark.

1. Load `Posts.csv` with GBK encoding  
2. Parse Chinese compact counts such as `2.2万` into numeric values  
3. Build a composite engagement score  
4. Create temporal, tagging, and text-length features  
5. Compare engagement by category, post type, tag strategy, and year  
6. Run statistical tests:
   - Mann–Whitney U test for **video vs normal** posts
   - Kruskal–Wallis test for **cross-category differences**
7. Estimate a simple linear benchmark on `log_engagement` using post characteristics
8. Surface the results in a Streamlit app with filters and recommendations

## 4. Key findings
- **Category is the dominant factor.** `Print` posts have the highest median engagement (18,714), while `Car` posts have the lowest (145).
- **Video posts materially outperform normal posts.** Median engagement is **3.45×** higher for video posts.
- **Tagging shows a sweet spot.** The best-performing bucket is **9+ tags**; both under-tagging and over-tagging reduce typical engagement.
- **Engagement rises over time.** Median engagement is much higher in later years than in 2021, showing growing audience interest in AIGC content.
- **Statistical evidence supports the patterns.**
  - Mann–Whitney U p-value: **6.821e-14**
  - Kruskal–Wallis p-value: **1.41e-90**

## 5. Practical recommendations
1. Prioritise categories with proven demand, especially print, literature, and advertisement-related formats.
2. Convert suitable posts into short-form video whenever production time allows.
3. Aim for roughly **5–6 tags** instead of tag stuffing.
4. Use save-oriented educational framing for categories where reference value matters.
5. Benchmark new content against the dashboard rather than relying on raw likes alone.

## 6. Repository structure
```text
acc102-aigc-engagement/
├── app.py
├── notebook.ipynb
├── README.md
├── reflection.md
├── requirements.txt
├── Posts.csv
└── figures/
    ├── fig1_category_engagement.png
    ├── fig2_post_type.png
    ├── fig3_tags.png
    ├── fig4_trend.png
    ├── fig5_breakdown.png
    ├── fig6_correlation_matrix.png
    └── fig7_benchmark_map.png
```

## 7. How to run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
streamlit run app.py
```

## 8. Demo checklist before submission
- Replace the placeholder Streamlit and GitHub links
- Record a **1–3 minute** walkthrough video
- Keep `Posts.csv` in the project root if running locally
- Confirm the app opens without path errors
- Ensure the notebook runs top-to-bottom in order

## 9. Limitations
- The dataset does not include follower counts, which likely influence engagement.
- This is an observational analysis, so correlations should not be interpreted as strict causal effects.
- The data may overrepresent posts surfaced by the scraping process.
- Engagement is platform- and algorithm-dependent, so results may shift over time.

## 10. Next steps
- Merge post-level and comment-level data for sentiment-aware analysis.
- Add creator-level controls if follower data becomes available.
- Extend the dashboard with upload-your-own-post benchmarking.
- Explore classification models for predicting high-engagement posts.
