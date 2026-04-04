"""
app.py — Streamlit Deployment App
World Development Measurement: Country Clustering
=================================================
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Development Clustering",
    page_icon="🌍",
    layout="wide",
)

# ── Helper: Load Model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str = "models/model_kmeans.pkl"):
    """Load the saved KMeans model from disk."""
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning(
            "⚠️  Pre-trained model not found at `models/model_kmeans.pkl`. "
            "A fresh KMeans(n_clusters=3) will be trained on the uploaded data."
        )
        return None


# ── Helper: Preprocess Data ───────────────────────────────────────────────────
@st.cache_data
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the preprocessing steps from the notebook:
      1. Strip currency / percentage characters
      2. Cast to float
      3. Group by Country (mean)
      4. Fill NaN with 0
      5. Drop low-signal columns
    """
    currency_cols = ["GDP", "Health Exp/Capita", "Tourism Inbound", "Tourism Outbound"]
    for col in currency_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )

    if "Business Tax Rate" in df.columns:
        df["Business Tax Rate"] = (
            df["Business Tax Rate"]
            .astype(str)
            .str.replace("%", "", regex=False)
        )

    float_cols = [
        "GDP", "Health Exp/Capita", "Tourism Inbound",
        "Tourism Outbound", "Business Tax Rate",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Group by country
    df_merged = df.groupby("Country").mean(numeric_only=True)
    df_merged.fillna(0, inplace=True)

    # Drop columns that were removed in the notebook
    drop_cols = ["Health Exp % GDP", "Number of Records"]
    df_merged.drop(
        columns=[c for c in drop_cols if c in df_merged.columns],
        inplace=True,
    )

    return df_merged


# ── Helper: Assign Cluster Labels ─────────────────────────────────────────────
CLUSTER_LABELS = {
    0: "🟢 Cluster 0 — Developing Economies",
    1: "🔵 Cluster 1 — Emerging Economies",
    2: "🔴 Cluster 2 — Developed Economies",
}


def label_clusters(predictions: np.ndarray) -> list[str]:
    return [CLUSTER_LABELS.get(p, f"Cluster {p}") for p in predictions]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/921/921490.png", width=80)
    st.title("🌍 World Dev Clustering")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Cluster Explorer", "🔮 Predict Country Cluster", "📈 EDA Insights"],
    )

    st.markdown("---")
    st.caption("Built with Scikit-learn & Streamlit")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🌍 World Development Measurement — Country Clustering")
    st.markdown(
        """
        This application segments countries across the globe into meaningful clusters
        based on **25 economic, health, demographic, and infrastructure indicators**.

        ---
        ### 🎯 Business Objective
        Help policymakers, researchers, and NGOs identify structural groupings among nations
        to benchmark performance and target development interventions.

        ### 🤖 Model Used
        **K-Means Clustering (3 clusters)** — selected after comparing K-Means, DBSCAN,
        and Hierarchical Clustering using Silhouette Score, Calinski-Harabasz Index,
        and Davies-Bouldin Index.

        ### 📂 How to Use
        1. Upload the world development dataset (CSV) in the **Cluster Explorer** page.
        2. View cluster assignments for each country.
        3. Use the **Predict** page to assign a new / custom country to a cluster.
        4. Explore key EDA insights in the **EDA Insights** page.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Clustering Algorithm", "K-Means")
    col2.metric("Optimal Clusters", "3")
    col3.metric("Features Used", "22")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTER EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Cluster Explorer":
    st.title("📊 Cluster Explorer")
    st.markdown("Upload the dataset to see cluster assignments for each country.")

    uploaded = st.file_uploader(
        "Upload CSV (World_development_mesurement.csv)",
        type=["csv", "xlsx"],
    )

    if uploaded:
        if uploaded.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)

        st.success(f"✅ Loaded {len(df_raw)} rows, {df_raw.shape[1]} columns.")

        with st.spinner("Preprocessing data…"):
            df_clean = preprocess(df_raw.copy())

        # Load or train model
        model = load_model()
        if model is None:
            model = KMeans(n_clusters=3, random_state=42)
            model.fit(df_clean)
            st.info("ℹ️  Trained a fresh KMeans model on the uploaded data.")

        predictions = model.predict(df_clean)
        df_result = df_clean.copy()
        df_result["Cluster"] = predictions
        df_result["Cluster Label"] = label_clusters(predictions)
        df_result.reset_index(inplace=True)

        # ── Summary ──────────────────────────────────────────────────────────
        st.markdown("### Cluster Distribution")
        dist = df_result["Cluster Label"].value_counts().reset_index()
        dist.columns = ["Cluster", "Count"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(dist, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.pie(
                dist["Count"],
                labels=dist["Cluster"],
                autopct="%1.1f%%",
                startangle=140,
            )
            ax.set_title("Countries per Cluster")
            st.pyplot(fig)

        # ── Full Table ────────────────────────────────────────────────────────
        st.markdown("### Country — Cluster Assignments")
        cluster_filter = st.multiselect(
            "Filter by Cluster",
            options=sorted(df_result["Cluster Label"].unique()),
            default=sorted(df_result["Cluster Label"].unique()),
        )
        filtered = df_result[df_result["Cluster Label"].isin(cluster_filter)]
        st.dataframe(
            filtered[["Country", "Cluster Label", "GDP", "Life Expectancy Female",
                       "Infant Mortality Rate", "Internet Usage"]],
            use_container_width=True,
            height=400,
        )

        # ── Elbow Curve ───────────────────────────────────────────────────────
        with st.expander("🔍 Show Elbow Curve (WCSS)"):
            ssd = []
            K = range(1, 11)
            for k in K:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(df_clean)
                ssd.append(km.inertia_)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.plot(K, ssd, "bx-")
            ax2.set_xlabel("Number of Clusters (K)")
            ax2.set_ylabel("WCSS (Inertia)")
            ax2.set_title("Elbow Method — Optimal K")
            st.pyplot(fig2)
    else:
        st.info("👆 Please upload the dataset to get started.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Country Cluster":
    st.title("🔮 Predict Cluster for a New Country")
    st.markdown(
        "Enter values for the development indicators below. "
        "The model will assign the country to the most similar cluster."
    )

    model = load_model()
    if model is None:
        st.error(
            "No pre-trained model found. Please run the notebook first to generate "
            "`models/model_kmeans.pkl`, or use the Cluster Explorer to train on uploaded data."
        )
        st.stop()

    # ── Input Form ────────────────────────────────────────────────────────────
    features = {
        "Birth Rate": (0.005, 0.06, 0.02),
        "Business Tax Rate": (5.0, 80.0, 30.0),
        "CO2 Emissions": (0.0, 1e10, 1e7),
        "Days to Start Business": (1, 200, 20),
        "Ease of Business": (1, 190, 80),
        "Energy Usage": (0.0, 1e13, 1e9),
        "GDP": (0.0, 2e13, 5e11),
        "Health Exp/Capita": (0.0, 10000.0, 1000.0),
        "Hours to do Tax": (0.0, 2500.0, 300.0),
        "Infant Mortality Rate": (0.001, 0.15, 0.03),
        "Internet Usage": (0.0, 1.0, 0.5),
        "Lending Interest": (0.0, 50.0, 10.0),
        "Life Expectancy Female": (40.0, 90.0, 73.0),
        "Life Expectancy Male": (40.0, 90.0, 68.0),
        "Mobile Phone Usage": (0.0, 2.5, 1.0),
        "Population 0-14": (0.05, 0.55, 0.28),
        "Population 15-64": (0.4, 0.75, 0.63),
        "Population 65+": (0.01, 0.28, 0.08),
        "Population Total": (1e4, 1.4e9, 3e7),
        "Population Urban": (0.1, 1.0, 0.55),
        "Tourism Inbound": (0.0, 2e11, 1e9),
        "Tourism Outbound": (0.0, 2e11, 1e9),
    }

    col1, col2 = st.columns(2)
    input_vals = {}
    feature_list = list(features.items())

    for i, (feat, (mn, mx, default)) in enumerate(feature_list):
        col = col1 if i % 2 == 0 else col2
        input_vals[feat] = col.number_input(
            feat, min_value=float(mn), max_value=float(mx), value=float(default)
        )

    if st.button("🔮 Predict Cluster", type="primary"):
        input_df = pd.DataFrame([input_vals])
        pred = model.predict(input_df)[0]
        label = CLUSTER_LABELS.get(pred, f"Cluster {pred}")
        st.success(f"### Predicted Cluster: {label}")
        st.balloons()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA Insights":
    st.title("📈 EDA Insights")
    st.markdown(
        "Upload the dataset to explore key findings from the Exploratory Data Analysis."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv", "xlsx"],
        key="eda_upload",
    )

    if uploaded:
        df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        df_clean = preprocess(df_raw.copy())

        tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Boxplots", "GDP Analysis"])

        with tab1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(16, 10))
            sns.heatmap(df_clean.corr(), annot=True, fmt=".1f", ax=ax, cmap="coolwarm")
            st.pyplot(fig)
            st.markdown(
                """
                **Key Findings:**
                - Birth Rate has strong **positive** correlation with Infant Mortality & Population 0–14.
                - Birth Rate has strong **negative** correlation with Life Expectancy.
                - CO₂ Emissions correlates strongly with GDP, Energy Usage, and Tourism.
                - Ease of Business shows a **negative** relationship with Internet Usage.
                """
            )

        with tab2:
            st.subheader("Boxplots — Feature Distribution")
            col_select = st.selectbox("Select Feature", options=df_clean.columns.tolist())
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            sns.boxplot(x=df_clean[col_select], ax=ax2)
            ax2.set_title(f"Distribution of {col_select}")
            st.pyplot(fig2)

        with tab3:
            st.subheader("Top 15 Countries by GDP")
            df_clean_reset = df_clean.reset_index()
            top_gdp = df_clean_reset.sort_values("GDP", ascending=False).head(15)
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            sns.barplot(x="Country", y="GDP", data=top_gdp, ax=ax3)
            plt.xticks(rotation=45, ha="right")
            ax3.set_title("Top 15 Countries by GDP")
            st.pyplot(fig3)
    else:
        st.info("👆 Upload the dataset to explore EDA charts.")
