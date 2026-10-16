from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

MODELS_DIR = Path(__file__).parent.parent / "models"
MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
# Theme colours from .streamlit/config.toml, reused for the readout line and the plots.
ACCENT = st.get_option("theme.primaryColor")
TEXT = st.get_option("theme.textColor")
SURFACE = st.get_option("theme.secondaryBackgroundColor")
RULE = st.get_option("theme.borderColor")
PLOT_RC = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "axes.edgecolor": RULE,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": RULE,
    "ytick.color": RULE,
    "xtick.labelcolor": TEXT,
    "ytick.labelcolor": TEXT,
    "text.color": TEXT,
    "legend.facecolor": SURFACE,
    "legend.edgecolor": RULE,
    "legend.labelcolor": TEXT,
}
SCORE = st.column_config.NumberColumn(format="%.3f")


@st.cache_resource
def load_bundle(track):
    """Models, test-set results and input defaults exported by the last cell of each notebook."""
    return joblib.load(MODELS_DIR / f"{track}.joblib")


def readout(container, figure, sentence):
    """The answer from the top-ranked model, set large above the comparison table."""
    container.markdown(f'<p class="readout"><strong>{figure}</strong>{sentence}</p>', unsafe_allow_html=True)


def regression_tab():
    bundle = load_bundle("regression")
    ranges = bundle["feature_ranges"]
    st.write(
        "Enter the ambient conditions and each regressor predicts the plant's net hourly output, PE. "
        "Inputs are limited to the range seen in training."
    )

    labels = {
        "AT": "Ambient temperature (°C)",
        "V": "Exhaust vacuum (cm Hg)",
        "AP": "Ambient pressure (mbar)",
        "RH": "Relative humidity (%)",
    }
    inputs = {}
    for column, (feature, label) in zip(st.columns(4), labels.items()):
        inputs[feature] = column.number_input(
            label,
            min_value=float(ranges.loc["min", feature]),
            max_value=float(ranges.loc["max", feature]),
            value=float(ranges.loc["median", feature]),
            key=f"reg_{feature}",
        )

    X = pd.DataFrame([inputs])
    X["AT_x_V"] = X["AT"] * X["V"]  # engineered feature from the regression notebook

    results = bundle["results"]
    table = pd.DataFrame({
        "Model": results["Model"],
        "Predicted PE (MW)": [bundle["models"][name].predict(X)[0] for name in results["Model"]],
        "Test R²": results["R2"],
        "Test RMSE (MW)": results["RMSE"],
        "Test MAE (MW)": results["MAE"],
    })
    best = table.iloc[0]
    readout(st, f"{best['Predicted PE (MW)']:.2f} MW", f"predicted by {best['Model']}, the model with the best test R²")
    st.subheader("Predictions from all 10 models, ranked by test R²")
    st.dataframe(
        table.round(3), hide_index=True,
        column_config={"Predicted PE (MW)": SCORE, "Test R²": SCORE, "Test RMSE (MW)": SCORE, "Test MAE (MW)": SCORE},
    )


def repayment_label(status):
    if status == -2:
        return "No consumption"
    if status == -1:
        return "Paid in full"
    if status == 0:
        return "Revolving credit"
    return f"{status} month{'s' if status > 1 else ''} late"


def classification_tab():
    bundle = load_bundle("classification")
    defaults = bundle["input_defaults"]
    st.write(
        "Describe a credit-card client and each classifier predicts whether they default on next month's payment. "
        "Defaults are the training-set medians."
    )

    st.subheader("Client")
    columns = st.columns(5)
    inputs = {
        "LIMIT_BAL": columns[0].number_input(
            "Credit limit (NT$)", min_value=10_000.0, value=float(defaults["LIMIT_BAL"]), step=10_000.0, key="cls_LIMIT_BAL"
        ),
        "SEX": columns[1].selectbox(
            "Sex", [1, 2], index=int(defaults["SEX"]) - 1, format_func={1: "Male", 2: "Female"}.get, key="cls_SEX"
        ),
        "EDUCATION": columns[2].selectbox(
            "Education", [1, 2, 3, 4], index=int(defaults["EDUCATION"]) - 1,
            format_func={1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}.get, key="cls_EDUCATION",
        ),
        "MARRIAGE": columns[3].selectbox(
            "Marital status", [1, 2, 3], index=int(defaults["MARRIAGE"]) - 1,
            format_func={1: "Married", 2: "Single", 3: "Other"}.get, key="cls_MARRIAGE",
        ),
        "AGE": columns[4].number_input("Age", min_value=21, max_value=79, value=int(defaults["AGE"]), key="cls_AGE"),
    }

    st.subheader("Last six months (2005)")
    statuses = list(range(-2, 9))
    for month, column, pay, bill, pay_amt in zip(MONTHS, st.columns(6), PAY_COLS, BILL_COLS, PAY_AMT_COLS):
        column.markdown(f"**{month}**")
        inputs[pay] = column.selectbox(
            "Repayment status", statuses, index=statuses.index(int(defaults[pay])), format_func=repayment_label, key=f"cls_{pay}"
        )
        inputs[bill] = column.number_input("Bill (NT$)", value=float(defaults[bill]), step=1_000.0, key=f"cls_{bill}")
        inputs[pay_amt] = column.number_input(
            "Paid (NT$)", min_value=0.0, value=float(defaults[pay_amt]), step=1_000.0, key=f"cls_{pay_amt}"
        )

    X = pd.DataFrame([inputs])
    # Engineered features from the classification notebook.
    X["AVG_BILL_AMT"] = X[BILL_COLS].mean(axis=1)
    X["UTILIZATION_RATIO"] = (X["AVG_BILL_AMT"] / X["LIMIT_BAL"]).clip(lower=0)
    X["DELAY_MONTHS_COUNT"] = (X[PAY_COLS] > 0).sum(axis=1)

    results = bundle["results"]
    models = [bundle["models"][name] for name in results["Model"]]
    table = pd.DataFrame({
        "Model": results["Model"],
        "Part": results["Part"],
        "Prediction": ["Default" if model.predict(X)[0] == 1 else "No default" for model in models],
        # SVC was fitted without probability=True, so it has no predict_proba.
        "P(default)": [model.predict_proba(X)[0, 1] if hasattr(model, "predict_proba") else None for model in models],
        "Test F1": results["Weighted F1"],
        "Test ROC-AUC": results["ROC-AUC"],
    })
    best = table.iloc[0]
    probability = "" if pd.isna(best["P(default)"]) else f", P(default) {best['P(default)']:.2f}"
    readout(st, best["Prediction"], f"from {best['Model']}, the model with the best test weighted F1{probability}")
    st.subheader("Predictions from all 10 models, ranked by test weighted F1")
    st.dataframe(
        table.round(3), hide_index=True,
        column_config={
            "P(default)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Test F1": SCORE,
            "Test ROC-AUC": SCORE,
        },
    )
    st.caption("SVM was trained without probability estimates, so it gives a class but no P(default).")


def plot_cluster_map(bundle, name, new_point):
    coordinates = bundle["pca_coordinates"]
    plot_df = pd.DataFrame({"x": coordinates[:, 0], "y": coordinates[:, 1], "Cluster": bundle["labels"][name].astype(str)})
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(
            data=plot_df, x="x", y="y", hue="Cluster", hue_order=sorted(plot_df["Cluster"].unique()),
            palette="tab10", s=30, alpha=0.6, ax=ax,
        )
        ax.scatter(
            new_point[0], new_point[1], marker="*", s=400, color=TEXT, edgecolor=ACCENT, linewidth=1.5, label="Your customer"
        )
        ax.set(title=f"PCA projection: {name}", xlabel="Principal component 1", ylabel="Principal component 2")
        legend = ax.legend(title="Cluster")
        # Shrink the star's legend handle to the dots' size so the rows are evenly spaced.
        legend.legend_handles[-1].set_sizes([80])
        fig.tight_layout()
    return fig


def clustering_tab():
    bundle = load_bundle("clustering")
    defaults = bundle["input_defaults"]
    st.write(
        "Enter a wholesale customer's annual spend per category (monetary units) and each model assigns them to a spending segment."
    )

    inputs = {
        feature: column.number_input(
            feature.replace("_", " "), min_value=0.0, value=float(defaults[feature]), step=500.0, key=f"clu_{feature}"
        )
        for column, feature in zip(st.columns(6), defaults.index)
    }
    scaled = bundle["preprocess"].transform(pd.DataFrame([inputs]))

    results = bundle["results"]
    table = pd.DataFrame({
        "Model": results["Model"],
        "Cluster": [int(bundle["models"][name].predict(scaled)[0]) for name in results["Model"]],
        "Silhouette ↑": results["Silhouette Score"],
        "Davies-Bouldin ↓": results["Davies-Bouldin Index"],
        "Calinski-Harabasz ↑": results["Calinski-Harabasz Index"],
    })
    st.subheader("Assigned cluster")
    st.dataframe(
        table.round(3), hide_index=True,
        column_config={"Silhouette ↑": SCORE, "Davies-Bouldin ↓": SCORE, "Calinski-Harabasz ↑": SCORE},
    )
    st.caption(
        "Agglomerative clustering cannot label new points, so the app assigns your customer to the nearest "
        "Ward cluster centroid. Cluster numbers are independent per model; use the profiles to compare them."
    )

    new_point = bundle["pca"].transform(scaled)[0]
    for column, (name, cluster) in zip(st.columns(2, gap="large"), zip(table["Model"], table["Cluster"])):
        readout(column, f"Cluster {cluster}", f"according to {name}")
        fig = plot_cluster_map(bundle, name, new_point)
        column.pyplot(fig)
        plt.close(fig)
        column.markdown(f"Median annual spend per {name} cluster")
        column.dataframe(bundle["profiles"][name])


st.set_page_config(page_title="ML Capstone Project", layout="wide")
# Only the readout line needs CSS; everything else is themed in .streamlit/config.toml.
st.html(f"""<style>
.readout {{ font-family: "IBM Plex Serif", serif; font-size: 1.125rem; line-height: 1.4; margin: 1.5rem 0 0.25rem; }}
.readout strong {{ font-weight: 500; font-size: 2.25rem; color: {ACCENT}; letter-spacing: -0.01em; margin-right: 0.5rem; }}
</style>""")
st.title("ML Capstone Project")
st.caption(
    "Enter one case and compare what each trained model predicts for it, "
    "next to that model's test-set score."
)

regression, classification, clustering = st.tabs([
    "Regression: power plant output",
    "Classification: credit default",
    "Clustering: wholesale customers",
])
with regression:
    regression_tab()
with classification:
    classification_tab()
with clustering:
    clustering_tab()
