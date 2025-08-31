import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="DBSCAN - Transaction Outliers", layout="centered")
st.title("💳 Day 18 — DBSCAN: Detect Outliers in Transaction Data")

# Generate synthetic transaction dataset
@st.cache_data
def generate_data(n=800, random_state=42):
    rng = np.random.RandomState(random_state)
    # normal transactions (clusters)
    cluster1 = np.column_stack([rng.normal(50, 8, n//3), rng.normal(12, 3, n//3)])   # amount, hour
    cluster2 = np.column_stack([rng.normal(200, 15, n//3), rng.normal(20, 2, n//3)])
    cluster3 = np.column_stack([rng.normal(400, 30, n//3), rng.normal(15, 4, n//3)])
    normal = np.vstack([cluster1, cluster2, cluster3])
    # outliers/fraud
    outliers = np.column_stack([rng.uniform(800, 2000, n//20), rng.uniform(0,24,n//20)])
    X = np.vstack([normal, outliers])
    df = pd.DataFrame(X, columns=["Amount", "Hour"])
    return df

df = generate_data()
st.subheader("📂 Sample Transactions")
st.dataframe(df.sample(10))

# Controls
eps = st.slider("DBSCAN eps (neighborhood radius)", 1.0, 50.0, 8.0)
min_samples = st.slider("DBSCAN min_samples", 2, 20, 5)

# Scale
scaler = StandardScaler()
Xs = scaler.fit_transform(df[["Amount", "Hour"]])

# DBSCAN
db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
labels = db.labels_
df["cluster"] = labels
df["outlier"] = df["cluster"] == -1

# Metrics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = (df["outlier"]).sum()
st.write(f"Detected clusters: **{n_clusters}** — Detected outliers: **{n_outliers}**")

# Plot
fig, ax = plt.subplots(figsize=(8,5))
palette = sns.color_palette("tab10", n_colors=max(1, n_clusters))
sns.scatterplot(x="Amount", y="Hour", hue="cluster", data=df.replace({"cluster":{-1: "outlier"}}),
                palette=palette, legend="full", ax=ax, s=40, edgecolor="k")
ax.set_title("Transactions: clusters and outliers (DBSCAN)")
st.pyplot(fig)

# Show top outliers
st.subheader("🔍 Top Outlier Transactions")
st.dataframe(df[df["outlier"]].sort_values(by="Amount", ascending=False).reset_index(drop=True))
st.success("✅ DBSCAN clustering & outlier detection complete")
