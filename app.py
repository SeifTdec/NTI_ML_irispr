import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


MODEL_PATH = "iris_model.pkl"
st.set_page_config(page_title="Iris AI Studio", page_icon="🌼", layout="wide")

st.markdown(
	"""
	<style>
	.stApp { background: radial-gradient(ellipse at top left, #20294b 0, #101426 48%, #0b1020 100%); color: #f5f7ff; }
	[data-testid="stSidebar"] { background: #121a32; border-right: 1px solid #293451; }
	[data-testid="stMetric"] { background: linear-gradient(145deg,#1c2846,#171f38); padding: 18px; border: 1px solid #344363; border-radius: 16px; }
	.hero { padding: 28px 32px; border-radius: 22px; background: linear-gradient(110deg,#25345b,#34275a 65%,#4a2860); border: 1px solid #61739b; margin-bottom: 24px; }
	.hero h1 { margin: 0; font-size: 2.5rem; color: white; }
	.hero p { color: #d4dcf3; margin-bottom: 0; }
	.tag { color:#ffbd69; font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:.75rem; }
	div.stButton > button { background: linear-gradient(90deg,#ff9900,#ffb84d); color:#171a27; border:0; border-radius:10px; font-weight:700; }
	</style>
	""",
	unsafe_allow_html=True,
)

iris = load_iris(as_frame=True)
feature_names = list(iris.feature_names)

try:
	with open(MODEL_PATH, "rb") as model_file:
		model = pickle.load(model_file)
except (FileNotFoundError, pickle.PickleError, EOFError, AttributeError, ValueError):
	model = RandomForestClassifier(n_estimators=180, random_state=42)
	model.fit(iris.data, iris.target)
	with open(MODEL_PATH, "wb") as model_file:
		pickle.dump(model, model_file)

st.sidebar.markdown("## 🌼 Iris AI Studio")
st.sidebar.caption("A tiny flower-classification playground")
page = st.sidebar.radio("Navigate", ["Overview", "Data summary", "Test a sample"])
st.sidebar.markdown("---")
st.sidebar.caption("Powered by scikit-learn · Model: `iris_model.pkl`")

if page == "Overview":
	st.markdown('<div class="hero"><div class="tag">AWS PartyRock-inspired · ML playground</div><h1>Meet the Iris classifier 🌼</h1><p>Explore the classic Iris dataset and try an interactive flower prediction.</p></div>', unsafe_allow_html=True)
	a, b, c = st.columns(3)
	a.metric("Flower samples", f"{len(iris.data):,}")
	b.metric("Iris species", str(len(iris.target_names)))
	c.metric("Input features", str(len(feature_names)))
	left, right = st.columns([1.2, 1])
	with left:
		st.subheader("Your model workspace")
		st.write("Explore dataset statistics, then enter measurements to identify an Iris species.")
		st.markdown("- **Setosa** · **Versicolor** · **Virginica**\n- Random forest classifier trained on the built-in sklearn dataset\n- Model saved locally as `iris_model.pkl`")
	with right:
		st.subheader("Species at a glance")
		counts = pd.Series(iris.target).map(dict(enumerate(iris.target_names))).value_counts().reindex(iris.target_names)
		st.bar_chart(counts, color="#ff9900")

elif page == "Data summary":
	st.markdown('<div class="hero"><div class="tag">Explore the data</div><h1>Dataset summary 📊</h1><p>150 measurements across three Iris species, with four numeric features.</p></div>', unsafe_allow_html=True)
	st.subheader("Feature statistics (cm)")
	st.dataframe(iris.data.describe().T.rename_axis("Feature"), use_container_width=True)
	st.subheader("Samples by species")
	counts = pd.Series(iris.target).map(dict(enumerate(iris.target_names))).value_counts().reindex(iris.target_names)
	st.bar_chart(counts, color="#ff9900")
	with st.expander("Preview raw data"):
		preview = iris.data.copy()
		preview["species"] = pd.Series(iris.target).map(dict(enumerate(iris.target_names)))
		st.dataframe(preview, use_container_width=True, hide_index=True)

else:
	st.markdown('<div class="hero"><div class="tag">Interactive prediction</div><h1>Test a new sample 🔮</h1><p>Adjust the flower measurements and let the model identify its species.</p></div>', unsafe_allow_html=True)
	st.subheader("Enter measurements (cm)")
	defaults = [5.1, 3.5, 1.4, 0.2]
	columns = st.columns(2)
	values = []
	for index, feature in enumerate(feature_names):
		with columns[index % 2]:
			values.append(st.number_input(feature.title(), min_value=0.0, max_value=10.0, value=defaults[index], step=0.1, format="%.1f"))
	if st.button("✨ Predict species", use_container_width=True):
		sample = pd.DataFrame([values], columns=feature_names)
		prediction = int(model.predict(sample)[0])
		st.success(f"The model predicts **{iris.target_names[prediction].title()}**.")
		if hasattr(model, "predict_proba"):
			probabilities = model.predict_proba(sample)[0]
			confidence = pd.Series(probabilities, index=iris.target_names, name="Confidence")
			st.subheader("Model confidence")
			st.bar_chart(confidence, color="#ff9900")
		st.caption("Predictions are estimates from a model trained on the classic Iris dataset.")
