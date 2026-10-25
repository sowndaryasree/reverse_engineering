"""Early Detection of Deceptive News Propagation using GNNs - Streamlit demo.
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import utils
from graph_utils import bfs_depth, truncate, radial_layout

st.set_page_config(page_title="Propagation-based News Risk (GNN)", page_icon="🕸️", layout="wide")

# ---------- safety checks: files must exist ----------
missing = utils.missing_files()
if missing:
    st.error("Missing required files in the `artifacts/` folder (or model_def.py): " + ", ".join(missing))
    st.info("Copy ALL files from the Colab `artifacts.zip` into the `artifacts` folder next to app.py, "
            "and keep model_def.py next to app.py. Then refresh this page.")
    st.stop()


@st.cache_resource
def get_model_and_cfg():
    cfg = utils.load_config()
    return utils.load_model(cfg), cfg


@st.cache_data
def get_samples():
    return utils.load_samples()


try:
    model, cfg = get_model_and_cfg()
    samples = get_samples()
    metrics = utils.load_metrics()
except Exception as e:  # shape mismatch, wrong torch version, corrupt file...
    st.error("Could not load the trained model. Copy this message for debugging:")
    st.code(repr(e))
    st.stop()

CLASS_NAMES = cfg.get("class_names", ["class_0", "class_1"])
FAKE_IDX = cfg.get("fake_class_index", None)  # set in config.json ONLY after confirming the label mapping


def cname(i):
    return CLASS_NAMES[i]


def draw_graph(depth, edge_index, keep, root):
    pos = radial_layout(depth)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    segs = [(pos[s], pos[t]) for s, t in zip(edge_index[0], edge_index[1]) if s < t]
    ax.add_collection(LineCollection(segs, colors="#c9ced6", linewidths=0.5, zorder=1))
    gone = ~keep & (depth >= 0)
    ax.scatter(pos[gone, 0], pos[gone, 1], s=10, c="#e3e6ea", zorder=2, label="not yet visible")
    cmap = plt.get_cmap("viridis")
    maxd = max(int(depth.max()), 1)
    vis = keep & (np.arange(len(depth)) != root)
    ax.scatter(pos[vis, 0], pos[vis, 1], s=14, c=depth[vis], cmap=cmap, vmin=1, vmax=maxd, zorder=3,
               label="visible to the model")
    ax.scatter([pos[root, 0]], [pos[root, 1]], s=140, c="#d62728", marker="*", zorder=4, label="news (root)")
    ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="lower left", fontsize=8, frameon=False)
    return fig


# ---------- sidebar ----------
st.sidebar.title("🕸️ News Propagation GNN")
page = st.sidebar.radio("Go to", ["Dashboard", "Prediction demo", "Held-out sample check", "About & limitations"])
st.sidebar.markdown("---")
st.sidebar.caption("Batch 11 · AI & ML · MCET Pollachi\n\n23EAP501 Reverse Engineering Project")
st.sidebar.warning("Outputs are **model predictions**, not verified facts.")

# ================= DASHBOARD =================
if page == "Dashboard":
    st.title("Early Detection of Deceptive News Propagation using GNNs")
    st.caption("Dataset: UPFD – %s propagation graphs (derived from FakeNewsNet; labels from PolitiFact fact-checkers)." % cfg["dataset"])
    ss = metrics["split_sizes"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Train graphs", ss["train"]); c2.metric("Validation graphs", ss["val"]); c3.metric("Held-out test graphs", ss["test"])

    st.subheader("Test performance (mean ± std over 5 random seeds)")
    rows = []
    for name, r in metrics["test_mean_std"].items():
        rows.append({"Model": name, **{k.replace("_", " "): f"{v[0]:.3f} ± {v[1]:.3f}" for k, v in r.items()}})
    st.table(pd.DataFrame(rows).set_index("Model"))
    st.caption("Baseline = averages node features, ignores graph structure. GNN = GraphSAGE using the retweet graph. "
               "The test set has %d graphs, so a gap of a few points is modest evidence, not proof." % ss["test"])

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix (GNN, seed 0)")
        st.image(str(utils.ART / "confusion_matrix.png"))
    with right:
        st.subheader("Early detection")
        st.image(str(utils.ART / "early_detection.png"))

    st.subheader("Early-detection table (test set)")
    ed = []
    for k, r in metrics["early_detection"].items():
        ed.append({"Information available": k, "Nodes visible (%)": round(r["nodes_kept_percent"], 1),
                   "Baseline accuracy": round(r[utils.BASE_NAME]["accuracy"], 3),
                   "GNN accuracy": round(r[utils.GNN_NAME]["accuracy"], 3),
                   "GNN macro-F1": round(r[utils.GNN_NAME]["f1_macro"], 3)})
    st.dataframe(pd.DataFrame(ed), hide_index=True)
    st.caption("K = number of retweet levels (hops from the news node) the model is allowed to see. "
               "Numbers for the early-detection table come from a single trained model (seed 0).")

# ================= PREDICTION DEMO =================
elif page == "Prediction demo":
    st.title("Prediction demo on real held-out propagation graphs")
    st.write("Pick a real test graph that the model never saw in training. Then limit how much of the spread the model "
             "is allowed to see, to simulate **early** detection.")
    idx = st.selectbox("Held-out test graph", range(len(samples)),
                       format_func=lambda i: f"Sample {i + 1} ({samples[i]['x'].shape[0]} users/nodes)")
    s = samples[idx]
    depth = bfs_depth(s["edge_index"], s["x"].shape[0], s["root"])
    options = ["1", "2", "3", "4", "Full graph"]
    choice = st.select_slider("How many retweet levels (hops) can the model see?", options=options, value="Full graph")
    k = None if choice == "Full graph" else int(choice)
    x_k, ei_k, keep = truncate(s["x"], s["edge_index"], depth, k)

    try:
        probs = utils.predict_proba(model, x_k, ei_k)
    except Exception as e:
        st.error("Prediction failed. Copy this error for debugging:"); st.code(repr(e)); st.stop()
    pred = int(np.argmax(probs))

    a, b = st.columns([1, 1])
    with a:
        st.subheader("Model prediction")
        st.metric("Predicted class", cname(pred), help="A prediction by the trained model - NOT a verified fact.")
        for i, p in enumerate(probs):
            st.progress(float(p), text=f"{cname(i)}: {p * 100:.1f}%")
        if FAKE_IDX is not None:
            st.metric("Misinformation-risk score", f"{probs[FAKE_IDX] * 100:.0f}%",
                      help="Model's estimated probability of the 'fake' class given the part of the spread it saw.")
        else:
            st.info("Class meanings are not configured yet (see README step 'Confirm the label mapping'), "
                    "so only class probabilities are shown.")
        st.caption("A high score means the *propagation pattern* resembles patterns of fact-checked fake news in the "
                   "training data. It does not prove the news is false or a scam.")
        st.subheader("Dataset label (fact-checker annotation)")
        st.write(f"**{cname(s['y'])}** — shown only for comparison; the model never saw it.")
        st.write(f"Nodes visible to model: **{int(keep.sum())} / {len(keep)}**")
    with b:
        st.subheader("Propagation graph")
        st.pyplot(draw_graph(depth, s["edge_index"], keep, s["root"]))
        st.caption("Star = the news. Rings = retweet levels. Grey nodes are hidden from the model at this setting.")

# ================= HELD-OUT CHECK =================
elif page == "Held-out sample check":
    st.title("Held-out sample check")
    st.write("Runs the saved model on the 10 real test graphs bundled with this app (5 per class) as a "
             "**self-test** that the Colab model and this app work together.")
    if st.button("Run check"):
        rows, correct = [], 0
        for i, s in enumerate(samples):
            depth = bfs_depth(s["edge_index"], s["x"].shape[0], s["root"])
            p = utils.predict_proba(model, *truncate(s["x"], s["edge_index"], depth, None)[:2])
            pr = int(np.argmax(p)); correct += int(pr == s["y"])
            rows.append({"Sample": i + 1, "Nodes": s["x"].shape[0], "Dataset label": cname(s["y"]),
                         "Model prediction": cname(pr), "Confidence": f"{p[pr] * 100:.1f}%",
                         "Match": "✅" if pr == s["y"] else "❌"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.success(f"{correct}/{len(samples)} correct on these 10 graphs.")
        st.caption("Only 10 graphs - this is a functional test, NOT a performance measure. "
                   "The real evaluation is the 221-graph test result on the Dashboard.")

# ================= ABOUT =================
else:
    st.title("About & limitations")
    st.markdown("""
**What this system is.** A graph neural network (GraphSAGE) that reads a news item's retweet propagation graph and
predicts whether its spread resembles that of fact-checked fake news. It reverse-engineers the idea behind
propagation-based detectors (e.g., UPFD), going beyond visualisation tools such as Hoaxy, which show spread but do not classify it.

**What it is not.**
- **Not a scam detector.** No public dataset combines scam labels with real propagation graphs. We use fake/real labels
  (PolitiFact) as a proxy for deceptive news. Scam-specific data is future work.
- **Not a fact-checker.** Predictions are statistical and can be wrong; they must never be presented as verified facts.
- **No free-text input.** The model works on propagation graphs with precomputed node features, so it cannot score a typed headline.

**Early detection.** The dataset files do not expose timestamps, so "early" means *only the first K retweet levels*.
A retweet cannot appear before the post it retweets, so no future information is used.

**Evaluation.** Official train/validation/test splits of the dataset; best epoch chosen on validation only; the test set is
used once for reporting; 5 seeds. The test set is small (221 graphs), so differences of a few points should be read cautiously.
""")
