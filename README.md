# Early Detection of Deceptive News Propagation using GNNs – Streamlit App
23EAP501 Reverse Engineering Project · Batch 11 · AI & ML · MCET Pollachi

## Folder layout (keep exactly like this)
```
scam_news_gnn_app/
├── app.py              <- the Streamlit app (run this)
├── utils.py            <- loads model/artifacts, predicts
├── graph_utils.py      <- early-stage truncation + graph drawing helpers
├── model_def.py        <- SAME model code used in Colab
├── requirements.txt
├── README.md
└── artifacts/          <- everything from Colab's artifacts.zip
    ├── gnn_weights.pt, baseline_weights.pt
    ├── config.json, metrics.json
    ├── confusion_matrix.png, early_detection.png
    ├── sample_graphs.pt, model_def.py
```

## Run on Windows with VS Code
1. Install **Python 3.10, 3.11 or 3.12** (tick "Add Python to PATH") and VS Code.
2. VS Code: **File > Open Folder…** and choose `scam_news_gnn_app`.
3. Open a terminal: **Terminal > New Terminal** (use *Command Prompt* via the dropdown, it avoids PowerShell permission issues).
4. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
5. Install packages (takes a few minutes, torch is large):
   ```
   pip install -r requirements.txt
   ```
6. Start the app:
   ```
   streamlit run app.py
   ```
   Your browser opens at http://localhost:8501.

## Common problems
- **"No matching distribution found for torch"**: your Python is too new/old. Install Python 3.11 and redo steps 4–5.
- **"Missing required files"**: copy ALL files from `artifacts.zip` into the `artifacts` folder.
- **Model fails to load / shape error**: send the message shown on screen. (The app was built for the config in `artifacts/config.json`: 300-dim input, hidden 64, GraphSAGE.)
- **PowerShell says scripts are disabled**: use Command Prompt, or run `Set-ExecutionPolicy -Scope Process Bypass` first.

## Confirm the label mapping (do this before the review)
The app shows `class_0` / `class_1` because we have not yet confirmed from the UPFD documentation which id means fake.
Check https://github.com/safe-graph/GNN-FakeNews, and once confirmed edit `artifacts/config.json`, e.g.:
```
"class_names": ["Real", "Fake"],
"fake_class_index": 1
```
(only if that is the confirmed mapping). The app then shows names and a "misinformation-risk score".

## What the app does and does not claim
- Predictions are **model outputs**, not verified facts.
- It is **not a scam detector**: no public dataset has scam labels plus real propagation graphs; fake/real fact-check labels are used as a proxy.
- "Early detection" = the model sees only the first K retweet levels (dataset files have no timestamps).
- All dashboard numbers come from the real Colab evaluation (`metrics.json`).
