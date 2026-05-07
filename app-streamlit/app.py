# app.py — Arabic Dialect Identification Dashboard
# =============================================================================
import os, sys, warnings, importlib.util, html
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import base64


# =============================================================================
# PATHS
# =============================================================================
BASE_PATH     = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_PATH, "..", "artifacts"))
OUTPUT_DIR    = os.path.join(ARTIFACTS_DIR, "output")
MODULE_PATH   = os.path.abspath(os.path.join(BASE_PATH, "..", "features.py"))
CSS_PATH      = os.path.join(BASE_PATH, "style.css")   # ← satu folder dgn app.py

# =============================================================================
# CONSTANTS
# =============================================================================
LOGO_WEBP = os.path.join(BASE_PATH, "adi.webp")
LOGO_PNG  = os.path.join(BASE_PATH, "adi-32.png")
LOGO_ICO  = os.path.join(BASE_PATH, "adi.ico")

ALGORITHMS = ["lightgbm", "xgboost"]
SCENARIOS  = ["original", "smote", "knsmote", "dbscansmote", "balanced_weighted"]
ALGO_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}
SCENARIO_LABELS = {
    "original": "Original",
    "smote": "SMOTE",
    "knsmote": "ASTRA-SMOTE",
    "dbscansmote": "SMOTE-RADIANT",
    "balanced_weighted": "Balanced Weighting",
}

SCENARIO_META = {
    "original": {
        "icon": '<i class="fa-solid fa-chart-column"></i>',
        "class": "adi-sc-or",
        "desc": "Original imbalanced distribution",
    },

    "smote": {
        "icon": '<i class="fa-solid fa-seedling"></i>',
        "class": "adi-sc-sm",
        "desc": "Standard synthetic oversampling",
    },

    "knsmote": {
        "icon": '<i class="fa-solid fa-circle-nodes"></i>',
        "class": "adi-sc-bl",
        "desc": "Cluster-guided safe oversampling",
    },

    "dbscansmote": {
        "icon": '<i class="fa-solid fa-braille"></i>',
        "class": "adi-sc-te",
        "desc": "Density-filtered oversampling",
    },

    "balanced_weighted": {
        "icon": '<i class="fa-solid fa-scale-balanced"></i>',
        "class": "adi-sc-pu",
        "desc": "Cost-sensitive learning",
    },
}

DIALECT_FLAGS = {
    "Syria": "🇸🇾", "Lebanon": "🇱🇧", "Palestine": "🇵🇸", "Jordan": "🇯🇴",
}

MODEL_DISPLAY_NAMES = {
    "XGBClassifier": "XGBoost",
    "LGBMClassifier": "LightGBM",
}

DEBUG_FIELDS = [
    "Cleaning",
    "Normalization (Light)",
    "Normalization (Aggressive)",
    "Char Stream",
]

TOTAL_MODELS = len(ALGORITHMS) * len(SCENARIOS)
TOTAL_DIALECTS = len(DIALECT_FLAGS)
TOTAL_ALGOS = len(ALGORITHMS)

DIALECT_COLORS = ["adi-fill-0", "adi-fill-1", "adi-fill-2", "adi-fill-3"]
SAMPLE_TEXT = "@__s24_ الله عليكي شو مثير"

# =============================================================================
# CONFIG  (MUST be first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="Arabic Dialect Identification",
    page_icon=LOGO_PNG,
    layout="wide",
)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

st.markdown("""
<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
""", unsafe_allow_html=True)

# =============================================================================
# INJECT CSS
# =============================================================================
def _load_css(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css(CSS_PATH)

def preserve_visual_order(text: str) -> str:
    return "\u202A" + text + "\u202C"


def safe_visual_text(text: str) -> str:
    text = html.escape(text)
    return preserve_visual_order(text)


def render_spaces(text: str) -> str:
    return text.replace(
        " ",
        '<span class="ws"></span>'
    )

def base64_logo(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# =============================================================================
# LOAD FEATURE MODULE
# =============================================================================
_spec = importlib.util.spec_from_file_location("features", MODULE_PATH)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["features"] = _mod
_spec.loader.exec_module(_mod)
from features import *  # noqa: F401,F403

# =============================================================================
# MODEL REGISTRY
# =============================================================================
def build_model_registry() -> dict:
    reg = {}
    for algo in ALGORITHMS:
        for sc in SCENARIOS:
            name = f"{ALGO_LABELS[algo]} + {SCENARIO_LABELS[sc]}"
            reg[name] = {"folder": f"{algo}_{sc}", "file": f"model_{algo}_{sc}.pkl",
                         "algorithm": algo, "scenario": sc}
    return reg

MODEL_REGISTRY = build_model_registry()

# =============================================================================
# ARTIFACT LOADING
# =============================================================================
@st.cache_resource
def load_artifacts(name: str):
    cfg  = MODEL_REGISTRY[name]
    mdl  = joblib.load(os.path.join(ARTIFACTS_DIR, cfg["folder"], "model", cfg["file"]))
    feat = joblib.load(os.path.join(OUTPUT_DIR, "shami_feature_extractor.pkl"))
    le   = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
    return mdl, feat, le

def best_iter(model) -> int:
    if hasattr(model, "best_iteration"):
        return model.best_iteration

    if hasattr(model, "best_iteration_"):
        return model.best_iteration_

    return 0

def predict_proba(model, X, bi: int) -> np.ndarray:
    if "XGB" in type(model).__name__:
        if bi > 0:
            return model.predict_proba(
                X,
                iteration_range=(0, bi + 1)
            )[0]

        return model.predict_proba(X)[0]

    if bi > 0:
        return model.predict_proba(
            X,
            num_iteration=bi
        )[0]

    return model.predict_proba(X)[0]

# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess(text):
    return clean_noise(text), clean_char(text)  

def preprocess_debug(text):
    cw  = clean_noise(text)                     
    nl  = normalize_light(cw)                   
    tks = TOK_AR.findall(nl)                    
    return {
        "Cleaning":                   cw,
        "Normalization (Light)":      nl,
        "Normalization (Aggressive)": normalize_aggr(cw),   
        "Char Stream":                clean_char(text),      
        "Tokenization":               tks,
        "Stopword Removed": remove_stopwords(text),
        "_cw": cw, "_cc": clean_char(text),  
    }

def remove_stopwords(text: str) -> str:
    cw  = clean_noise(text)          
    nl  = normalize_light(cw)        
    tks = TOK_AR.findall(nl)         

    filtered = [t for t in tks if t not in AR_STOP]  

    return " ".join(filtered)

# =============================================================================
# PREDICTION
# =============================================================================
def run_prediction(text, model, feat, le, bi, debug=False):

    if debug:
        info    = preprocess_debug(text)
        cw, cc  = info.pop("_cw"), info.pop("_cc")
    else:
        cw, cc  = preprocess(text)
        info    = None

    # =========================================
    # INVALID INPUT GUARD
    # =========================================
    if not cw.strip() or not cc.strip():
        return {
            "is_valid": False,
            "prediction": None,
            "probabilities": np.array([]),
            "ranking": pd.DataFrame(),
            "clean_word": cw,
            "clean_char": cc,
            "debug_info": info,
        }

    # =========================================
    # NORMAL PIPELINE
    # =========================================
    X = feat.transform(pd.DataFrame({
        "text": [text],
        "clean_word": [cw],
        "clean_char": [cc]
    }))

    proba = predict_proba(model, X, bi)

    pred = le.classes_[np.argmax(proba)]

    rdf = (
        pd.DataFrame({
            "Dialect": le.classes_,
            "Probability": proba
        })
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "is_valid": True,
        "prediction": pred,
        "probabilities": proba,
        "ranking": rdf,
        "clean_word": cw,
        "clean_char": cc,
        "debug_info": info,
    }

# =============================================================================
# HTML HELPERS
# =============================================================================
def _h(html: str):
    st.markdown(html, unsafe_allow_html=True)

def html_result_card(pred: str, proba: np.ndarray, is_valid: bool = True) -> str:

    # =========================================
    # INVALID INPUT
    # =========================================
    if not is_valid:
        return """
        <div class="adi-result">
          <div class="adi-result-flag">⚠️</div>
          <div>
            <div class="adi-result-dialect">
                Dialect Not Detected
            </div>
            <div class="adi-result-sub">
                Input does not contain sufficient Arabic dialectal content
            </div>
          </div>
          <div class="adi-conf-badge">N/A</div>
        </div>
        """

    # =========================================
    # NORMAL RESULT
    # =========================================
    flag = DIALECT_FLAGS.get(pred, "🌍")
    conf = f"{np.max(proba)*100:.2f}%"

    return f"""
    <div class="adi-result">
      <div class="adi-result-flag">{flag}</div>
      <div>
        <div class="adi-result-dialect">{pred}</div>
        <div class="adi-result-sub">Top predicted dialect</div>
      </div>
      <div class="adi-conf-badge">{conf}</div>
    </div>
    """

def html_prob_bars(rdf: pd.DataFrame, classes) -> str:
    color_map = {cls: DIALECT_COLORS[i] for i, cls in enumerate(classes)}
    rows = ""
    for _, row in rdf.iterrows():
        d    = row["Dialect"]
        pct  = row["Probability"] * 100
        flag = DIALECT_FLAGS.get(d, "🌍")
        rows += f"""
    <div class="adi-bar-row">
      <div class="adi-bar-lbl">{flag} {d}</div>
      <div class="adi-bar-track">
        <div class="adi-bar-fill {color_map.get(d,'adi-fill-0')}" style="width:{pct:.1f}%"></div>
      </div>
      <div class="adi-bar-pct">{pct:.1f}%</div>
    </div>"""
    return f'<div class="adi-bars">{rows}</div>'

def html_info_grid(items: list, rtl: bool = False) -> str:
    val_class = "adi-info-val-rtl" if rtl else "adi-info-val"

    cells = "".join(
        f'<div class="adi-info-pill">'
        f'<div class="adi-info-lbl">{lbl}</div>'
        f'<div class="{val_class}"><span>{val}</span></div>'
        f'</div>'
        for lbl, val in items
    )

    return f'<div class="adi-info-grid">{cells}</div>'

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(f"""
        <div class="adi-logo-wrap">
            <img src="data:image/webp;base64,{base64_logo(LOGO_WEBP)}"
                class="adi-logo-img">
            <div>
                <div class="adi-logo-name">Arabic Dialect ID</div>
                <div class="adi-logo-sub">Imbalance-Aware NLP</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    _h('<div class="adi-section">Configuration</div>')
    selected_model = st.selectbox(
        "model", list(MODEL_REGISTRY.keys()), label_visibility="collapsed"
    )
    debug_mode = st.checkbox("Enable Debug Mode", value=False)

    model, feat, le = load_artifacts(selected_model)
    bi              = best_iter(model)

    mtype = MODEL_DISPLAY_NAMES.get(
        type(model).__name__,
        type(model).__name__
    )

    _h('<div class="adi-section">Experiment Configuration</div>')
    _h(html_info_grid([
        ("Algorithm",      mtype),
        ("Optimal Iteration", str(bi)),
        ("Scenario", SCENARIO_LABELS[MODEL_REGISTRY[selected_model]["scenario"]]),
        ("Classes",        str(len(le.classes_))),
    ]))

    _h('<div class="adi-section">Levantine Dialects</div>')
    chips = "".join(
        f'<div class="adi-sb-chip"><span>{DIALECT_FLAGS.get(c,"🌍")}</span>{c}</div>'
        for c in le.classes_
    )
    _h(f'<div class="adi-sb-chips">{chips}</div>')

    _h('<div class="adi-section">Imbalance Mitigation Strategies</div>')

    scenario_cards = ""

    for key in SCENARIOS:
        meta = SCENARIO_META[key]

        scenario_cards += f"""
        <div class="adi-sc-card {meta['class']}">
        <div class="adi-sc-icon">{meta['icon']}</div>
        <div class="adi-sc-title">{SCENARIO_LABELS[key]}</div>
        <div class="adi-sc-desc">{meta['desc']}</div>
        </div>
        """

    _h(f"""
        <div class="adi-sc-grid">
        {scenario_cards}
        </div>
        """)

# =============================================================================
# MAIN — HERO
# =============================================================================
chips_html = "".join(
    f'<div class="adi-chip"><span>{flag}</span>{name}</div>'
    for name, flag in DIALECT_FLAGS.items()
)
_h(f"""
<div class="adi-hero">
  <div class="adi-eyebrow">
    <div class="adi-eyebrow-dot"></div>
    Research-Grade Arabic NLP
  </div>
  
  <h1>
    <em>Arabic</em> Dialect<br>
    Identification
  </h1>
  <div class="adi-desc">
    Interactive demonstration of a feature-engineered Arabic dialect
    identification pipeline for Levantine dialects using LightGBM,
    XGBoost, and density-guided imbalance mitigation strategies.
  </div>
  <div class="adi-chips">{chips_html}</div>
  <div class="adi-stat-row">
    <div class="adi-stat">
      <div class="adi-stat-num">{TOTAL_MODELS}</div>
      <div class="adi-stat-label">Experimental Setups</div>
    </div>
    <div class="adi-stat">
      <div class="adi-stat-num">{TOTAL_DIALECTS}</div>
      <div class="adi-stat-label">Target Dialects</div>
    </div>
    <div class="adi-stat">
      <div class="adi-stat-num">{TOTAL_ALGOS}</div>
      <div class="adi-stat-label">Boosting Models</div>
    </div>
  </div>
</div>""")

# =============================================================================
# SINGLE PREDICTION
# =============================================================================
_h('<div class="adi-section">Interactive Inference</div>')

text_input = st.text_area(
    "Input Arabic text:", value=SAMPLE_TEXT, height=110,
)

if st.button(
    "Run Inference",
    type="primary",
    icon=":material/neurology:"
):
    if not text_input.strip():
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Running inference…"):
            res = run_prediction(text_input, model, feat, le, bi, debug=debug_mode)

    _h(html_result_card(
        res["prediction"],
        res["probabilities"],
        res["is_valid"]
    ))

    # =========================================
    # ONLY SHOW DETAILS FOR VALID INPUT
    # =========================================
    if res["is_valid"]:

        _h('<div class="adi-section">Class Probability Distribution</div>')

        _h(
            '<div class="adi-card">'
            + html_prob_bars(res["ranking"], le.classes_)
            + '</div>'
        )

        _h('<div class="adi-section">Preprocessing Pipeline Output</div>')

        _h(html_info_grid([
            (
                "Stopword Removed",
                render_spaces(
                    safe_visual_text(
                        remove_stopwords(text_input)
                    )
                )
            ),
            (
                "Clean Char",
                render_spaces(
                    safe_visual_text(res["clean_char"])
                )
            )
        ], rtl=True))

        if debug_mode and res["debug_info"]:

            _h('<div class="adi-section">Pipeline Diagnostics</div>')

            for lbl in DEBUG_FIELDS:

                st.markdown(
                    f'<div class="debug-label">{lbl}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="rtl-debug">
                        {render_spaces(
                            safe_visual_text(
                                res["debug_info"][lbl]
                            )
                        )}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =============================================================================
# BATCH PREDICTION
# =============================================================================
st.markdown("---")
_h('<div class="adi-section">Batch Inference</div>')

batch_input = st.text_area("Input multiple Arabic texts (1 line = 1 text):", height=160)

if st.button(
    "Run Batch Inference",
    icon=":material/layers:"
):
    lines = [ln.strip() for ln in batch_input.split("\n") if ln.strip()]
    if not lines:
        st.warning("No input provided.")
    else:
        rows = []
        with st.spinner(f"Processing {len(lines)} texts…"):
            for txt in lines:
                r = run_prediction(txt, model, feat, le, bi)
                rows.append({
                    "Text": txt,
                    "Flag": (
                        DIALECT_FLAGS.get(r["prediction"], "⚠️")
                        if r["is_valid"]
                        else "⚠️"
                    ),

                    "Prediction": (
                        r["prediction"]
                        if r["is_valid"]
                        else "Dialect Not Detected"
                    ),

                    "Confidence": (
                        f"{np.max(r['probabilities'])*100:.2f}%"
                        if r["is_valid"]
                        else "N/A"
                    ),
                })
        df_out = pd.DataFrame(rows)
        st.success(f"✅ Successfully processed {len(df_out)} text samples.")
        st.dataframe(df_out, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            data=df_out.to_csv(index=False).encode("utf-8-sig"),
            file_name="prediction_results.csv",
            mime="text/csv",
        )