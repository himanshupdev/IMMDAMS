import os
import io
import json
import time
import shutil
import datetime
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from PIL import Image as PILImage
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="IMMDAMS",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark industrial theme */
.stApp {
    background-color: #0f1117;
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Header banner */
.immdams-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    border-radius: 4px;
}
.immdams-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #58a6ff;
    margin: 0;
    letter-spacing: -0.5px;
}
.immdams-header p {
    color: #8b949e;
    margin: 0.3rem 0 0 0;
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #58a6ff;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

/* Pipeline badge */
.badge-doc {
    background: #1f3a5f;
    color: #58a6ff;
    padding: 2px 10px;
    border-radius: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid #2d5a8e;
}
.badge-photo {
    background: #1f3a2a;
    color: #3fb950;
    padding: 2px 10px;
    border-radius: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid #2d6e3e;
}

/* Result card */
.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.result-card .filename {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #e0e0e0;
    font-weight: 600;
    margin-bottom: 0.5rem;
    word-break: break-all;
}
.result-card .category {
    font-size: 1.1rem;
    font-weight: 600;
    color: #58a6ff;
}
.result-card .confidence {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #3fb950;
}
.result-card .tags {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 0.3rem;
}

/* Tag pills */
.tag-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    padding: 1px 8px;
    border-radius: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    margin: 2px;
}

/* JSON viewer */
.json-block {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #c9d1d9;
    overflow-x: auto;
    white-space: pre;
}

/* Folder tree */
.folder-tree {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.80rem;
    color: #8b949e;
}
.folder-tree .folder {
    color: #58a6ff;
    font-weight: 600;
}
.folder-tree .file {
    color: #c9d1d9;
}

/* Progress bar color */
.stProgress > div > div > div > div {
    background-color: #58a6ff;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 2px dashed #30363d;
    border-radius: 6px;
    background: #0d1117;
}

/* Buttons */
.stButton > button {
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    padding: 0.4rem 1.2rem;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #2d333b;
    border-color: #58a6ff;
    color: #58a6ff;
}

/* HSV gauge */
.hsv-bar-container {
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin: 4px 0 8px 0;
    overflow: hidden;
}
.hsv-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}

/* Divider */
hr {
    border-color: #30363d;
    margin: 1rem 0;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #8b949e;
    padding: 0.5rem 1.2rem;
    background: transparent;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("organized_output")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAT_COLORS = {
    "Portrait_People": "#e74c3c",
    "Animals_Nature": "#27ae60",
    "Vehicles_Street": "#2980b9",
    "Electronics_Workspace": "#8e44ad",
    "Interiors_Furniture": "#e67e22",
    "Food_Drink": "#c0392b",
    "Sports_Outdoor": "#16a085",
    "Accessories_Fashion": "#d35400",
    "Uncategorized": "#7f8c8d",
}


# ─────────────────────────────────────────────
#  LOAD CONFIG
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline_config():
    cfg_path = MODELS_DIR / "pipeline_config.json"
    if not cfg_path.exists():
        st.error(f"pipeline_config.json not found in {MODELS_DIR}/")
        st.stop()
    with open(cfg_path) as f:
        return json.load(f)


@st.cache_resource
def load_class_names():
    path = MODELS_DIR / "class_names.json"
    if not path.exists():
        st.error(f"class_names.json not found in {MODELS_DIR}/")
        st.stop()
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────
class DocumentClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_f, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────
#  LOAD MODELS (cached — loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_doc_model():
    class_names = load_class_names()

    # Prefer SWA model if it exists
    swa_path = MODELS_DIR / "resnet50_v2_swa.pth"
    best_path = MODELS_DIR / "resnet50_v2_best.pth"

    ckpt_path = swa_path if swa_path.exists() else best_path
    if not ckpt_path.exists():
        st.error(
            f"No model checkpoint found in {MODELS_DIR}/. Expected resnet50_v2_best.pth or resnet50_v2_swa.pth"
        )
        st.stop()

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)

    model = DocumentClassifier(num_classes=len(class_names), dropout=0.4)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    model_name = "SWA" if swa_path.exists() else "Best checkpoint"
    return model, class_names, model_name


@st.cache_resource
def load_detector():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights)
    detector.to(DEVICE).eval()
    return detector


# ─────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

tta_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.FiveCrop(224),
        transforms.Lambda(
            lambda crops: torch.stack(
                [
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(
                        transforms.ToTensor()(c)
                    )
                    for c in crops
                ]
            )
        ),
    ]
)


# ─────────────────────────────────────────────
#  PIPELINE FUNCTIONS
# ─────────────────────────────────────────────
def route_file(image_path: str, threshold: int = 50) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"pipeline": "document", "saturation": 0.0}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = round(float(hsv[:, :, 1].mean()), 2)
    return {
        "pipeline": "document" if sat < threshold else "photograph",
        "saturation": sat,
        "threshold": threshold,
    }


@torch.no_grad()
def classify_document(image_path, model, class_names, use_tta=True):
    img = PILImage.open(image_path).convert("RGB")
    if use_tta:
        tensor = tta_transform(img).unsqueeze(0)
        B, n, C, H, W = tensor.size()
        tensor = tensor.view(B * n, C, H, W).to(DEVICE)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).mean(0)
    else:
        tensor = val_transform(img).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(model(tensor), dim=1)[0]

    top3 = probs.topk(3).indices.tolist()
    return {
        "predicted_class": class_names[top3[0]],
        "confidence": round(probs[top3[0]].item(), 4),
        "top3": [
            {"class": class_names[i], "confidence": round(probs[i].item(), 4)}
            for i in top3
        ],
    }


@torch.no_grad()
def detect_objects(image_path, detector, cfg):
    coco_labels = cfg["coco_labels"]
    conf_thresh = cfg["det_conf_threshold"]
    max_det = cfg["det_max_objects"]

    img = PILImage.open(image_path).convert("RGB")
    tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
    out = detector(tensor)[0]

    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()

    mask = scores >= conf_thresh
    boxes = boxes[mask][:max_det]
    scores = scores[mask][:max_det]
    labels = labels[mask][:max_det]

    detections, label_names = [], []
    for box, score, idx in zip(boxes, scores, labels):
        name = coco_labels[idx]
        if name in ("N/A", "__background__"):
            continue
        label_names.append(name)
        detections.append(
            {
                "label": name,
                "confidence": round(float(score), 4),
                "bbox": [round(float(v), 1) for v in box],
            }
        )

    cat = assign_category(label_names, cfg)
    return {
        "image_size": list(img.size),
        "num_detections": len(detections),
        "detections": detections,
        "primary_category": cat["primary_category"],
        "all_categories": cat["all_categories"],
        "tags": cat["tags"],
    }


def assign_category(detected_labels, cfg):
    label_to_cat = {
        lbl: cat for cat, lbls in cfg["category_map"].items() for lbl in lbls
    }
    matched = {label_to_cat[l] for l in detected_labels if l in label_to_cat}
    if not matched:
        return {
            "primary_category": "Uncategorized",
            "all_categories": ["Uncategorized"],
            "tags": list(set(detected_labels)),
        }
    primary = next(
        (c for c in cfg["category_priority"] if c in matched), list(matched)[0]
    )
    return {
        "primary_category": primary,
        "all_categories": sorted(matched),
        "tags": list(set(detected_labels)),
    }


def build_metadata(image_path, router, model_result):
    if router["pipeline"] == "photograph":
        category = model_result["primary_category"]
        all_cats = model_result["all_categories"]
        tags = model_result["tags"]
        model_out = {"detections": model_result["detections"]}
    else:
        category = model_result["predicted_class"]
        all_cats = [category]
        tags = [category]
        model_out = {
            "predicted_class": model_result["predicted_class"],
            "confidence": model_result["confidence"],
            "top3": model_result["top3"],
        }
    folder_name = category.replace(" ", "_").replace("/", "_")
    return {
        "original_filename": Path(image_path).name,
        "processed_at": datetime.datetime.now().isoformat(),
        "pipeline": router["pipeline"],
        "hsv_saturation": router["saturation"],
        "assigned_category": category,
        "all_categories": all_cats,
        "assigned_folder": str(OUTPUT_DIR / folder_name),
        "tags": tags,
        "model_outputs": model_out,
    }


def store_asset(image_path, metadata):
    folder = Path(metadata["assigned_folder"])
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / Path(image_path).name
    shutil.copy2(image_path, dest)
    json_path = folder / (Path(image_path).stem + "_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return str(dest)


def draw_detections(image_path, result):
    """Draw bounding boxes and return matplotlib figure."""
    img = np.array(PILImage.open(image_path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(img)
    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        label_to_cat = {
            lbl: cat
            for cat, lbls in st.session_state.cfg["category_map"].items()
            for lbl in lbls
        }
        color = CAT_COLORS.get(
            label_to_cat.get(det["label"], "Uncategorized"), "#7f8c8d"
        )
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        )
        ax.text(
            x1,
            y1 - 5,
            f'{det["label"]} {det["confidence"]:.2f}',
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"),
        )
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def get_folder_tree():
    """Return folder tree as string."""
    if not OUTPUT_DIR.exists() or not any(OUTPUT_DIR.iterdir()):
        return "  (empty — no files processed yet)"
    lines = []
    for folder in sorted(OUTPUT_DIR.iterdir()):
        if folder.is_dir():
            imgs = [
                f
                for f in folder.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
            jsons = [f for f in folder.iterdir() if f.suffix == ".json"]
            lines.append(
                f"  📁 {folder.name}/  ({len(imgs)} file{'s' if len(imgs)!=1 else ''})"
            )
            for img in sorted(imgs)[:3]:
                lines.append(f"      {img.name}")
            if len(imgs) > 3:
                lines.append(f"      ... and {len(imgs)-3} more")
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  PROCESS ONE FILE
# ─────────────────────────────────────────────
def process_file(uploaded_file, doc_model, class_names, detector, cfg, use_tta):
    """Run full pipeline on one uploaded file. Returns (metadata, figure or None)."""
    # Save to temp file
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        # Step 1: Route
        router = route_file(tmp_path, threshold=cfg["hsv_sat_threshold"])

        # Step 2: Infer
        fig = None
        if router["pipeline"] == "document":
            model_result = classify_document(
                tmp_path, doc_model, class_names, use_tta=use_tta
            )
        else:
            model_result = detect_objects(tmp_path, detector, cfg)
            if model_result["num_detections"] > 0:
                fig = draw_detections(tmp_path, model_result)

        # Step 3: Metadata + store
        # Use original filename for output
        dest_tmp = str(Path(tempfile.gettempdir()) / uploaded_file.name)
        shutil.copy2(tmp_path, dest_tmp)
        metadata = build_metadata(dest_tmp, router, model_result)
        store_asset(dest_tmp, metadata)

        return metadata, fig

    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
#  INIT SESSION STATE
# ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []
if "total_files" not in st.session_state:
    st.session_state.total_files = 0
if "cfg" not in st.session_state:
    st.session_state.cfg = load_pipeline_config()


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
with st.spinner("Loading models..."):
    cfg = st.session_state.cfg
    doc_model, class_names, mname = load_doc_model()
    detector = load_detector()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style='font-family: IBM Plex Mono, monospace; font-size:1.1rem;
                font-weight:600; color:#58a6ff; margin-bottom:0.3rem;'>
        ⚙ SYSTEM
    </div>
    """,
        unsafe_allow_html=True,
    )

    device_label = "CUDA (GPU)" if DEVICE.type == "cuda" else "CPU"
    st.markdown(
        f"""
    <div class='result-card'>
        <div class='tags'>Device&nbsp;&nbsp;&nbsp; {device_label}</div>
        <div class='tags'>Doc model&nbsp; {mname}</div>
        <div class='tags'>Detector&nbsp;&nbsp; FasterRCNN V2</div>
        <div class='tags'>Classes&nbsp;&nbsp;&nbsp; {len(class_names)}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;
                font-weight:600;color:#8b949e;margin-bottom:0.5rem;'>
        OPTIONS
    </div>
    """,
        unsafe_allow_html=True,
    )

    use_tta = st.toggle("5-crop TTA (higher accuracy)", value=True)
    hsv_threshold = st.slider(
        "HSV saturation threshold",
        min_value=10,
        max_value=100,
        value=cfg["hsv_sat_threshold"],
        help="Files below this → Document pipeline. Above → Photo pipeline.",
    )
    cfg["hsv_sat_threshold"] = hsv_threshold

    st.markdown("---")
    st.markdown(
        """
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;
                font-weight:600;color:#8b949e;margin-bottom:0.5rem;'>
        OUTPUT
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.code(str(OUTPUT_DIR.absolute()), language=None)

    if st.button("🗑 Clear all results"):
        st.session_state.results = []
        st.session_state.total_files = 0
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(exist_ok=True)
        st.rerun()


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(
    """
<div class='immdams-header'>
    <h1>🗂 IMMDAMS</h1>
    <p>Intelligent Multi-Modal Digital Asset Management System</p>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  METRIC CARDS
# ─────────────────────────────────────────────
results = st.session_state.results
n_docs = sum(1 for r in results if r["pipeline"] == "document")
n_photos = sum(1 for r in results if r["pipeline"] == "photograph")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
    <div class='metric-card'>
        <div class='value'>{len(results)}</div>
        <div class='label'>Files processed</div>
    </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
    <div class='metric-card'>
        <div class='value'>{n_docs}</div>
        <div class='label'>Documents</div>
    </div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
    <div class='metric-card'>
        <div class='value'>{n_photos}</div>
        <div class='label'>Photographs</div>
    </div>""",
        unsafe_allow_html=True,
    )
with c4:
    cats = len({r["assigned_category"] for r in results})
    st.markdown(
        f"""
    <div class='metric-card'>
        <div class='value'>{cats}</div>
        <div class='label'>Categories used</div>
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_upload, tab_results, tab_output, tab_about = st.tabs(
    [
        "  📤 Upload  ",
        "  📋 Results  ",
        "  🗂 Output  ",
        "  ℹ About  ",
    ]
)


# ════════════════════════════════════════════
#  TAB 1 — UPLOAD
# ════════════════════════════════════════════
with tab_upload:
    st.markdown("### Drop files to process")
    st.caption("Supported: JPG, JPEG, PNG  ·  Single or batch upload")

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")

        if st.button("▶ Run Pipeline", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            new_results = []

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.82rem;"
                    f"color:#8b949e;'>Processing {i+1}/{len(uploaded_files)}: "
                    f"{uploaded_file.name}</div>",
                    unsafe_allow_html=True,
                )
                try:
                    metadata, fig = process_file(
                        uploaded_file, doc_model, class_names, detector, cfg, use_tta
                    )
                    new_results.append({**metadata, "_fig": fig})
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            st.session_state.results.extend(new_results)
            st.session_state.total_files += len(new_results)
            status_text.empty()

            st.success(
                f"✅ {len(new_results)} file(s) processed. View results in the **Results** tab."
            )
            st.rerun()


# ════════════════════════════════════════════
#  TAB 2 — RESULTS
# ════════════════════════════════════════════
with tab_results:
    if not results:
        st.markdown(
            """
        <div style='text-align:center;padding:3rem;color:#8b949e;
                    font-family:IBM Plex Mono,monospace;font-size:0.85rem;'>
            No results yet — upload files in the Upload tab.
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        for r in reversed(results):
            with st.expander(f"📄 {r['original_filename']}", expanded=False):
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    # Pipeline badge
                    badge_class = (
                        "badge-doc" if r["pipeline"] == "document" else "badge-photo"
                    )
                    badge_label = (
                        "DOCUMENT" if r["pipeline"] == "document" else "PHOTOGRAPH"
                    )
                    st.markdown(
                        f"<span class='{badge_class}'>{badge_label}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True
                    )

                    # Category + confidence
                    st.markdown(
                        f"""
                    <div style='font-size:1.2rem;font-weight:600;color:#58a6ff;'>
                        {r['assigned_category'].replace('_', ' ')}
                    </div>""",
                        unsafe_allow_html=True,
                    )

                    if r["pipeline"] == "document":
                        conf = r["model_outputs"].get("confidence", 0)
                        st.markdown(
                            f"""
                        <div style='font-family:IBM Plex Mono,monospace;
                                    font-size:0.82rem;color:#3fb950;margin-top:2px;'>
                            {conf*100:.1f}% confidence
                        </div>""",
                            unsafe_allow_html=True,
                        )

                        # Top-3
                        st.markdown(
                            "<div style='margin-top:0.7rem;margin-bottom:0.3rem;"
                            "font-size:0.78rem;color:#8b949e;'>Top-3 predictions</div>",
                            unsafe_allow_html=True,
                        )
                        for p in r["model_outputs"].get("top3", []):
                            pct = p["confidence"] * 100
                            bar_color = (
                                "#58a6ff"
                                if p == r["model_outputs"]["top3"][0]
                                else "#30363d"
                            )
                            st.markdown(
                                f"""
                            <div style='font-family:IBM Plex Mono,monospace;
                                        font-size:0.78rem;color:#c9d1d9;margin-bottom:4px;'>
                                {p['class']:<25} {pct:5.1f}%
                                <div class='hsv-bar-container'>
                                    <div class='hsv-bar-fill'
                                         style='width:{min(pct,100)}%;
                                                background:{bar_color};'></div>
                                </div>
                            </div>""",
                                unsafe_allow_html=True,
                            )

                    else:
                        # Tags
                        tags_html = " ".join(
                            f"<span class='tag-pill'>{t}</span>"
                            for t in r.get("tags", [])
                        )
                        st.markdown(
                            f"<div style='margin-top:0.5rem;'>{tags_html}</div>",
                            unsafe_allow_html=True,
                        )

                    # HSV saturation
                    sat = r.get("hsv_saturation", 0)
                    sat_pct = min(sat / 255 * 100, 100)
                    sat_color = (
                        "#3fb950" if r["pipeline"] == "photograph" else "#58a6ff"
                    )
                    st.markdown(
                        f"""
                    <div style='margin-top:0.8rem;font-family:IBM Plex Mono,monospace;
                                font-size:0.75rem;color:#8b949e;'>
                        HSV saturation: {sat:.1f}
                        <div class='hsv-bar-container'>
                            <div class='hsv-bar-fill'
                                 style='width:{sat_pct:.1f}%;background:{sat_color};'></div>
                        </div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

                    # Stored path
                    st.markdown(
                        f"""
                    <div style='margin-top:0.5rem;font-family:IBM Plex Mono,monospace;
                                font-size:0.72rem;color:#8b949e;word-break:break-all;'>
                        → {r['assigned_folder']}
                    </div>""",
                        unsafe_allow_html=True,
                    )

                with col_right:
                    # Bounding box figure for photos
                    if r["pipeline"] == "photograph" and r.get("_fig"):
                        st.pyplot(r["_fig"], use_container_width=True)
                        plt.close(r["_fig"])
                    elif r["pipeline"] == "photograph":
                        n_det = r["model_outputs"].get("detections", [])
                        st.markdown(
                            f"""
                        <div style='text-align:center;padding:2rem;color:#8b949e;
                                    font-family:IBM Plex Mono,monospace;font-size:0.82rem;'>
                            {len(n_det)} detection(s) below confidence threshold
                        </div>""",
                            unsafe_allow_html=True,
                        )

                    # JSON metadata
                    st.markdown(
                        "<div style='margin-top:0.5rem;font-family:IBM Plex Mono,"
                        "monospace;font-size:0.75rem;color:#8b949e;'>Metadata JSON</div>",
                        unsafe_allow_html=True,
                    )
                    display_meta = {k: v for k, v in r.items() if k != "_fig"}
                    st.json(display_meta, expanded=False)


# ════════════════════════════════════════════
#  TAB 3 — OUTPUT FOLDER TREE
# ════════════════════════════════════════════
with tab_output:
    st.markdown("### Organized folder structure")
    st.caption(f"Root: `{OUTPUT_DIR.absolute()}`")

    tree = get_folder_tree()
    st.markdown(
        f"""
    <div class='folder-tree'>
{tree}
    </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # Category breakdown chart
    if results:
        st.markdown("### Category distribution")
        from collections import Counter

        cat_counts = Counter(r["assigned_category"] for r in results)
        cats_sorted = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        fig2.patch.set_facecolor("#161b22")
        ax2.set_facecolor("#161b22")
        labels_ = [c[0].replace("_", " ") for c in cats_sorted]
        values_ = [c[1] for c in cats_sorted]
        colors_ = [CAT_COLORS.get(c[0], "#7f8c8d") for c in cats_sorted]
        bars = ax2.barh(labels_, values_, color=colors_, height=0.5)
        ax2.set_xlabel("Files", color="#8b949e", fontsize=9)
        ax2.tick_params(colors="#c9d1d9", labelsize=8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for spine in ["bottom", "left"]:
            ax2.spines[spine].set_color("#30363d")
        ax2.xaxis.label.set_color("#8b949e")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ════════════════════════════════════════════
#  TAB 4 — ABOUT
# ════════════════════════════════════════════
with tab_about:
    st.markdown(
        """
### Intelligent Multi-Modal Digital Asset Management System

IMMDAMS automatically classifies, tags, and organizes unstructured image and document
files by routing them through two specialized AI pipelines.

---

**Pipeline 1 — Document Classifier**
- Model: ResNet50 (ImageNet V2 weights)
- Head: 2048 → 512 → 16
- Training: 8,000 images, 16 classes, CutMix + OHEM + SWA
- Accuracy: **74.62%** (with 5-crop TTA at inference)
- Classes: advertisement, budget, email, file_folder, form, handwritten, invoice,
  letter, memo, news_article, presentation, questionnaire, resume, scientific,
  scientific_report, specification

**Pipeline 2 — Object Detector**
- Model: Faster R-CNN ResNet50-FPN V2 (COCO pretrained)
- COCO box mAP: **46.7**
- 80 COCO classes → 8 folder categories
- No fine-tuning required

**Router**
- OpenCV HSV saturation heuristic
- Saturation < threshold → Document pipeline
- Saturation ≥ threshold → Photo pipeline
- Threshold adjustable in sidebar

---

**Tech stack:** Python · PyTorch · OpenCV · Streamlit · torchvision

**Built by:** Himanshu Pandey
"""
    )
