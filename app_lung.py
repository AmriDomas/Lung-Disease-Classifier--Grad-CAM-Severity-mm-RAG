# app.py
# Fixed Streamlit app: Upload Keras model (.h5), upload X-ray (.jpg/.png/.dcm), Grad-CAM, area->severity, simulate delay, RAG retrieval + LLM.
import tempfile
import os
import json
import hashlib
from pathlib import Path
from io import BytesIO

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import save_img as kimage_save_img
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess

# RAG deps
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import hf_hub_download

# DICOM optional
try:
    import pydicom
except Exception:
    pydicom = None

# Optional OpenAI LLM
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Utility: model load
# ----------------------------

@st.cache_resource(show_spinner=True)
def load_keras_model_from_file(path):
    return keras.models.load_model(path, compile=False)

# ----------------------------
# Grad-CAM
# ----------------------------
def _get_layer_recursive(model, name):
    # cari layer di top-level dulu
    try:
        return model.get_layer(name)
    except Exception:
        pass
    # cari di submodel (mis. Sequential yang membungkus base_model)
    for l in model.layers:
        if hasattr(l, "get_layer"):
            try:
                return l.get_layer(name)
            except Exception:
                continue
    return None

def make_gradcam_heatmap(img_array, model, pred_index=None, last_conv_name="block5_conv4"):
    # --- Sanity checks & coercion ---
    if img_array is None:
        raise ValueError("img_array is None. Pastikan x_proc sudah terisi dan dipreproses.")
    arr = img_array
    if isinstance(arr, tf.Tensor):
        arr = arr.numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)  # (H,W,C) -> (1,H,W,C)
    if arr.ndim != 4:
        raise ValueError(f"img_array harus 4D (1,H,W,3), dapat shape={arr.shape}")

    # pastikan input float32
    tf_img = tf.convert_to_tensor(arr, dtype=tf.float32)

    # --- Ambil layer konvolusi target ---
    target_layer = _get_layer_recursive(model, last_conv_name)
    if target_layer is None:
        raise ValueError(f"Layer '{last_conv_name}' tidak ditemukan di model.")

    # --- Model untuk output conv + pred ---
    grad_model = tf.keras.Model([model.inputs], [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        tape.watch(tf_img)  # jaga-jaga
        conv_outputs, preds = grad_model(tf_img, training=False)  # (1,Hc,Wc,C), (1,num_classes)

        if pred_index is None:
            # ambil kelas prediksi tertinggi
            pred_index = tf.argmax(preds[0])
        # loss = logit/prob kelas target
        loss = preds[:, pred_index]

    # --- Gradien w.r.t feature maps ---
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None. Cek nama layer conv & jalur komputasinya.")

    # Global average pooling gradien per channel -> bobot
    # conv_outputs: (1,Hc,Wc,C); grads sama shape
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Timbang feature map dengan bobot gradien
    conv_outputs = conv_outputs[0]  # (Hc,Wc,C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (Hc,Wc)

    # ReLU & normalisasi
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap)
    heatmap = heatmap / (denom + 1e-8)

    # Resize ke ukuran input
    H, W = arr.shape[1], arr.shape[2]
    heatmap = cv2.resize(heatmap.numpy().astype(np.float32), (W, H))

    return heatmap, int(pred_index.numpy()) if hasattr(pred_index, "numpy") else int(pred_index)


def overlay_heatmap_on_image(orig_uint8, heatmap, alpha=0.45):
    # orig_uint8: HxW or HxWx3 uint8
    hm_uint8 = np.uint8(255 * heatmap)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)  # BGR
    if orig_uint8.ndim == 2:
        orig_bgr = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2BGR)
    else:
        # Ensure proper BGR conversion
        if orig_uint8.shape[2] == 3:
            orig_bgr = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2BGR)
        else:
            orig_bgr = orig_uint8
    overlay = cv2.addWeighted(orig_bgr.astype(np.uint8), 1.0, hm_color.astype(np.uint8), alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb

# ----------------------------
# Area / Severity / Simulation helpers
# ----------------------------
def heatmap_to_mask(heatmap, threshold=0.3, min_area_px=30):
    mask = (heatmap >= threshold).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask, dtype=np.uint8)
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area_px:
            cv2.drawContours(clean, [c], -1, 1, -1)
    return clean

def estimate_pixel_spacing(image_width_px, assumed_chest_width_mm=350.0):
    mm_per_px = assumed_chest_width_mm / max(1, image_width_px)
    return (mm_per_px, mm_per_px)

def pixels_to_mm2(mask, pixel_spacing):
    py, px = pixel_spacing
    pixel_area_mm2 = py * px
    pixel_count = int(np.sum(mask))
    return pixel_count * pixel_area_mm2, pixel_count

def compute_simple_lung_mask(img_gray):
    # Heuristic lung mask; optional better with U-Net
    img = cv2.resize(img_gray, (img_gray.shape[1], img_gray.shape[0]))
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lung_mask = (th==0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_OPEN, kernel)
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return lung_mask
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)[:2]
    new = np.zeros_like(lung_mask)
    for c in contours:
        cv2.drawContours(new, [c], -1, 1, -1)
    return new

# severity mapping (example; calibrate in production)
def severity_from_area_mm2(area_mm2, lung_area_mm2=None):
    if area_mm2 < 1000:
        cat = "Mild"
    elif area_mm2 < 4000:
        cat = "Moderate"
    else:
        cat = "Severe"
    pct = None
    if lung_area_mm2 and lung_area_mm2 > 0:
        pct = 100.0 * area_mm2 / lung_area_mm2
    return {"category": cat, "area_mm2": float(area_mm2), "pct_of_lung": (None if pct is None else float(pct))}

# growth simulation params (tunable)
DEFAULT_GROWTH_PARAMS = {
    "Lung_Opacity": {"type":"linear","k":1.4},
    "Viral Pneumonia": {"type":"linear","k":1.8},
    "Normal": {"type":"linear","k":1.0}
}

def simulate_area_growth(area_mm2, disease_label, delay_factor):
    params = DEFAULT_GROWTH_PARAMS.get(disease_label, {"type":"linear","k":1.2})
    k = float(params.get("k", 1.2))
    t = float(np.clip(delay_factor, 0.0, 1.0))
    if params.get("type","linear") == "linear":
        area_new = area_mm2 * (1.0 + (k - 1.0) * t)
    else:
        area_new = area_mm2 * (k ** t)
    return float(area_new)

# simulate delay effect on heatmap (visual)
def simulate_delay_on_heatmap(heatmap, delay_factor=0.0, threshold=0.3, amplify=1.4):
    h = (heatmap >= threshold).astype(np.uint8)
    max_iter = 20
    iters = int(np.round(delay_factor * max_iter))
    if iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        dilated = cv2.dilate(h, kernel, iterations=iters)
    else:
        dilated = h
    sim = heatmap.copy()
    sim[dilated==1] = np.clip(sim[dilated==1] * amplify, 0, 1.0)
    sim = cv2.GaussianBlur(sim.astype(np.float32), (3,3), 0)
    sim = np.clip(sim, 0, 1.0)
    return sim, dilated

# ----------------------------
# RAG: build index & retriever
# ----------------------------
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_faiss_index_from_kb(kb_dir="kb/", index_dir="rag_index", max_chunk=800):
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(EMB_MODEL)
    docs_meta = []
    vectors = []
    
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        st.warning(f"Knowledge base directory {kb_dir} not found. Creating empty directory.")
        kb_path.mkdir(parents=True, exist_ok=True)
        return 0
    
    for p in kb_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf",".txt",".md"}:
            txt = ""
            if p.suffix.lower()==".pdf":
                try:
                    from pdfminer.high_level import extract_text
                    txt = extract_text(str(p))
                except Exception as e:
                    st.warning(f"pdf read error {p}: {e}")
                    continue
            else:
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    st.warning(f"text read error {p}: {e}")
                    continue
            
            paras = [pp.strip() for pp in txt.split("\n") if pp.strip()]
            cur = ""
            chunks = []
            for par in paras:
                if len(cur)+len(par) < max_chunk:
                    cur = (cur + "\n" + par).strip()
                else:
                    if cur: chunks.append(cur)
                    cur = par
            if cur: chunks.append(cur)
            if not chunks:
                continue
            
            try:
                emb = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
                for ch, vec in zip(chunks, emb):
                    rid = hashlib.md5((str(p)+ch[:50]).encode()).hexdigest()
                    docs_meta.append({"id": rid, "source": str(p), "text": ch})
                    vectors.append(vec.astype('float32'))
            except Exception as e:
                st.warning(f"Embedding error for {p}: {e}")
                continue
    
    if len(vectors)==0:
        st.warning("No KB docs found or processed successfully")
        return 0
    
    xb = np.vstack(vectors).astype('float32')
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "docs.jsonl"), "w", encoding="utf-8") as f:
        for m in docs_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return len(vectors)

class RAGRetriever:
    def __init__(self, index_dir="rag_index", emb_model=EMB_MODEL, k=3):
        self.k = k
        self.model = SentenceTransformer(emb_model)
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "docs.jsonl"), "r", encoding="utf-8") as f:
            self.docs = [json.loads(l) for l in f.read().splitlines() if l.strip()]
    
    def retrieve(self, query):
        qv = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(qv)
        D, I = self.index.search(qv.astype('float32'), self.k)
        results = []
        for j, i in enumerate(I[0]):
            if i < len(self.docs):  # Safety check
                m = self.docs[i].copy()
                m['score'] = float(D[0][j])
                results.append(m)
        return results

def build_rag_prompt_with_severity(label, prob, area_mm2, pct_lung, current_cat, simulated_area_mm2, simulated_cat, retrieved_docs):
    ctx = "\n\n".join([f"[{i+1}] {d['text']}\n(Source: {d['source']})" for i,d in enumerate(retrieved_docs)])
    prompt = f"""
You are a clinical assistant. Use ONLY the context (below) to analyze a chest X-ray finding.

Observation:
- Model prediction: {label} (confidence {prob:.2f})
- Measured lesion area: {area_mm2:.1f} mm^2
- Percent of lung affected (estimate): {('N/A' if pct_lung is None else f'{pct_lung:.1f}%')}
- Current severity: {current_cat}

Simulation:
- Projected lesion area after delay: {simulated_area_mm2:.1f} mm^2
- Projected severity after delay: {simulated_cat}

Task:
1) Using the context, explain what this finding may indicate and how lesion size correlates with severity.
2) Discuss plausible clinical implications of projected increase and urgency (outpatient vs inpatient).
3) Provide evidence-based next steps citing retrieved sources as [1],[2]...
4) Add a short caution: this is estimation based on heatmap proxy; recommend clinical validation.

Context:
{ctx}

Answer concisely in Indonesian with bullets and cite sources by [number].
"""
    return prompt

# ----------------------------
# UI: Streamlit app
# ----------------------------
st.set_page_config(layout="wide", page_title="Lung CNN + Grad-CAM + RAG + Severity")
st.title("Lung Disease Classifier — Grad-CAM + Severity (mm²) + RAG")

left, right = st.columns([2, 1])

with left:
    # ========================
    # API Key Input
    # ========================
    st.subheader("🔑 API Key Input")
    api_key_input = st.text_input("Masukkan OpenAI API Key:", type="password")

    if st.button("Apply API Key"):
        if api_key_input:
            st.session_state['openai_api_key'] = api_key_input
            st.success("✅ API Key berhasil diset")
        else:
            st.error("⚠️ API Key tidak boleh kosong")

    st.markdown("---")

    # ========================== UI ==========================
    st.header("1) Upload model & image")
    model_file = st.file_uploader("Upload Keras model (.h5) (optional)", type=["h5","keras"], help="If you don't upload, the app cannot predict.")
    uploaded_img = st.file_uploader("Upload image (jpg/png) or DICOM (.dcm)", type=["jpg","jpeg","png","dcm"])
    classnames_text = st.text_input("Class names (comma separated). Keep empty for default.", value="Lung_Opacity,Normal,Viral Pneumonia")
    class_names = [c.strip() for c in classnames_text.split(",") if c.strip()]

    # load model
    model_obj = None
    if model_file:
        tmp_path = "temp_model.h5"
        try:
            with open(tmp_path, "wb") as f:
                f.write(model_file.read())
            model_obj = load_keras_model_from_file(tmp_path)
            st.success("Model loaded.")
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            st.error(f"Failed loading model: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        st.info("Upload a trained Keras .h5 model (input 224x224x3, softmax output).")

    # process image if uploaded
    if uploaded_img:
        try:
            # handle DICOM
            if str(uploaded_img.name).lower().endswith(".dcm") and pydicom:
                ds = pydicom.dcmread(uploaded_img)
                arr = ds.pixel_array
                if arr.ndim == 2:
                    img_uint8 = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = arr.astype(np.uint8)
                pixel_spacing = None
                try:
                    if 'PixelSpacing' in ds:
                        ps = ds.PixelSpacing
                        pixel_spacing = (float(ps[0]), float(ps[1]))
                    elif 'ImagerPixelSpacing' in ds:
                        ps = ds.ImagerPixelSpacing
                        pixel_spacing = (float(ps[0]), float(ps[1]))
                except Exception:
                    pixel_spacing = None
            else:
                # regular image
                img_pil = Image.open(uploaded_img).convert("RGB")
                img_rgb = np.array(img_pil)
                pixel_spacing = None

            # display
            st.image(img_rgb, caption="Input (resized preview)", width=600)

            # prepare input for model
            img_resized = cv2.resize(img_rgb, (224,224))
            display_orig = img_resized.copy()
            x = np.expand_dims(img_resized.astype(np.float32), axis=0)
            x_proc = vgg_preprocess(x.copy())  # change if model expects diff preprocess

            if model_obj is not None:
                try:
                    preds = model_obj.predict(x_proc)
                    num_classes = preds.shape[-1]
                    if len(class_names) != num_classes:
                        st.warning(f"Provided class_names length ({len(class_names)}) != model output ({num_classes}). Using generic names.")
                        class_names = [f"class_{i}" for i in range(num_classes)]
                    pred_idx = int(np.argmax(preds[0]))
                    pred_prob = float(np.max(preds[0]))

                    st.subheader("Prediction")
                    d = {class_names[i]: float(preds[0][i]) for i in range(num_classes)}
                    st.json(d)

                    # Grad-CAM
                    st.subheader("Grad-CAM")

                    try:
                        # Pakai layer fixed "block5_conv4"
                        heatmap, top_idx = make_gradcam_heatmap(
                            x_proc, model_obj, pred_index=None, last_conv_name="block5_conv4"
                        )
                        st.success(f"Grad-CAM successful! Top predicted index={top_idx}")

                        # siapkan base image untuk overlay
                        if isinstance(x_proc, tf.Tensor):
                            img_vis = x_proc.numpy()[0]
                        else:
                            img_vis = np.asarray(x_proc)[0]
                        # img_vis biasanya [0..1], ubah ke [0..255]
                        img_vis_255 = (np.clip(img_vis, 0, 1) * 255).astype(np.uint8)

                        # buat colormap heatmap (JET)
                        heatmap_uint8 = np.uint8(255 * heatmap)
                        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
                        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)      # ke RGB

                        # overlay
                        alpha = 0.4
                        overlay = cv2.addWeighted(img_vis_255, 1.0, heatmap_color, alpha, 0)

                        # tampilkan
                        st.image(img_vis_255, caption="Input", width=200)
                        st.image(heatmap_color, caption="Heatmap", width=200)
                        st.image(overlay, caption="Overlay", width=200)

                    except Exception as e:
                        st.error(f"Grad-CAM failure: {e}")
                        heatmap = np.zeros((224,224), dtype=np.float32)
                        overlay = display_orig.copy()

                    # Simulate delay controls
                    st.write("Simulate delay / worsening:")
                    delay_slider = st.slider("Delay factor (0 = none … 1 = max)", 0.0, 1.0, 0.0, step=0.01)
                    threshold = st.slider("Threshold for lesion mask", 0.0, 1.0, 0.30, step=0.01)
                    amplify = st.slider("Amplify intensity inside dilated lesion", 1.0, 3.0, 1.5, step=0.1)

                    sim_heat, mask = simulate_delay_on_heatmap(heatmap, delay_factor=delay_slider, threshold=threshold, amplify=amplify)
                    overlay = overlay_heatmap_on_image(display_orig, sim_heat, alpha=0.55)

                    # area calculation
                    if pixel_spacing is None:
                        # fallback estimate; allow user to customize assumed chest width
                        assumed_width_mm = st.number_input("Assumed chest width (mm) for calibration (if DICOM not available)", value=350.0, step=10.0)
                        pixel_spacing = estimate_pixel_spacing(display_orig.shape[1], assumed_chest_width_mm=float(assumed_width_mm))
                    area_mm2, px_count = pixels_to_mm2(heatmap_to_mask(sim_heat, threshold=threshold), pixel_spacing)

                    # optional lung mask & percent
                    img_gray = cv2.cvtColor(display_orig, cv2.COLOR_RGB2GRAY)
                    lung_mask = compute_simple_lung_mask(img_gray)
                    lung_area_mm2, lung_px = pixels_to_mm2(lung_mask, pixel_spacing) if lung_mask is not None else (None, None)
                    pct_lung = None
                    if lung_area_mm2 and lung_area_mm2 > 0:
                        pct_lung = 100.0 * area_mm2 / lung_area_mm2

                    current_sev = severity_from_area_mm2(area_mm2, lung_area_mm2)
                    simulated_area = simulate_area_growth(area_mm2, class_names[pred_idx], delay_slider)
                    simulated_sev = severity_from_area_mm2(simulated_area, lung_area_mm2)

                    # display images and metrics
                    col_o, col_h, col_s = st.columns(3)
                    col_o.image(display_orig, caption="Original (224x224)", width=224)
                    col_h.image((np.uint8(255*sim_heat)), caption="Simulated Heatmap (0..255)", width=224)
                    col_s.image(overlay, caption=f"Overlay (delay={delay_slider:.2f})", width=224)

                    st.markdown("**Computed metrics**")
                    st.write(f"- Lesion area (mm²): **{area_mm2:.1f}**")
                    st.write(f"- Pixels lesion: {px_count}")
                    st.write(f"- Lung area (mm²) estimate: {lung_area_mm2:.1f}" if lung_area_mm2 else "- Lung area: N/A")
                    st.write(f"- Percent of lung affected: {pct_lung:.2f}%" if pct_lung is not None else "- Percent of lung: N/A")
                    st.write(f"- Current severity: **{current_sev['category']}**")
                    st.write(f"- Projected area after delay: **{simulated_area:.1f} mm²** → **{simulated_sev['category']}**")

                    # timeline plot area vs delay factor for small set of delay points
                    delays = np.linspace(0,1,21)
                    areas = [simulate_area_growth(area_mm2, class_names[pred_idx], d) for d in delays]
                    plt.figure(figsize=(6,3))
                    plt.plot(delays, areas, marker='o')
                    plt.axvline(delay_slider, color='red', linestyle='--')
                    plt.scatter([delay_slider], [simulated_area], color='red')
                    plt.xlabel("Delay factor (0..1)")
                    plt.ylabel("Projected lesion area (mm²)")
                    plt.title("Projected area vs delay")
                    st.pyplot(plt.gcf())
                    plt.close()

                    # Save preview
                    if st.button("Save results (images & metrics)"):
                        try:
                            out_dir = "pred-results"
                            os.makedirs(out_dir, exist_ok=True)
                            base = hashlib.md5((uploaded_img.name + str(pred_idx)).encode()).hexdigest()[:8]
                            
                            # Use PIL to save images instead of kimage_save_img
                            Image.fromarray(display_orig.astype(np.uint8)).save(os.path.join(out_dir, f"{base}_orig.png"))
                            Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(out_dir, f"{base}_overlay.png"))
                            
                            meta = {
                                "filename": uploaded_img.name,
                                "pred": class_names[pred_idx],
                                "prob": pred_prob,
                                "area_mm2": area_mm2,
                                "sim_area_mm2": simulated_area,
                                "severity": current_sev['category'],
                                "sim_severity": simulated_sev['category']
                            }
                            with open(os.path.join(out_dir, f"{base}_meta.json"), "w", encoding="utf-8") as f:
                                json.dump(meta, f, ensure_ascii=False, indent=2)
                            st.success(f"Saved to {out_dir}")
                        except Exception as e:
                            st.error(f"Save error: {e}")

                    # RAG retrieval and LLM prompt
                    st.header("RAG Explanation + LLM (optional)")
                    # Build index button
                    if st.button("Build FAISS index from data/kb (if not built)"):
                        try:
                            count = build_faiss_index_from_kb(kb_dir="kb/", index_dir="rag_index")
                            if count > 0:
                                st.success(f"Built index with {count} chunks.")
                            else:
                                st.warning("No documents found or processed.")
                        except Exception as e:
                            st.error(f"Index build failed: {e}")

                    if os.path.exists("rag_index/faiss.index"):
                        try:
                            retr = RAGRetriever(index_dir="rag_index", emb_model=EMB_MODEL, k=3)
                            query = f"{class_names[pred_idx]} chest imaging findings severity management"
                            retrieved = retr.retrieve(query)
                            st.markdown("Top retrieved context:")
                            for i, d in enumerate(retrieved, 1):
                                with st.expander(f"[{i}] {os.path.basename(d['source'])} (score={d['score']:.3f})"):
                                    st.write(d['text'][:1500])
                            
                            prompt = build_rag_prompt_with_severity(class_names[pred_idx], pred_prob, area_mm2, pct_lung, current_sev['category'], simulated_area, simulated_sev['category'], retrieved)
                            st.markdown("LLM Prompt (for traceability):")
                            st.code(prompt[:2000] + ("..." if len(prompt) > 2000 else ""), language="text")
                            
                            # Use session state for API key
                            if 'openai_api_key' in st.session_state and st.session_state['openai_api_key']:
                                if st.button("Call OpenAI to summarize (costs may apply)"):
                                    try:
                                        import openai
                                        client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
                                        response = client.chat.completions.create(
                                            model="gpt-4o-mini",
                                            messages=[{"role":"user","content":prompt}],
                                            temperature=0.0,
                                            max_tokens=500
                                        )
                                        out = response.choices[0].message.content
                                        st.markdown("**LLM Summary (based on retrieved context)**")
                                        st.write(out)
                                    except Exception as e:
                                        st.error(f"LLM call failed: {e}")
                            else:
                                st.info("Set OpenAI API Key above to enable LLM summarization.")
                        except Exception as e:
                            st.error(f"RAG error: {e}")
                    else:
                        st.info("Build FAISS index first to enable RAG retrieval.")
                
                except Exception as e:
                    st.error(f"Model prediction error: {e}")
            else:
                st.warning("Upload a model to run prediction.")
        
        except Exception as e:
            st.error(f"Image processing error: {e}")
    else:
        st.info("Upload an image to preview.")

with right:
    st.header("About & Notes")
    st.markdown("""
    **What this app does**
    - Accepts Keras .h5 model (input: 224x224x3) and an X-ray image (jpg/png or DICOM).
    - Produces prediction, Grad-CAM heatmap, overlay.
    - Estimates lesion area (mm²) from heatmap (requires DICOM PixelSpacing for accurate mm conversion; otherwise use assumed chest width).
    - Simulates lesion growth under delayed handling (user slider).
    - Builds local RAG index from `data/kb/` (PDF/TXT) and retrieves supporting literature; optional LLM summary.

    **Important assumptions & cautions**
    - Grad-CAM is explanatory *not* a segmentation model. Area estimates from Grad-CAM are approximate.
    - For accurate mm²: use DICOM with PixelSpacing. For JPEG/PNG, set 'assumed chest width' (mm) to calibrate.
    - Growth simulation parameters are heuristic; calibrate with longitudinal clinical data.
    - This is *not* a clinical decision tool. Always confirm with clinicians and diagnostics.

    **Next steps (optional)**
    - Replace Grad-CAM proxy with a segmentation U-Net trained to segment opacities for precise area.
    - Calibrate severity thresholds and growth parameters using annotated clinical data.
    - Add authentication & logging for production deployment.
    """)
    st.markdown("---")

    st.caption("Built for demo/portfolio. Use responsibly.")
