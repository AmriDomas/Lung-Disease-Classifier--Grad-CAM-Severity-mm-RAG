# Lung-Disease-Classifier--Grad-CAM-Severity-mm-RAG

## 📌 About & Notes

What this app does
 - Accepts Keras .h5 model (input shape: 224x224x3) and a chest X-ray image (.jpg/.png or DICOM).
 - Produces:
     - Prediction (classification).
     - Grad-CAM heatmap + overlay.
 - Estimates lesion area (mm²) from Grad-CAM heatmap.
     - Uses DICOM PixelSpacing for accurate mm conversion if available.
     - Otherwise uses assumed chest width input for calibration.
 - Simulates lesion growth under delayed handling (via user slider).
 - Builds a local RAG index from data/kb/ (PDF/TXT files) and retrieves supporting literature.
     - Optionally, uses an LLM to summarize retrieved passages.

## ⚠️ Important assumptions & cautions

   - Grad-CAM is explanatory, not segmentation. Area estimates are approximate.
   - For accurate mm² calculation:
      - Use DICOM with PixelSpacing.
      - For .jpg/.png, set assumed chest width (mm) manually.
   - Growth simulation parameters are heuristic → should be calibrated with longitudinal clinical data.
   - 🚨 This is not a clinical decision tool. Always confirm with clinicians and diagnostics.

## 🚀 Next steps (optional ideas)

   - Replace Grad-CAM proxy with a segmentation U-Net for precise opacity segmentation.
   - Calibrate severity thresholds and growth parameters using annotated clinical datasets.
   - Add authentication & logging for production deployment.

## 🛠️ How to Run
  1. Clone repo
     ```bash
     git clone https://github.com/AmriDomas/Lung-Disease-Classifier--Grad-CAM-Severity-mm-RAG.git
     cd Lung-Disease-Classifier--Grad-CAM-Severity-mm-RAG
     ```

  2. Install requirements
     ```bash
     pip install -r requirements.txt
     ```

 3.  Prepare knowledge base (optional RAG)
     Put your PDF/TXT medical references in:
     ```bash
     kb/
     ```

4. Run Streamlit app
   ```bash
   streamlit run app_lung.py
   ```

## 📥 Model

   - The app requires you to upload your trained Keras model (.h5).
   - Example: trained classifier with classes [Lung_Opacity, Normal, Viral Pneumonia].
   - You can host your model file on HuggingFace (for storage/backup), but in the app you upload manually via UI.

## 📂 Repo Structure

   ```bash
  ├── app_lung.py        # Main Streamlit app
  ├── requirements.txt   # Python dependencies
  ├── kb/                # Knowledge base (PDF/TXT for RAG)
  ├── utils/             # (Optional) helper functions
  └── README.md          # This file
  ```

## ✅ Example Workflow

  1. Start the app with streamlit run app_lung.py.
  2. Upload your trained Keras .h5 model.
  3. Upload chest X-ray (.jpg/.png) or DICOM (.dcm) image.
  4. View prediction, Grad-CAM heatmap, and lesion area estimation.
  5. Optionally explore RAG tab → search literature from data/kb/.

## 🔒 Disclaimer

This app is for research and educational purposes only.
It is not a medical diagnostic tool. Always consult clinicians and verified diagnostics for patient care.


