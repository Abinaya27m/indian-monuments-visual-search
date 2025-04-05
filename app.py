import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import clip
import torch
import streamlit as st
from PIL import Image
import numpy as np
import faiss

# -------------------------------
# 🎨 Page Config & Title
# -------------------------------
st.set_page_config(page_title="Indian Monuments Visual Search", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2F4F4F;'>🕌 Indian Monuments Visual Search 🔍</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Search Indian monument images using natural language descriptions 💬</p>", unsafe_allow_html=True)

# -------------------------------
# 🚀 Load CLIP Model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------------
# 📂 Dataset Path (Change if needed)
# -------------------------------
base_path = "./images"

# -------------------------------
# 🧠 Feature Extraction Function
# -------------------------------
@st.cache_data
def load_image_features():
    image_features = []
    image_paths = []
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    total = 0

    for split in ["train", "test"]:
        split_path = os.path.join(base_path, split)
        for monument in os.listdir(split_path):
            monument_path = os.path.join(split_path, monument)
            if os.path.isdir(monument_path):
                total += len(os.listdir(monument_path))

    progress = st.progress(0)
    processed = 0

    for split in ["train", "test"]:
        split_path = os.path.join(base_path, split)
        for monument in os.listdir(split_path):
            monument_path = os.path.join(split_path, monument)
            if not os.path.isdir(monument_path):
                continue
            for img_file in os.listdir(monument_path):
                img_path = os.path.join(monument_path, img_file)

                if not os.path.isfile(img_path) or not any(img_file.lower().endswith(ext) for ext in valid_exts):
                    continue
                try:
                    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feature = model.encode_image(image)
                    image_features.append(feature.cpu().numpy()[0])
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
                processed += 1
                progress.progress(min(processed / total, 1.0))

    return np.array(image_features).astype("float32"), image_paths


# -------------------------------
# 🏗️ Build FAISS Index
# -------------------------------
image_features_np, image_paths = load_image_features()
index = faiss.IndexFlatL2(512)
index.add(image_features_np)

# -------------------------------
# 📘 Sidebar Info
# -------------------------------
st.sidebar.title("📌 Project Info")
st.sidebar.markdown("""
**Project Title:** Visual Search using Vision-Language Models  
**Submitted by:** Abinaya M  
**Institute:** Sri Sai Ram Institute of Technology  
**Training:** Intel Unnati Industrial Training 2025  

**Technologies Used:**  
- OpenAI CLIP  
- FAISS  
- Streamlit  
""")

# -------------------------------
# 🔍 Text Input + Search
# -------------------------------
query = st.text_input("📝 Enter a description (e.g., 'white marble dome', 'red fort in daylight')")

if query:
    with st.spinner("Searching... please wait..."):
        try:
            text_tokens = clip.tokenize([query]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens).cpu().numpy().astype("float32")
            D, I = index.search(text_features, 5)
            results = [image_paths[i] for i in I[0]]

            # -------------------------------
            # 📸 Show Results
            # -------------------------------
            st.markdown("## 🔍 Top 5 Matching Images")
            cols = st.columns(5)
            for i, path in enumerate(results):
                with cols[i]:
                    st.image(path, caption=os.path.basename(path), use_container_width=True)

            
        except Exception as e:
            st.error(f"Something went wrong during search: {e}")
