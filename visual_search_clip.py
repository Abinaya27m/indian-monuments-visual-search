import os
import clip
import torch
from PIL import Image
import numpy as np
import faiss

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Dataset path
base_path = r"D:\Indian-monuments\images"

# Store features and paths
image_features = []
image_paths = []

# Load images from train and test
for split in ["train", "test"]:
    split_path = os.path.join(base_path, split)
    for monument in os.listdir(split_path):
        monument_path = os.path.join(split_path, monument)
        if not os.path.isdir(monument_path):
            continue
        for img_file in os.listdir(monument_path):
            img_path = os.path.join(monument_path, img_file)
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = model.encode_image(image)
                image_features.append(feature.cpu().numpy()[0])
                image_paths.append(img_path)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

# Convert to NumPy array
image_features_np = np.array(image_features).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(512)
index.add(image_features_np)

# Function for text search
def search_images_by_text(query, top_k=5):
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).cpu().numpy().astype("float32")
    D, I = index.search(text_features, top_k)
    results = [image_paths[i] for i in I[0]]
    return results

# Example search
if __name__ == "__main__":
    query = input("Enter a search query (e.g., 'white marble dome'): ")
    results = search_images_by_text(query, top_k=5)
    print("Top matching images:")
    for path in results:
        print(path)
