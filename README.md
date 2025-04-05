# Indian Monuments Visual Search

This project enables **visual search** of Indian monuments using natural language descriptions (e.g., "white marble dome", "red fort in daylight") using **Vision-Language Models (VLMs)** like **OpenAI's CLIP**.

It leverages AI to retrieve the most relevant images based on your text query.

## 🌟 Features

- 🔎 Search images using text (Text-to-Image retrieval)
- 🤖 Powered by CLIP (Contrastive Language-Image Pretraining)
- ⚡ Fast similarity search with FAISS
- 🌐 Streamlit-based web interface
- 🏛️ Indian Monument dataset 

## 📸 Example Queries

- `"white marble dome"`
- `"ancient stone temple"`
- `"palace with garden"`
- `"minarets and domes"`

## 🚀 Technologies Used

- OpenAI CLIP
- FAISS
- PyTorch
- Streamlit
- Python

## 📁 Dataset

Indian Monuments Image Dataset  
Source: [Kaggle - Indian Monuments Image Dataset](https://www.kaggle.com/datasets/danushkumarv/indian-monuments-image-dataset)

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run app.py
