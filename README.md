# GSoC 2026 — ArtExtract Evaluation Tasks
## HumanAI Umbrella Organization

This repository contains solutions for the ArtExtract GSoC 2025 evaluation tasks.

### Task 1: Convolutional-Recurrent Architectures
**Notebook**: `Task1_CNN_RNN_ArtClassifier.ipynb`

A CNN-RNN model for classifying art by **Style** (27 classes), **Artist** (129 classes), and **Genre** (11 classes) using the WikiArt dataset.

- **Architecture**: ResNet-50 backbone → Spatial patch extraction → Bidirectional LSTM → Attention → Multi-task classification heads
- **Dataset**: WikiArt via HuggingFace (streaming, memory-efficient)
- **Evaluation**: Accuracy, F1-Score, Confusion Matrices, Outlier Detection via embedding analysis

### Task 2: Painting Similarity
**Notebook**: `Task2_PaintingSimilarity.ipynb`

A similarity search model to find visually similar paintings (e.g., similar portraits, poses, compositions) using the National Gallery of Art open dataset.

- **Architecture**: Pretrained CNN feature extractor → Embedding projection → FAISS nearest-neighbor search
- **Dataset**: NGA Open Data (CSV metadata + IIIF thumbnail images)
- **Evaluation**: Precision@K, Mean Average Precision, t-SNE visualization

---

### Setup (Google Colab)
1. Open either notebook in Google Colab
2. All dependencies are installed automatically via `!pip install` cells
3. Set `DEMO_MODE = True` for quick testing, `False` for full training
4. Run all cells

### Setup (Local)
```bash
pip install -r requirements.txt
jupyter notebook
```

### Project Structure
```
humanAI artextract/
├── README.md
├── requirements.txt
├── utils.py
├── Task1_CNN_RNN_ArtClassifier.ipynb
└── Task2_PaintingSimilarity.ipynb
```

### Author
Nisha Goswami — GSoC 2026 Applicant

