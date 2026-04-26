# 🚫 Toxic Comment Classification

A multi-label NLP project that detects and classifies toxic comments using deep learning. Built with PyTorch and HuggingFace Transformers, it identifies multiple categories of toxicity in user-generated text.

---

## 📌 Overview

Online platforms face significant challenges moderating harmful content at scale. This project tackles that problem by training a multi-label classifier capable of detecting whether a comment is toxic — and exactly *how* it's toxic — across six categories simultaneously.

The project follows the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset format.

---

## 🏷️ Toxicity Categories

| Label | Description |
|---|---|
| `toxic` | General toxicity |
| `severe_toxic` | Extremely harmful content |
| `obscene` | Profane or vulgar language |
| `threat` | Content threatening harm |
| `insult` | Personal attacks or derogatory language |
| `identity_hate` | Hate speech targeting identity groups |

---

## 🧠 Approach

- **Text Preprocessing** — Cleaning, lowercasing, and normalizing raw comment text
- **Exploratory Data Analysis** — Label distribution analysis and word cloud visualizations
- **Feature Engineering** — Tokenization via HuggingFace `tokenizers`
- **Modeling** — Fine-tuned Transformer model (e.g., BERT) for multi-label classification using PyTorch
- **NLP Utilities** — spaCy (`en_core_web_sm`) for linguistic analysis
- **Evaluation** — Per-label and macro metrics (accuracy, F1, AUC-ROC)

---

## 📁 Project Structure

```
toxic-comment-classification/
├── notebooks/
│   └── Toxic Comment Classification.ipynb   # Main analysis & modeling notebook
├── environment.yml                           # Conda environment specification
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Deep Learning | `torch`, `transformers` |
| NLP | `spacy`, `tokenizers` |
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `wordcloud` |
| Environment | `jupyter`, `jupyterlab` |
| GPU Support | NVIDIA CUDA 12.6 |

---

## ⚙️ Setup & Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- (Optional) NVIDIA GPU with CUDA 12.6 for accelerated training

### 1. Clone the Repository

```bash
git clone https://github.com/JenilGoti/toxic-comment-classification.git
cd toxic-comment-classification
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate toxic-comment-classification
```

> **Note:** The `environment.yml` pins exact package versions. If you encounter conflicts, you can install the core dependencies manually:
>
> ```bash
> pip install torch transformers spacy pandas numpy matplotlib wordcloud
> python -m spacy download en_core_web_sm
> ```

### 3. Launch the Notebook

```bash
jupyter lab
# or
jupyter notebook
```

Open `notebooks/Toxic Comment Classification.ipynb` to get started.

---

## 📊 Dataset

This project uses the **Jigsaw Toxic Comment Classification** dataset from Kaggle.

1. Accept the competition rules and download the data from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
2. Place the CSV files in a `data/` directory at the project root:

```
toxic-comment-classification/
└── data/
    ├── train.csv
    ├── test.csv
    └── test_labels.csv
```

---

## 🚀 Usage

Run the notebook cells top-to-bottom. The workflow is:

1. **Load & explore** the dataset (label distributions, text length stats, word clouds)
2. **Preprocess** comments (cleaning, tokenization)
3. **Train** the Transformer-based multi-label classifier
4. **Evaluate** performance per toxicity category
5. **Predict** on new comments

---

## 📈 Results

| Accuracy | 0,9 |

---

## 🤝 Contributing

Contributions and suggestions are welcome! Feel free to open an issue or submit a pull request.

---
