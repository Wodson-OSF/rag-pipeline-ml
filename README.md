# 🚀 RAG Pipeline ML

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Wodson-OSF/rag-pipeline-ml?style=social)](https://github.com/Wodson-OSF/rag-pipeline-ml/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Wodson-OSF/rag-pipeline-ml?style=social)](https://github.com/Wodson-OSF/rag-pipeline-ml/network)

> **RAG Pipeline with real embeddings (sentence-transformers) for Portuguese language support**

---

## 📋 Table of Contents

- [🚀 RAG Pipeline ML](#-rag-pipeline-ml)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Overview](#-overview)
  - [✨ Features](#-features)
    - [DataCleaner](#datacleaner)
    - [RAG Pipeline](#rag-pipeline)
  - [🏗️ Architecture](#️-architecture)
  - [🛠️ Installation](#️-installation)
    - [Prerequisites](#prerequisites)
    - [Clone the repository](#clone-the-repository)
    - [Install dependencies](#install-dependencies)
  - [🚀 Quick Start](#-quick-start)
    - [Data Cleaning Example](#data-cleaning-example)
    - [RAG Pipeline Example](#rag-pipeline-example)
  - [📁 Project Structure](#-project-structure)
  - [🔧 Technologies](#-technologies)
  - [👨‍💻 Author](#-author)

---

## 🎯 Overview

This project implements a complete **RAG (Retrieval-Augmented Generation) Pipeline** with:

1. **DataCleaner**: Professional data validation for ML pipelines
2. **RAG Pipeline**: Semantic search with real embeddings (sentence-transformers)
3. **Multilingual Support**: Optimized for Portuguese language

---

## ✨ Features

### DataCleaner
- ✅ **Age validation**: 0-120 range, integer type
- ✅ **Email validation**: RFC 5322 compliant format
- ✅ **Income validation**: Non-negative float
- ✅ **Text cleaning**: Normalization for NLP
- ✅ **Error tracking**: Comprehensive error collection
- ✅ **Statistics**: Success rate, validation metrics

### RAG Pipeline
- ✅ **Real embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- ✅ **Semantic search**: Cosine similarity retrieval
- ✅ **Context generation**: Prompt building with retrieved documents
- ✅ **Multilingual support**: Portuguese language optimization
- ✅ **Batch processing**: Index multiple documents at once

---

## 🏗️ Architecture

```
[Raw Data]
    ↓
[DataCleaner] → Validated Data
    ↓
[RAG Pipeline]
    ↓
[Documents] → [Embeddings] → [Index]
    ↓
[Query] → [Retrieval] → [Context]
    ↓
[Response Generation] → [Answer]
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Clone the repository
```bash
git clone https://github.com/Wodson-OSF/rag-pipeline-ml.git
cd rag-pipeline-ml
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Data Cleaning Example
```python
from src.data_cleaner import DataCleaner

cleaner = DataCleaner()

data = [
    {"age": 25, "email": "joao@gmail.com", "income": 5000.00},
    {"age": -10, "email": "invalid-email", "income": 3000.00},
]

cleaned_data, errors = cleaner.clean_dataset(data)
stats = cleaner.get_statistics()

print(f"✅ Valid: {stats['valid_records']}")
print(f"❌ Invalid: {stats['invalid_records']}")
```

### RAG Pipeline Example
```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Index documents
docs = [
    {"id": "doc1", "text": "Resetar senha: Acesse 'Esqueci minha senha' no login."},
    {"id": "doc2", "text": "Limpar cache: Configurações > Armazenamento > Limpar cache."},
]

pipeline.batch_index(docs)

# Ask a question
result = pipeline.ask("Como resetar minha senha?")
print(result['response'])
```

---

## 📁 Project Structure

```
rag-pipeline-ml/
├── src/
│   ├── __init__.py
│   ├── data_cleaner.py      # Data validation
│   └── rag_pipeline.py      # RAG implementation
├── tests/
│   └── test_data_cleaner.py
├── .env.example             # Environment variables template
├── .gitignore              # Ignored files
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

---

## 🔧 Technologies

| Technology | Purpose |
|------------|---------|
| **Python 3.12** | Core language |
| **sentence-transformers** | Real embeddings (all-MiniLM-L6-v2) |
| **scikit-learn** | Machine learning utilities |
| **pandas/numpy** | Data processing |
| **pytest** | Unit testing |
| **black/isort** | Code formatting |

---

## 👨‍💻 Author

**Wodson**

- GitHub: [@Wodson-OSF](https://github.com/Wodson-OSF)
- LinkedIn: [Seu LinkedIn](https://www.linkedin.com/in/seu-perfil)

---

**Made with ❤️ for the Machine Learning community**