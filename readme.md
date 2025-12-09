# Half Marathon Time Predictor 🏃‍♂️⏱️

**AI-powered half marathon performance prediction using machine learning and natural language processing**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3-green.svg)](https://pycaret.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-purple.svg)](https://openai.com)

## 🎯 Project Overview

Half Marathon Time Predictor is an intelligent sports analytics application that predicts your half marathon finish time based on your 5km performance. Using machine learning (PyCaret), natural language processing (OpenAI GPT-4o-mini), AI observability (Langfuse), and cloud storage (Digital Ocean Spaces).

### Key Statistics
- **Dataset**: 8,000+ runners from Wrocław Half Marathon 2023-2024
- **ML Model**: PyCaret regression pipeline (best_model.pkl)
- **Prediction Accuracy**: R² > 0.85 (strong correlation)
- **Categories**: 14 age/gender groups (M20-M80, K20-K80)

## ✨ Key Features

- 🤖 **Natural Language Input**: Describe yourself conversationally
- 📊 **ML Prediction**: PyCaret regression with AutoML
- 🔍 **AI Observability**: Langfuse monitoring for LLM calls
- ☁️ **Cloud Storage**: Digital Ocean Spaces for model/data
- 📈 **Comparative Analytics**: Histogram with your ranking

## 🚀 Quick Start

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure .env**
```env
OPENAI_API_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
AWS_ACCESS_KEY_ID=your_do_key
AWS_SECRET_ACCESS_KEY=your_do_secret
```

3. **Run app**
```bash
streamlit run app.py
```

## 💻 Usage

1. **Enter your data**: "Jestem mężczyzną, mam 32 lata, 5km w 24:15"
2. **Click "Analizuj"**: GPT-4o-mini extracts gender, age, time
3. **View prediction**: See your half marathon time
4. **Compare**: Histogram shows your ranking

## 🧮 How It Works

1. **NLP Extraction**: GPT-4o-mini parses natural language
2. **Feature Engineering**: Calculate age category, 5km pace
3. **ML Prediction**: PyCaret model predicts half marathon time
4. **Visualization**: Compare with 8,000+ runners

## 📊 Tech Stack

- **Frontend**: Streamlit
- **ML**: PyCaret, scikit-learn
- **NLP**: OpenAI GPT-4o-mini
- **Monitoring**: Langfuse
- **Storage**: Digital Ocean Spaces (S3-compatible)
- **Data**: pandas, numpy
- **Viz**: matplotlib, seaborn, plotly

## 📁 Project Structure

```
├── app.py                 # Main Streamlit app (327 lines)
├── 1.ipynb               # Data exploration
├── 2.ipynb               # Feature engineering (1.5MB)
├── 3.ipynb               # Model training (1.8MB)
├── best_model.pkl        # Trained model (32KB)
├── df.csv                # Combined dataset (2.4MB)
└── requirements.txt      # Dependencies
```

## 🎓 Key Learning Points

1. End-to-end ML pipeline (data → training → deployment)
2. PyCaret AutoML for model selection
3. OpenAI GPT-4o-mini for NLP
4. Langfuse for LLM observability
5. Digital Ocean Spaces (S3-compatible storage)
6. Sports analytics domain knowledge

## 📄 License

Educational project for AI/Data Science course.

---

**Built with** 🏃 Python, PyCaret, OpenAI, Langfuse & Streamlit
