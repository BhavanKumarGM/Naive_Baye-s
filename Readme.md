# Naive Bayes Text Classification (scikit-learn)

This project implements a **Multinomial Naive Bayes** classifier for **text classification** using the **20 Newsgroups dataset** provided by scikit-learn.

The goal of this project is to demonstrate:
- Proper preprocessing of text data
- Correct use of Naive Bayes for document classification
- End-to-end machine learning workflow (data → model → evaluation)

---

## 📌 Dataset

- **Source:** sklearn.datasets.fetch_20newsgroups
- **Categories used:**
  - `sci.space`
  - `rec.autos`
- Headers, footers, and quotes are removed to reduce noise.

---

## ⚙️ Approach

1. Load text data from sklearn
2. Split data into training and testing sets
3. Convert text into numerical features using **TF-IDF**
4. Train a **Multinomial Naive Bayes** classifier
5. Evaluate the model using accuracy and classification report
6. Predict classes for new unseen text

---

## 🧠 Model Used

- **Algorithm:** Multinomial Naive Bayes  
- **Why Naive Bayes?**
  - Works well for high-dimensional text data
  - Fast and efficient
  - Strong baseline for NLP tasks

---

## 📦 Requirements

- Python **3.10 or higher**
- Required Python packages:
  - scikit-learn
  - numpy
  - scipy

Install dependencies using:

```bash
pip install scikit-learn numpy scipy
