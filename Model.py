# ================================
# COMPLETE NAIVE BAYES PIPELINE
# ================================

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
categories = ["sci.space", "rec.autos"]

data = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes")
)

X_text = data.data
y = data.target

# 2. Train-test split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Text preprocessing (TF-IDF)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# 4. Train Naive Bayes model
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 6. Predict on new unseen text
new_docs = [
    "NASA launched a spacecraft into deep space",
    "My car engine is making a strange noise"
]

new_docs_vec = vectorizer.transform(new_docs)
predictions = model.predict(new_docs_vec)

print("\nNew Predictions:")
for doc, label in zip(new_docs, predictions):
    print(f"Text: {doc}")
    print(f"Predicted Class: {data.target_names[label]}\n")

# 7. Inspect learned parameters (optional but important)
print("Class log priors:", model.class_log_prior_)
print("Feature log prob shape:", model.feature_log_prob_.shape)