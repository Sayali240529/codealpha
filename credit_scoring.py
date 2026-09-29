import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("credit_data.csv")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
data = pd.read_csv("credit_data.csv")

print("First five records:")
print(data.head())

# 2. Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# 3. Separate input and output
X = data[["Income", "Debt", "PaymentHistory", "CreditScore"]]
y = data["Creditworthy"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Create model
model = LogisticRegression()

# 7. Train model
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# 9. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

print("\nModel Performance")
print("-------------------------")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Credit Scoring - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

plt.xticks([0, 1], ["Poor", "Good"])
plt.yticks([0, 1], ["Poor", "Good"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()