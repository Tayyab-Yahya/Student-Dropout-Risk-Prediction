"""
============================================================
STUDENT DROPOUT RISK PREDICTION — COMPLETE PROJECT
============================================================
Phases:
  1. Load data
  2. Train/test split
  3. Feature scaling
  4. From-scratch logistic regression (gradient descent)
  5. Model evaluation
  6. Student risk prediction function + multiple test cases

This is a cleaned-up, corrected version of the original notebook.
One bug from the original was fixed here: the original re-created
an *unfitted* StandardScaler() right before using it to transform
new student input, which shadowed the scaler fitted on the
training data and would raise a NotFittedError. Here the same
scaler object fitted in Phase 3 is reused everywhere.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# ============================================================
# PHASE 1: LOAD DATA
# ============================================================

DATA_PATH = "/content/cleaned_data.csv"  # change this to your local path if needed

df = pd.read_csv(DATA_PATH)

x = df.drop(columns=["Dropout"])
y = df["Dropout"]

# Exact column order the model was trained on — required so that
# any new student input lines up with the right weight.
FEATURE_ORDER = x.columns.tolist()

# ============================================================
# PHASE 2: TRAIN / TEST SPLIT
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y
)

# ============================================================
# PHASE 3: FEATURE SCALING
# ============================================================

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ============================================================
# PHASE 4: FROM-SCRATCH LOGISTIC REGRESSION
# ============================================================


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)


def calculate_loss(y_true, probabilities):
    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )


n_features = x_train_scaled.shape[1]
weights = np.zeros(n_features)
bias = 0.0

learning_rate = 0.01
epochs = 2000
loss_history = []

for epoch in range(epochs):
    probabilities = predict(x_train_scaled, weights, bias)
    loss = calculate_loss(y_train, probabilities)
    loss_history.append(loss)

    n = len(y_train)
    dw = (1 / n) * np.dot(x_train_scaled.T, (probabilities - y_train))
    db = (1 / n) * np.sum(probabilities - y_train)

    weights = weights - (learning_rate * dw)
    bias = bias - (learning_rate * db)

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}  |  Loss: {loss:.4f}")

train_probabilities = predict(x_train_scaled, weights, bias)
train_pred = (train_probabilities >= 0.5).astype(int)
train_accuracy = accuracy_score(y_train, train_pred)
print("\nTraining Accuracy (%):", train_accuracy * 100)

plt.figure(figsize=(5, 3))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.tight_layout()
plt.show()

# ============================================================
# PHASE 5: MODEL EVALUATION
# ============================================================

test_probabilities = predict(x_test_scaled, weights, bias)
y_pred = (test_probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, test_probabilities)

print("\nModel Evaluation Results")
print("=" * 40)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Dropout", "Dropout"]))

ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Dropout", "Dropout"]
).plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ============================================================
# PHASE 6: STUDENT RISK PREDICTION APPLICATION
# ============================================================
#
# NOTE ON THE FIX: the original notebook did
#     scaler = StandardScaler()          # <-- creates a brand-new, UNFITTED scaler
#     X_input_scaled = scaler.transform(X_input)   # <-- crashes: NotFittedError
# right inside this section, which silently overwrote the scaler
# fitted back in Phase 3. Below, we simply reuse that same fitted
# `scaler` object instead of creating a new one.


def build_student_row(
    age,
    gender,                 # "Female" or "Male"
    family_income,
    internet_access,        # "Yes" or "No"
    study_hours_per_day,
    attendance_rate,
    assignment_delay_days,
    travel_time_minutes,
    part_time_job,          # "Yes" or "No"
    scholarship,            # "Yes" or "No"
    stress_index,
    gpa,
    cgpa,
    semester,                # "Year 1" | "Year 2" | "Year 3" | "Year 4"
    department,              # "Business" | "CS" | "Engineering" | "Science" | "Baseline"
    parental_education,      # "High School" | "Master" | "PhD" | "Baseline"
):
    """Builds a single-row DataFrame with columns in FEATURE_ORDER."""

    row = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Family_Income": family_income,
        "Internet_Access": 1 if internet_access == "Yes" else 0,
        "Study_Hours_per_Day": study_hours_per_day,
        "Attendance_Rate": attendance_rate,
        "Assignment_Delay_Days": assignment_delay_days,
        "Travel_Time_Minutes": travel_time_minutes,
        "Part_Time_Job": 1 if part_time_job == "Yes" else 0,
        "Scholarship": 1 if scholarship == "Yes" else 0,
        "Stress_Index": stress_index,
        "GPA": gpa,
        "CGPA": cgpa,
        "Semester_Year 2": 1 if semester == "Year 2" else 0,
        "Semester_Year 3": 1 if semester == "Year 3" else 0,
        "Semester_Year 4": 1 if semester == "Year 4" else 0,
        "Department_Business": 1 if department == "Business" else 0,
        "Department_CS": 1 if department == "CS" else 0,
        "Department_Engineering": 1 if department == "Engineering" else 0,
        "Department_Science": 1 if department == "Science" else 0,
        "Parental_Education_High School": 1 if parental_education == "High School" else 0,
        "Parental_Education_Master": 1 if parental_education == "Master" else 0,
        "Parental_Education_PhD": 1 if parental_education == "PhD" else 0,
    }

    return pd.DataFrame([row])[FEATURE_ORDER]


def predict_student_risk(**student_fields):
    """
    Takes raw student fields (see build_student_row for the full list
    of arguments), scales them with the SAME fitted scaler used in
    training, runs them through the from-scratch logistic regression,
    and returns (probability, probability_percentage, risk_category).
    """

    input_data = build_student_row(**student_fields).astype(float)

    # Reuses the scaler fitted in Phase 3 — NOT a new StandardScaler().
    # (Passing the DataFrame, not .values, keeps the column names that
    # the scaler was originally fitted with, avoiding a feature-name warning.)
    X_input_scaled = scaler.transform(input_data)

    probability = predict(X_input_scaled, weights, bias)[0]
    probability_percentage = probability * 100

    if probability < 0.30:
        risk_category = "Low Risk"
    elif probability < 0.60:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    return probability, probability_percentage, risk_category


def print_prediction(label, **student_fields):
    probability, probability_percentage, risk = predict_student_risk(**student_fields)
    print("=" * 45)
    print(f"  {label}")
    print("=" * 45)
    print(f"Dropout Probability : {probability_percentage:.2f}%")
    print(f"Risk Category        : {risk}")
    print("=" * 45, "\n")


# ============================================================
# MULTIPLE TEST CASES (for screenshots / deliverable)
# ============================================================

if __name__ == "__main__":

    print_prediction(
        "TEST CASE 1 — Strong standing",
        age=20, gender="Female", family_income=62000, internet_access="Yes",
        study_hours_per_day=6, attendance_rate=94, assignment_delay_days=0,
        travel_time_minutes=15, part_time_job="No", scholarship="Yes",
        stress_index=3.2, gpa=3.6, cgpa=3.5,
        semester="Year 2", department="CS", parental_education="Master",
    )

    print_prediction(
        "TEST CASE 2 — Mixed signals",
        age=22, gender="Male", family_income=34000, internet_access="Yes",
        study_hours_per_day=3.5, attendance_rate=75, assignment_delay_days=4,
        travel_time_minutes=45, part_time_job="Yes", scholarship="No",
        stress_index=6.0, gpa=2.4, cgpa=2.5,
        semester="Year 3", department="Business", parental_education="Baseline",
    )

    print_prediction(
        "TEST CASE 3 — Struggling, disengaged",
        age=24, gender="Male", family_income=14000, internet_access="No",
        study_hours_per_day=1.0, attendance_rate=48, assignment_delay_days=14,
        travel_time_minutes=95, part_time_job="Yes", scholarship="No",
        stress_index=8.7, gpa=1.4, cgpa=1.6,
        semester="Year 4", department="Baseline", parental_education="High School",
    )