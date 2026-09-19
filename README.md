# Student Dropout Risk Prediction

An end-to-end Machine Learning project that predicts the probability of student dropout using academic, demographic, financial, and behavioral features.

## Project Overview

Student dropout can often be associated with multiple factors such as attendance, academic performance, study habits, financial circumstances, assignment delays, stress, and other student characteristics.

This project implements a **Logistic Regression** model to estimate a student's dropout probability and classify the student into:

* Low Risk
* Medium Risk
* High Risk

The project also demonstrates how a trained ML model can be used as an educational early-warning system.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Logistic Regression
* StandardScaler
* Git & GitHub

## Machine Learning Pipeline

The project follows these steps:

1. Load the student dataset
2. Separate features and target
3. Split the data into training and testing sets
4. Standardize the features
5. Train Logistic Regression using gradient descent
6. Evaluate the model
7. Predict dropout risk for individual students

## Features

The model uses information such as:

* Age
* Gender
* Family Income
* Internet Access
* Study Hours per Day
* Attendance Rate
* Assignment Delay
* Travel Time
* Part-Time Job
* Scholarship
* Stress Index
* GPA
* CGPA
* Semester
* Department
* Parental Education

## Model

Instead of directly using Scikit-learn's LogisticRegression implementation, Logistic Regression is implemented from scratch using:

* Sigmoid activation
* Binary cross-entropy loss
* Gradient descent
* Model weights and bias

The model uses 2,000 training epochs with a learning rate of 0.01.

## Model Evaluation

The trained model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report

The test set consists of 20% of the dataset, while stratification is used to preserve the target-class distribution.

## Student Risk Prediction

After training, the system accepts the characteristics of an individual student and returns:

**Dropout Probability:** Percentage probability of dropout

**Risk Category:**

| Probability  | Risk Category |
| ------------ | ------------- |
| Below 30%    | Low Risk      |
| 30%–59.99%   | Medium Risk   |
| 60% or above | High Risk     |

The same StandardScaler fitted during training is reused when processing new students, ensuring that new inputs are transformed consistently with the training data.

## Example Prediction

The project includes multiple test cases representing different student profiles:

### Test Case 1 — Strong Standing

A student with strong academic performance, high attendance, low assignment delays, and low stress.

### Test Case 2 — Mixed Signals

A student with moderate academic performance and attendance, alongside financial, workload, and stress-related factors.

### Test Case 3 — Struggling and Disengaged

A student with low attendance, low GPA/CGPA, significant assignment delays, high stress, and other risk factors.

The application produces a dropout probability and corresponding risk category for each student.

## Educational Early-Warning System

This project can serve as a prototype for an educational early-warning system.

In a real educational environment, authorized academic staff could use regularly updated student information to identify students who may require additional support.

For example, a system could flag students based on combinations of:

* Declining academic performance
* Low attendance
* Increasing assignment delays
* High stress indicators
* Financial difficulties
* Reduced engagement

Once a student is identified as potentially at risk, the institution could provide appropriate support such as academic advising, tutoring, financial guidance, or student counseling.

The prediction should be treated as a **support signal rather than a final decision about a student**. Human review and appropriate institutional policies would remain important.

## Limitations

This project is a demonstration of machine learning and should not be treated as a definitive system for making decisions about students.

Potential limitations include:

* Dataset quality and representativeness
* Possible bias in historical data
* Limited features
* Prediction uncertainty
* Changing student circumstances
* Difference between the training dataset and real-world students

A production system would require additional validation, privacy protections, monitoring, and institutional review.

## How to Run

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Navigate to the project:

```bash
cd student-dropout-risk-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python student_dropout_prediction.py
```

## Project Workflow

```text
Student Dataset
       ↓
Data Preparation
       ↓
Train/Test Split
       ↓
Feature Scaling
       ↓
Logistic Regression
       ↓
Model Evaluation
       ↓
Student Input
       ↓
Dropout Probability
       ↓
Risk Category
```

## Conclusion

This project demonstrates a complete Machine Learning workflow, from dataset preparation and model training to evaluation and individual student risk prediction.

It also demonstrates how a predictive model can potentially be integrated into an educational early-warning workflow to help institutions identify students who may benefit from additional support.
