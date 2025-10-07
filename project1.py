# Read data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Step 1: Data Processing
data = pd.read_csv("Project 1 Data.csv")

# Step 2: Data Visualization
# Make histogram and capture counts + bins
counts, bins, patches = plt.hist(
    data['Step'], 
    bins=range(1, 15), 
    align='left', 
    rwidth=0.8)

# Add labels above bars
for i in range(len(counts)):
    plt.text(bins[i], counts[i] + 5, str(int(counts[i])), ha='center')

plt.xticks(range(1, 14))
plt.ylim(0,270)
plt.xlabel("Step")
plt.ylabel("Count")
plt.title("Histogram of Steps (1–13)")
plt.grid(axis='y')
plt.show()

# Define exactly 13 distinct colors (any you like)
colors = [
    "red", "blue", "green", "orange", "purple", "brown", "pink",
    "gray", "olive", "cyan", "magenta", "yellow", "teal"
]

# Create figure
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')

# Plot each step separately with its own color
for step in sorted(data['Step'].unique()):
    subset = data[data['Step'] == step]
    ax.scatter(
        subset['X'], subset['Y'], subset['Z'],
        color=colors[step-1],   # match step index to color
        label=f"Step {step}",
        s=30
    )

# Labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Scatter Plot of 13 Steps")

# Show legend
ax.legend(loc="lower right", bbox_to_anchor=(1.2, -0.05))

plt.show()

# Step 3: Correlation Analysis
from sklearn.model_selection import train_test_split

x = data[['X', 'Y', 'Z']]
y = data['Step']  # multiclass labels 1..13

# 3) Stratified 80/20 split to preserve Step distribution
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y
)

train_with_y = x_train.copy()
train_with_y['Step'] = y_train

corr_matrix = train_with_y.corr()
sns.heatmap(np.abs(corr_matrix))
print(corr_matrix)
# The correlation analysis shows that Step has a strong negative correlation with X (–0.75),
# suggesting that as X increases, the step values tend to decrease consistently.

# # Step 4: Classification Model Development/Engineering

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Build the first classifier based on support vector machines
pipe_mc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability= False, random_state=42))
])

param_grid_mc = {
    "clf__kernel": ["linear", "rbf"],
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto"]  
}

gs_mc = GridSearchCV(
    pipe_mc,
    param_grid=param_grid_mc,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    verbose=1
)

gs_mc.fit(x_train, y_train)
clf1 = gs_mc.best_estimator_

print("\nSVM Classifier")
print("Best params:", gs_mc.best_params_)
print("Training accuracy:", clf1.score(x_train, y_train))
print("Test accuracy:", clf1.score(x_test, y_test))

# Build the second classifier based on Random Forest
from sklearn.ensemble import RandomForestClassifier

pipe_rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 16],
    "min_samples_leaf": [2, 4, 8],
}
gs_rf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, scoring="f1_macro", cv=5, n_jobs=1, refit=True)
gs_rf.fit(x_train, y_train)
clf2 = gs_rf.best_estimator_
print("\nRandom Forest Classifier")
print("Best params:", gs_rf.best_params_)
print("Training accuracy:", clf2.score(x_train, y_train))
print("Test accuracy:", clf2.score(x_test, y_test))

# Build the third classifier based on K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

pipe_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid_knn = {
    "knn__n_neighbors": [3, 5, 7],
    "knn__weights": ["uniform", "distance"]
}

gs_knn = GridSearchCV(
    estimator=pipe_knn,
    param_grid=param_grid_knn,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

gs_knn.fit(x_train, y_train)
clf3 = gs_knn.best_estimator_
print("\nK-Nearest Neighbors Classifier")
print("Best Params:", gs_knn.best_params_)
print("Train Accuracy:", clf3.score(x_train, y_train))
print("Test Accuracy:", clf3.score(x_test, y_test))

# Build the fourth classifier based on logistic regression with RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
pipe_log = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        random_state=42
    ))
])

parameter_log = {
    "clf__C": loguniform(1e-3, 1e3),
    "clf__penalty": ["l2"],
}

rs_log = RandomizedSearchCV(
    estimator=pipe_log,
    param_distributions=parameter_log,
    n_iter=40,
    scoring="roc_auc_ovr",         
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

rs_log.fit(x_train, y_train)
clf4 = rs_log.best_estimator_
print("\nRogistic Regression Classifier")
print("Best Params:", rs_log.best_params_)
print("Train Accuracy:", clf4.score(x_train, y_train))
print("Test Accuracy:", clf4.score(x_test, y_test))

# # Step 5: Model Performance Analysis
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
# SVc
y_pred_clf1 = clf1.predict(x_test)

cm_clf1 = confusion_matrix(y_test, y_pred_clf1)
print("Confusion Matrix (SVM):")
print(cm_clf1)

precision_clf1 = precision_score(y_test, y_pred_clf1, average='weighted')
recall_clf1 = recall_score(y_test, y_pred_clf1, average='weighted')
f1_clf1 = f1_score(y_test, y_pred_clf1, average='weighted')

print("Precision:", precision_clf1)
print("Recall:", recall_clf1)
print("F1 Score:", f1_clf1)

# Random Forest
y_pred_clf2 = clf2.predict(x_test)

cm_clf2 = confusion_matrix(y_test, y_pred_clf2)
print("Confusion Matrix (Random Forest):")
print(cm_clf2)

precision_clf2 = precision_score(y_test, y_pred_clf2, average='weighted')
recall_clf2 = recall_score(y_test, y_pred_clf2, average='weighted')
f1_clf2 = f1_score(y_test, y_pred_clf2, average='weighted')

print("Precision:", precision_clf2)
print("Recall:", recall_clf2)
print("F1 Score:", f1_clf2)

# K Neighbors
y_pred_clf3 = clf3.predict(x_test)

cm_clf3 = confusion_matrix(y_test, y_pred_clf3)
print("Confusion Matrix (K Neighbors):")
print(cm_clf3)

precision_clf3 = precision_score(y_test, y_pred_clf3, average='weighted')
recall_clf3 = recall_score(y_test, y_pred_clf3, average='weighted')
f1_clf3 = f1_score(y_test, y_pred_clf3, average='weighted')

print("Precision:", precision_clf3)
print("Recall:", recall_clf3)
print("F1 Score:", f1_clf3)

# Logistic
y_pred_clf4 = clf4.predict(x_test)


cm_clf4 = confusion_matrix(y_test, y_pred_clf4)
print("Confusion Matrix (Logistic Regression):")
print(cm_clf4)

precision_clf4 = precision_score(y_test, y_pred_clf4, average='weighted')
recall_clf4 = recall_score(y_test, y_pred_clf4, average='weighted')
f1_clf4 = f1_score(y_test, y_pred_clf4, average='weighted')

print("Precision:", precision_clf4)
print("Recall:", recall_clf4)
print("F1 Score:", f1_clf4)

# Step 6: Stacked Model Performance Analysis
from sklearn.ensemble import StackingClassifier
stack_model = StackingClassifier(
    estimators=[("rf", pipe_rf), ("lr", pipe_log)],
    final_estimator=LogisticRegression(max_iter=500, random_state=42),
    stack_method="predict_proba",
    cv=5,
    n_jobs=-1
)

stack_model.fit(x_train, y_train)

y_pred_stack = stack_model.predict(x_test)

acc = accuracy_score(y_test, y_pred_stack)
prec = precision_score(y_test, y_pred_stack, average="weighted")
f1 = f1_score(y_test, y_pred_stack, average="weighted")

print("\nStacked Model")
print("Stacked Model Accuracy:", acc)
print("Stacked Model Precision (weighted):", prec)
print("Stacked Model F1 Score (weighted):", f1)

cm_stack = confusion_matrix(y_test, y_pred_stack)
print("Stack Confusion Matrix:\n", cm_stack)

# Step 7: Model Evaluation - 10 Marks
import joblib

joblib.dump(clf1, "best_model_rf.joblib")

loaded_model = joblib.load("best_model_rf.joblib")

X_new = pd.DataFrame([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3.0, 1.8],
    [9.4, 3.0, 1.3]
], columns=["X", "Y", "Z"]) 
# Predict maintenance step
predicted_steps = loaded_model.predict(X_new)

print("\nPredicted Maintenance Steps:")
for coords, step in zip(X_new.values, predicted_steps):
    print(f"Coordinates {coords} → Step {step}")















