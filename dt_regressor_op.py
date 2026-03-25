import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import graphviz

# ----------------------------
# 1. Load Data
# ----------------------------
df = pd.read_csv("your_data.csv")

# ----------------------------
# 2. Define Features
# ----------------------------
categorical_cols = [
    'MRCH_CAT_CD',
    'RCUR_PYMN_IN',
    'CNP_IND',
    'Payment_Method',
    'SPEND_BAND'
]

boolean_cols = ['is_tokenised']

target = 'INTR_RT_CD'

X = df[categorical_cols + boolean_cols]
y = df[target]

# Convert boolean → int
X[boolean_cols] = X[boolean_cols].astype(int)

# ----------------------------
# 3. Train-Test Split (70:30)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ----------------------------
# 4. Preprocessing Pipeline
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# ----------------------------
# 5. Model
# ----------------------------
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', dt_model)
])

# ----------------------------
# 6. Train
# ----------------------------
pipeline.fit(X_train, y_train)

# ----------------------------
# 7. Predictions
# ----------------------------
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Probabilities
y_test_proba = pipeline.predict_proba(X_test)

# ----------------------------
# 8. Metrics
# ----------------------------
def evaluate(y_true, y_pred, dataset_name):
    print(f"\n--- {dataset_name} Metrics ---")
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))

evaluate(y_train, y_train_pred, "Train")
evaluate(y_test, y_test_pred, "Test")

print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))

# ----------------------------
# 9. Cross Validation
# ----------------------------
cv_results = cross_validate(
    pipeline,
    X,
    y,
    cv=5,
    scoring=['precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=True
)

print("\n--- Cross Validation ---")
for key, value in cv_results.items():
    print(key, np.mean(value))

# ----------------------------
# 10. Export Decision Tree
# ----------------------------
# Get trained tree
tree_model = pipeline.named_steps['model']

# Get feature names after encoding
feature_names = pipeline.named_steps['preprocessor']\
    .get_feature_names_out()

dot_data = export_graphviz(
    tree_model,
    out_file=None,
    feature_names=feature_names,
    class_names=tree_model.classes_.astype(str),
    filled=True,
    rounded=True,
    special_characters=True
)

# Save as DOT
with open("decision_tree.dot", "w") as f:
    f.write(dot_data)

# Convert to SVG
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="svg")
