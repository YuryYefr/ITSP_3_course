import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import category_encoders as ce
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

# 1. Load the dataset
df = pd.read_csv('./car_evaluation.csv')  # Replace with actual path
column_names = ['class', 'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
df.columns = column_names

# 2. Encode categorical variables
encoder = ce.OrdinalEncoder(cols=column_names)
df = encoder.fit_transform(df)

# 3. Split data into features (X) and target (y)
X = df.drop(columns=['class'])
y = df['class'] - 1  # class labels starting from 0

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train, predict and evaluate model
def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"{model_name} Confusion Matrix:\n", conf_matrix)

    # Classification Report
    class_report = classification_report(y_test, y_pred)
    print(f"\n{model_name} Classification Report:\n", class_report)

    # AUC for each class
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    roc_auc = {}
    for i in range(4):  # For each class
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[f'class_{i}'] = auc(fpr, tpr)
    print(f"\n{model_name} AUC for each class:", roc_auc)

    # Plotting the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plotting the ROC Curves for each class
    plt.figure(figsize=(10, 6))
    for i in range(4):  # For each class
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} ROC (area = {roc_auc[f"class_{i}"]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# 5. Initialize models
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', random_state=42)
xgbrf_model = xgb.XGBRFClassifier(objective='multi:softprob', num_class=4, eval_metric='mlogloss', random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 6. Evaluate each model
print("XGBoost Model Evaluation")
train_and_evaluate_model(xgb_model, "XGBoost")

print("\nXGBRF Model Evaluation")
train_and_evaluate_model(xgbrf_model, "XGBRF")

print("\nRandom Forest Model Evaluation")
train_and_evaluate_model(rf_model, "Random Forest")
