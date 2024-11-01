import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('./car_evaluation.csv')

# Adding column names
column_names = ['class', 'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
df.columns = column_names

# 2. Encode categorical variables
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
df = encoder.fit_transform(df)

# 3. Split data into features (X) and target (y)
X = df.drop(columns=['class'])  # 'class' is the target column
y = df['class']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 6. Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# 7. Metrics
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", class_report)

# AUC score (One-vs-Rest for each class)
y_test_binarized = label_binarize(y_test, classes=rf_model.classes_)
roc_auc_dict = {}
for i, class_label in enumerate(rf_model.classes_):
    roc_auc_dict[class_label] = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])

print("\nAUC for each class:", roc_auc_dict)

# 8. Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 9. Plotting the ROC Curves for each class
plt.figure(figsize=(10, 8))
for i, class_label in enumerate(rf_model.classes_):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.2f})')

# Plotting the diagonal reference line
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc="lower right")
plt.show()
