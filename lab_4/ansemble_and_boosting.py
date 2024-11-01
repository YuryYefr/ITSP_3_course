from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from lab_4.helpers import process_data

data = process_data()
# Spliting target variable and independent variables
X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']

# Transforming to numerical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting the data into training set and testset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Spliting target variable and independent variables
X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Gradient Boosting
gbc = GradientBoostingClassifier(n_estimators=100, random_state=0)
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)
acc_gbc = accuracy_score(y_test, y_pred_gbc)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=0)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
acc_ada = accuracy_score(y_test, y_pred_ada)

# Ensemble Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gbc', gbc), ('xgb', xgb), ('ada', ada)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
acc_voting = accuracy_score(y_test, y_pred_voting)

# Print scores
print(f"Random Forest Accuracy: {acc_rf}")
print(f"Gradient Boosting Accuracy: {acc_gbc}")
print(f"XGBoost Accuracy: {acc_xgb}")
print(f"AdaBoost Accuracy: {acc_ada}")
print(f"Voting Classifier Accuracy: {acc_voting}")
