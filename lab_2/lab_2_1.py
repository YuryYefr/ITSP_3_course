import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

from helpers import data_normalizer

data = data_normalizer()
X, y = data['X'], data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Нуль-гіпотеза
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
y_dummy_pred = dummy_clf.predict(X_test)

# 2. Навчання моделей
# Байєсовий класифікатор
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_nb_pred = nb_model.predict(X_test)

# Метод опорних векторів
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_svm_pred = svm_model.predict(X_test)

# 3. Оцінка метрик
# Метрики для нуль-гіпотези
print("Dummy Classifier Metrics:")
print("Accuracy:", accuracy_score(y_test, y_dummy_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_dummy_pred))

# Метрики для Байєсового класифікатора
print("\nNaive Bayes Metrics:")
print("Accuracy:", accuracy_score(y_test, y_nb_pred))
print("Recall:", recall_score(y_test, y_nb_pred, average='macro'))
print("F1-Score:", f1_score(y_test, y_nb_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_nb_pred))
print(classification_report(y_test, y_nb_pred))

# Метрики для методу опорних векторів
print("\nSVM Metrics:")
print("Accuracy:", accuracy_score(y_test, y_svm_pred))
print("Recall:", recall_score(y_test, y_svm_pred, average='macro'))
print("F1-Score:", f1_score(y_test, y_svm_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_svm_pred))
print(classification_report(y_test, y_svm_pred))

# 4. Перевірка на оверфітинг
# Оцінка метрик для тренувального набору даних
y_nb_train_pred = nb_model.predict(X_train)
y_svm_train_pred = svm_model.predict(X_train)

print("\nOverfitting Check for Naive Bayes:")
print("Train Accuracy:", accuracy_score(y_train, y_nb_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_nb_pred))

print("\nOverfitting Check for SVM:")
print("Train Accuracy:", accuracy_score(y_train, y_svm_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_svm_pred))

# Виконаємо PCA для зменшення вимірності до 2D на тестових даних
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Створимо DataFrame для зручності побудови
pca_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2'])
pca_df['Predicted'] = y_svm_pred
pca_df['Actual'] = y_test.reset_index(drop=True)  # Скидаємо індекси для y_test для коректного злиття

# Побудуємо графік кластеризації для SVM на тестовій вибірці
plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Actual', style='Predicted', palette='Set1', s=100, alpha=0.7)
plt.title("SVM Clusters Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Classes", loc='best')
plt.show()
