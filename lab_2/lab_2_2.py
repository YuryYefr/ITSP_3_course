import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from lab_2.helpers import data_normalizer

data = data_normalizer()
X, y = data['X'], data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Параметри KNN з різними метриками
metrics = {'Euclidean': 'euclidean', 'Manhattan': 'manhattan', 'Minkowski': 'minkowski'}

for metric_name, metric in metrics.items():
    print(f"\n--- {metric_name} Distance Metric ---")

    # Модель KNN
    knn_model = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn_model.fit(X_train, y_train)
    y_knn_pred = knn_model.predict(X_test)

    # Оцінка метрик
    print("Accuracy:", accuracy_score(y_test, y_knn_pred))
    print("Recall:", recall_score(y_test, y_knn_pred, average='macro'))
    print("F1-Score:", f1_score(y_test, y_knn_pred, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_knn_pred))
    # print(classification_report(y_test, y_knn_pred))

    # Візуалізація кластерів за допомогою PCA
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    # Створимо DataFrame для візуалізації
    pca_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2'])
    pca_df['Predicted'] = y_knn_pred
    pca_df['Actual'] = y_test.reset_index(drop=True)

    # Графік
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Actual', style='Predicted', palette='Set1', s=100, alpha=0.7)
    plt.title(f"KNN Clusters with {metric_name} Distance Metric")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Classes", loc='best')
    plt.show()
