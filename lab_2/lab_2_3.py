from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

from lab_2.helpers import data_normalizer

data = data_normalizer()
X, y_true = data['X'], data['y']

# Словник для зберігання результатів
results = {}


# Функція для кластеризації та обчислення метрик
def cluster_and_evaluate(model, model_name):
    model.fit(X)
    labels = model.labels_

    # Обчислення метрик
    silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else None
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    # Зберігання результатів
    results[model_name] = {
        'Silhouette Coefficient': silhouette,
        'Adjusted Rand Index (ARI)': ari,
        'Normalized Mutual Information (NMI)': nmi
    }

    # Візуалізація кластерів через PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set1', s=50, alpha=0.7)
    plt.title(f"Clustering Visualization: {model_name}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Clusters", loc='best')
    plt.show()


# Кластеризація та оцінка для кожного алгоритму
# AGNES (Agglomerative Clustering)
agnes = AgglomerativeClustering(n_clusters=3)
cluster_and_evaluate(agnes, "AGNES")

# BIRCH
birch = Birch(n_clusters=3)
cluster_and_evaluate(birch, "BIRCH")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_and_evaluate(dbscan, "DBSCAN")

# Виведення результатів
print("Clustering Evaluation Metrics:")
for method, metrics in results.items():
    print(f"\n{method} Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")


# Нульова модель: всі точки в одному кластері
def evaluate_null_model(y_true):
    # Всі точки в одному кластері (наприклад, "0")
    null_labels = np.zeros_like(y_true)

    ari_null = adjusted_rand_score(y_true, null_labels)
    nmi_null = normalized_mutual_info_score(y_true, null_labels)

    print("\nNull Model Metrics:")
    print("Adjusted Rand Index (ARI):", ari_null)
    print("Normalized Mutual Information (NMI):", nmi_null)


# Виконання нульової моделі
evaluate_null_model(y_true)

# Перевірка на оверфітінг (розбиття на train/test для кластеризації)
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

# Наприклад, AGNES кластеризація на train і перевірка на test
agnes = AgglomerativeClustering(n_clusters=3)
agnes.fit(X_train)
train_labels = agnes.labels_

# Кластеризація test set на основі того ж алгоритму
agnes_test = AgglomerativeClustering(n_clusters=3)
agnes_test.fit(X_test)
test_labels = agnes_test.labels_

# Оцінка на train і test для AGNES
print("\nAGNES Overfitting Check:")
print("Train ARI:", adjusted_rand_score(y_train, train_labels))
print("Test ARI:", adjusted_rand_score(y_test, test_labels))
print("Train NMI:", normalized_mutual_info_score(y_train, train_labels))
print("Test NMI:", normalized_mutual_info_score(y_test, test_labels))
