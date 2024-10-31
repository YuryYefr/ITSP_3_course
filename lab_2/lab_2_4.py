from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv('./mall_customers.csv')
df.dropna(inplace=True)
df = df.rename(columns={'Annual Income (k$)': 'income'})
X = df.drop(columns=['income'])
y_true = df['income']
X = pd.get_dummies(df, columns=['Genre'])

# Словник для результатів
results = {}

# Функція кластеризації та обчислення метрик
def cluster_and_evaluate(model, model_name):
    model.fit(X)
    labels = model.labels_

    silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else None
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

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


# Кластеризація Affinity Propagation
ap_model = AffinityPropagation(random_state=42)
cluster_and_evaluate(ap_model, "Affinity Propagation")

# Кластеризація KMeans з фіксованою кількістю кластерів
kmeans_model = KMeans(n_clusters=3, random_state=42)
cluster_and_evaluate(kmeans_model, "K-Means")

# Виведення результатів метрик для кожного методу
print("Clustering Evaluation Metrics:")
for method, metrics in results.items():
    print(f"\n{method} Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")


# Нульова модель
def evaluate_null_model(y_true):
    null_labels = np.zeros_like(y_true)
    ari_null = adjusted_rand_score(y_true, null_labels)
    nmi_null = normalized_mutual_info_score(y_true, null_labels)

    print("\nNull Model Metrics:")
    print("Adjusted Rand Index (ARI):", ari_null)
    print("Normalized Mutual Information (NMI):", nmi_null)


evaluate_null_model(y_true)

# Перевірка на оверфітінг: розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

# Оцінка для Affinity Propagation на train/test
ap_train = AffinityPropagation(random_state=42)
ap_train.fit(X_train)
train_labels_ap = ap_train.labels_

ap_test = AffinityPropagation(random_state=42)
ap_test.fit(X_test)
test_labels_ap = ap_test.labels_

print("\nAffinity Propagation Overfitting Check:")
print("Train ARI:", adjusted_rand_score(y_train, train_labels_ap))
print("Test ARI:", adjusted_rand_score(y_test, test_labels_ap))
print("Train NMI:", normalized_mutual_info_score(y_train, train_labels_ap))
print("Test NMI:", normalized_mutual_info_score(y_test, test_labels_ap))

# Оцінка для KMeans на train/test
kmeans_train = KMeans(n_clusters=3, random_state=42)
kmeans_train.fit(X_train)
train_labels_kmeans = kmeans_train.labels_

kmeans_test = KMeans(n_clusters=3, random_state=42)
kmeans_test.fit(X_test)
test_labels_kmeans = kmeans_test.labels_

print("\nK-Means Overfitting Check:")
print("Train ARI:", adjusted_rand_score(y_train, train_labels_kmeans))
print("Test ARI:", adjusted_rand_score(y_test, test_labels_kmeans))
print("Train NMI:", normalized_mutual_info_score(y_train, train_labels_kmeans))
print("Test NMI:", normalized_mutual_info_score(y_test, test_labels_kmeans))
