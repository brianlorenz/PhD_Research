from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def kmeans(pixel_seds):
    X = pixel_seds.T
    range_n_clusters = [2, 4, 6, 8, 10, 12, 14]
    range_n_clusters = [8]

    for n_clusters in range_n_clusters:
        kmeans_out = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(X) # requires (samples, features). In this case, each pixel is sample and each image is a feature
        silhouette_avg = silhouette_score(X, kmeans_out.labels_)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
    cluster_values = kmeans_out.labels_ + 1
    
    return cluster_values