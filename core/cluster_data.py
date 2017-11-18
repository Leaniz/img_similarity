import os
from sklearn.cluster import (KMeans, SpectralClustering, AffinityPropagation,
                             AgglomerativeClustering, Birch, DBSCAN,
                             FeatureAgglomeration, MiniBatchKMeans, MeanShift)
from sklearn.metrics import silhouette_score
from sklearn.externals import joblib
from datetime import datetime

import core.const as const


def cluster_support_data(data, n_clusters_list, option="kmeans", verbose=0):

    f_path = os.path.abspath(__file__)
    wd = os.path.dirname(f_path) + "\\models\\"

    features = [col for col in data.columns if col not in const.EXCLUDED_COLS]
    X = data[features]
    max_score = 0

    for n_clusters in n_clusters_list:

        if option == "kmeans":
            clst = KMeans(n_clusters=n_clusters,
                          random_state=const.RANDOM_STATE).fit(X)
            preds = clst.predict(X)

        elif option == "spectral":
            clst = SpectralClustering(n_clusters=n_clusters,
                                      random_state=const.RANDOM_STATE).fit(X)
            preds = clst.fit_predict(X)

        score = silhouette_score(X, preds, random_state=const.RANDOM_STATE)

        if verbose:
            print(("Silhouette score of "
                   "{:5.4} with {} and {} clusters").format(score, option,
                                                            n_clusters))

        if score > max_score:

            if not verbose:
                print(("Silhouette score of "
                       "{:5.4} with {} and {} clusters").format(score, option,
                                                                n_clusters))

            t = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            model_name = "clst_{}_{}_support_{}_{}.p".format(option,
                                                             n_clusters,
                                                             int(score * 100),
                                                             t)
            model_path = wd + model_name
            joblib.dump(clst, model_path)

            max_score = score
