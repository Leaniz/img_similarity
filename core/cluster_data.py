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
    max_score = 0.
    random_s = const.RANDOM_STATE
    centers = None
    best_model = {}

    for n_clusters in n_clusters_list:

        if option == "kmeans":
            clst = KMeans(n_clusters=n_clusters,
                          random_state=random_s).fit(X)
            preds = clst.predict(X)
            centers = clst.cluster_centers_

        elif option == "spectral":
            clst = SpectralClustering(n_clusters=n_clusters,
                                      random_state=random_s).fit(X)
            preds = clst.fit_predict(X)

        elif option == "affinity":
            clst = AffinityPropagation().fit(X)
            preds = clst.predict(X)
            n_clusters = len(clst.cluster_centers_indices_)

        elif option == "agglomerative":
                clst = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
                preds = clst.fit_predict(X)

        elif option == "birch":
            clst = Birch(n_clusters=n_clusters).fit(X)
            preds = clst.fit_predict(X)

        elif option == "dbscan":
            clst = DBSCAN().fit(X)
            preds = clst.fit_predict(X)
            n_clusters = len(set(preds))

        elif option == "featureagg":
            try:
                clst = FeatureAgglomeration(n_clusters=n_clusters).fit(X)
                trans = clst.transform(X)
                preds = [list(t).index(max(t)) for t in trans]
            except ValueError:
                pass

        elif option == "mbkmeans":
            clst = MiniBatchKMeans(n_clusters=n_clusters,
                                   random_state=random_s).fit(X)
            preds = clst.predict(X)

        elif option == "meanshift":
            clst = MeanShift().fit(X)
            preds = clst.fit_predict(X)
            n_clusters = len(set(preds))

        if len(set(preds)) < 2:
            score = 0.
        else:
            score = silhouette_score(X, preds, random_state=random_s)

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

            best_model = {"model": clst, "preds": preds, "centers": centers}

        if option in ["affinity", "dbscan", "meanshift"]:
            break

    return best_model
