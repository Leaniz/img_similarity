import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

import core.const as const


def plot_cluster_results(df, preds, centers):

    cols = [col for col in df.columns if col not in const.EXCLUDED_COLS]
    col_pairs = itertools.combinations(cols, 2)
    predictions = pd.DataFrame(preds, columns=['Cluster'])
    plot_data = pd.concat([predictions, df], axis=1)

    # Color map
    cmap = cm.get_cmap('Set1')

    for x, y in col_pairs:

        # Generate the cluster plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Color the points based on assigned cluster
        for i, cluster in plot_data.groupby('Cluster'):
            cluster.plot(ax=ax, kind='scatter', x=x, y=y,
                         color=cmap((i) * 1.0 / (len(centers) - 1)),
                         label='Cluster %i' % (i), s=30)

        # Plot centers with indicators
        for i, c in enumerate(centers):
            ax.scatter(x=c[0], y=c[1], color='white', edgecolors='black',
                       alpha=1, linewidth=2, marker='o', s=200)
            ax.scatter(x=c[0], y=c[1], marker='$%d$' % (i), alpha=1, s=100)

        # Set plot title
        ax.set_title("Cluster Learning - Centroids Marked by Number")


def plot_cluster_results_3d(df, preds, centers):

    cols = [col for col in df.columns if col not in const.EXCLUDED_COLS]
    col_pairs = itertools.combinations(cols, 3)
    predictions = pd.DataFrame(preds, columns=['Cluster'])
    plot_data = pd.concat([predictions, df], axis=1)

    # Color map
    cmap = cm.get_cmap('Set1')

    for x, y, z in col_pairs:

        # Generate the cluster plot
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, cluster in plot_data.groupby('Cluster'):
            ax.scatter(cluster[x], cluster[y], cluster[z],
                       c=cmap((i) * 1.0 / (len(centers) - 1)))

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)

        # Plot centers with indicators
#        for i, c in enumerate(centers):
#            ax.scatter(c[0], c[1], c[2], color='white',
#                       edgecolors='black', alpha=1, linewidth=2,
#                       marker='o', s=200)
#            ax.scatter(c[0], c[1], c[2], marker='$%d$' % (i),
#                       alpha=1, s=100)

        # Set plot title
        ax.set_title("Cluster Learning - Centroids Marked by Number")
