from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

import core.const as const


def select_features(df, y_col):
    r_s = const.RANDOM_STATE

    features = [col for col in df.columns if col not in const.EXCLUDED_COLS]
    features.remove(y_col)

    X = df[features]
    y = df[y_col].tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=r_s)

    clf = RandomForestRegressor(random_state=r_s)
    clf = clf.fit(X_train, Y_train)

    score = clf.score(X_test, Y_test)
    feat_importances = clf.feature_importances_

    feat_list = [(i, f) for (f, i) in zip(features, feat_importances)]

    dict_out = {"score": score,
                "feat_importances": feat_list}

    return dict_out


def df_groupby_mean(df, key, value):
    return df.groupby(key)[value].mean()


def get_avg_price_area(df):
    avg_price = df_groupby_mean(df, "district_clean", "price_area").to_dict()
    df["avg_price_area"] = df.apply(lambda x: avg_price[x["district_clean"]],
                                    axis=1)

    return df


def scale_data(df):
    num_f = list(df.select_dtypes(include=['int64']).columns)
    num_f = [col for col in num_f if col not in const.EXCLUDED_COLS]

    for column in num_f:
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(np.array(df[column]).reshape(-1, 1))

    return df


def remove_outliers(df, verbose=0):
    # Dictionary to count the number of times a row has an outlier in a
    # feature. {Row: count of features with outliers}
    outliers_count = {}

    # For each feature find the data points with extreme high or low values
    for feature in const.OUTLIER_COLS:

        # Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(df[feature], 25)

        # Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(df[feature], 75)

        # Use the interquartile range to calculate an outlier step
        # (1.5 times the interquartile range)
        step = 1.5 * (Q3 - Q1)

        # Get indexes of records with outliers in 'feature'
        outliers_row = df[~((df[feature] >= Q1 - step) &
                            (df[feature] <= Q3 + step))].index.tolist()

        # Display the outliers
        if verbose:
            print(("Data points considered outliers for the feature"
                   " '{0}': {1}").format(feature, len(outliers_row)))

        # Populate dictionary. If the index exists, add one to the count.
        # Else, initialize with count = 1
        for row in outliers_row:
            try:
                outliers_count[row] = outliers_count[row] + 1
            except KeyError:
                outliers_count[row] = 1

    # Create a dataframe with the indexes, then filter by Count > 0
    df_outliers_count = pd.DataFrame.from_dict(outliers_count, orient='index')
    df_outliers_count.columns = ['Count']
    outliers_idx = df_outliers_count[df_outliers_count['Count'] > 0]

    # Select rows with outliers
    outliers = sorted(outliers_idx.index.tolist())

    # Remove the outliers, if any were specified
    return df.drop(outliers)
