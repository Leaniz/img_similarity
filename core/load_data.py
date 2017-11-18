import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import core.const as const


def remove_spanish_chars(s):
    for accent, no_accent in const.CHARACTER_REMOVE.items():
        s = s.replace(accent, no_accent)

    return s


def process_field(row, field):

    value = row[field]

    if field == "energy":
        if "indicado" in value:
            value = "not_stated"
        elif "año" in value:
            value = "stated"
        elif "exento" in value:
            value = "exempt"
        elif "trámite" in value:
            value = "in_progress"

    if field == "district":
        value = value.replace("Distrito", "").strip().lower().replace(" ", "_")
        value = remove_spanish_chars(value)

    if field == "floor":
        try:
            int_val = int(value)
            if int_val >= 10:
                value = "10+"
        except ValueError:
            if isinstance(value, str):
                if "chalet" in value.lower():
                    value = "chalet"
                elif value == "Bajo":
                    value = "0"
                elif value == "Semi-sótano":
                    value = "basement"
                elif value == "Entreplanta":
                    value = "mid_floor"
            else:
                value = "unknown"

    if field == "furniture":
        if not isinstance(value, str):
            value = "unknown"
        elif value == "Totalmente amueblado y equipado":
            value = "kitchen_yes_furniture_yes"
        elif value == "Cocina equipada y casa sin amueblar":
            value = "kitchen_yes_furniture_no"
        elif value == "Cocina sin equipar y casa sin amueblar":
            value = "kitchen_no_furniture_no"

    if field == "garage":
        if not isinstance(value, str):
            value = "no_garage"
        elif " 0 eur/mes" in value or value == ("Plaza de garaje incluida en "
                                                "el precio"):
            value = "garage_included"
        else:
            value = "garage_extra_cost"

    if field == "rooms":
        if value == "Sin":
            value = 0
        else:
            value = int(value)

    if field == "size_plot":
        if np.isnan(value):
            value = 0

    if field == "status":
        if not isinstance(value, str):
            value = "unknown"
        elif value == "Segunda mano/buen estado":
            value = "second_hand_good"
        elif value == "Segunda mano/para reformar":
            value = "second_hand_bad"

    return value


def get_orientations(orientation):
    north = False
    east = False
    west = False
    south = False

    if isinstance(orientation, str):
        if "norte" in orientation:
            north = True
        if "este" in orientation:
            east = True
        if "oeste" in orientation:
            west = True
        if "sur" in orientation:
            south = True

    return north, east, west, south


def df_groupby(df, key):
    return df.groupby(key).size().reset_index().sort_values(key)


def clean_support_data(file_path, file_name_out=None):

    if not file_path.endswith(".xlsx"):
        print("The file must be an excel (xlsx) file")
        return

    df = pd.read_excel(file_path)
    df["energy_clean"] = df.apply(process_field, axis=1, args=("energy",))
    df["district_clean"] = df.apply(process_field, axis=1, args=("district",))
    df["floor_clean"] = df.apply(process_field, axis=1, args=("floor",))
    df["garage_clean"] = df.apply(process_field, axis=1, args=("garage",))
    df["rooms_clean"] = df.apply(process_field, axis=1, args=("rooms",))
    df["status_clean"] = df.apply(process_field, axis=1, args=("status",))
    df["furniture_clean"] = df.apply(process_field, axis=1,
                                     args=("furniture",))

    orientation_list = df["orientation"].tolist()
    orientations_dict = {"north": [], "east": [], "west": [], "south": []}

    for orientation in orientation_list:
        orientations = get_orientations(orientation)
        orientations_dict["north"].append(orientations[0])
        orientations_dict["east"].append(orientations[1])
        orientations_dict["west"].append(orientations[2])
        orientations_dict["south"].append(orientations[3])

    for key, values in orientations_dict.items():
        df[key] = values

    if file_name_out:
        path_out = "/".join(file_path.split("/")[: -1]) + "/"
        df[const.OUT_COLS].to_excel(path_out + file_name_out)

    return df[const.OUT_COLS]


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


def scale_data(df):
    num_f = list(df.select_dtypes(include=['int64']).columns)
    num_f = [col for col in num_f if col not in const.EXCLUDED_COLS]

    for column in num_f:
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(np.array(df[column]).reshape(-1, 1))

    return df
