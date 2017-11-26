import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
from tqdm import tqdm

import core.const as const


def remove_spanish_chars(s):
    for accent, no_accent in const.CHARACTER_REMOVE.items():
        s = s.replace(accent, no_accent)

    return s


def process_field(row, field):

    try:
        value = row[field]
    except KeyError:
        value = 0

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
            value = False
        elif value == "Totalmente amueblado y equipado":
            value = True
        else:
            value = False

    if field == "garage":
        if not isinstance(value, str):
            value = False
        elif " 0 eur/mes" in value or value == ("Plaza de garaje incluida en "
                                                "el precio"):
            value = True
        else:
            value = False

    if field == "kitchen":
        value = row["furniture"]
        if not isinstance(value, str):
            value = False
        elif value == "Totalmente amueblado y equipado":
            value = True
        elif value == "Cocina equipada y casa sin amueblar":
            value = True
        elif value == "Cocina sin equipar y casa sin amueblar":
            value = False

    if field == "price_per_sm":
        price = row["price"]
        size = row["size"]
        value = price / size

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
    df["energy"] = df.apply(process_field, axis=1, args=("energy",))
    df["district"] = df.apply(process_field, axis=1, args=("district",))
    df["floor"] = df.apply(process_field, axis=1, args=("floor",))
    df["hasGarage"] = df.apply(process_field, axis=1, args=("garage",))
    df["rooms"] = df.apply(process_field, axis=1, args=("rooms",))
    df["status"] = df.apply(process_field, axis=1, args=("status",))
    df["hasFurniture"] = df.apply(process_field, axis=1, args=("furniture",))
    df["price_per_sm"] = df.apply(process_field, axis=1,
                                  args=("price_per_sm",))
    df["hasEquippedKitchen"] = df.apply(process_field, axis=1,
                                        args=("kitchen",))

    orientation_list = df["orientation"].tolist()
    orientations_dict = {"north_oriented": [], "east_oriented": [],
                         "west_oriented": [], "south_oriented": []}

    for orientation in orientation_list:
        orientations = get_orientations(orientation)
        orientations_dict["north_oriented"].append(orientations[0])
        orientations_dict["east_oriented"].append(orientations[1])
        orientations_dict["west_oriented"].append(orientations[2])
        orientations_dict["south_oriented"].append(orientations[3])

    for key, values in orientations_dict.items():
        df[key] = values

    if file_name_out:
        path_out = "/".join(file_path.split("/")[: -1]) + "/"
        df[const.OUT_COLS].to_excel(path_out + file_name_out)

    return df[const.OUT_COLS]


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # return x
    return np.expand_dims(x, axis=0)


def load_imgs(data_path, model=None, verbose=0):
    out_imgs = []

    folders = os.listdir(data_path)
    for folder in tqdm(folders):
        imgs_path = data_path + "/" + folder
        imgs = os.listdir(imgs_path)
        imgs_list = [img for img in imgs if img.endswith(".jpg")]
        for img in imgs_list:
            img_path = imgs_path + "/" + img
            label = img.split("_")[0]
            try:
                tensor = path_to_tensor(os.path.abspath(img_path))
                if model:
                    tensor = model.predict(tensor)
                record = [int(folder), label, tensor]
                out_imgs.append(record)
            except OSError:
                if verbose:
                    print("Picture '{}' is corrupt".format(img_path))
                else:
                    pass

    return out_imgs
