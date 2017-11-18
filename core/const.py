CHARACTER_REMOVE = {"á": "a",
                    "é": "e",
                    "í": "i",
                    "ó": "o",
                    "ú": "u",
                    "Á": "A",
                    "É": "E",
                    "Í": "I",
                    "Ó": "O",
                    "Ú": "U",
                    "ñ": "n",
                    "Ñ": "N"
                    }

OUT_COLS = ['ID',
            'admitsPets',
            'bathrooms',
            'district_clean',
            'exterior',
            'hasAircon',
            'hasCupboards',
            'hasGarden',
            'hasLift',
            'hasPool',
            'hasStorage',
            'hasTerrace',
            'price',
            'size_const',
            'status_clean',
            'energy_clean',
            'floor_clean',
            'furniture_clean',
            'garage_clean',
            'rooms_clean',
            'north',
            'east',
            'west',
            'south']

OUTLIER_COLS = ["bathrooms", "price", "rooms_clean", "size_const"]

EXCLUDED_COLS = ["ID"]

RANDOM_STATE = 34093458
