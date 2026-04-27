import os
from dotenv import load_dotenv

set_15 = [
    "aatrox", "ahri", "akali", "ashe", "braum", "caitlyn", "darius", "dr. mundo", "ekko", "ezreal",
    "gangplank", "garen", "gnar", "gwen", "janna", "jarvan iv", "jayce", "jhin", "jinx", "kai'sa",
    "kalista", "karma", "katarina", "kayle", "kennen", "kobuko", "kog'maw", "k'sante", "lee sin",
    "leona", "lucian", "lulu", "lux", "malphite", "malzahar", "naafiri", "neeko", "poppy", "rakan",
    "rammus", "rell", "ryze", "samira", "senna", "seraphine", "sett", "shen", "sivir", "smolder",
    "swain", "syndra", "twisted fate", "udyr", "varus", "vi", "viego", "volibear", "xayah",
    "xin zhao", "yasuo", "yone", "yuumi", "zac", "ziggs", "zyra"
]

for champ in set_15:
    folder_name = f"FeatureMatch/champs/{champ.replace(' ', '_')}"
    try:
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")