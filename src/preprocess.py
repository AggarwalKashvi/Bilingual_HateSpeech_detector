import os
import json
import pandas as pd
import re

# =========================
# PROJECT ROOT
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

TRAIN_CSV = os.path.join(DATA_RAW, "train.csv")
ENG_CSV = os.path.join(DATA_RAW, "english_2021.csv")
HIN_CSV = os.path.join(DATA_RAW, "hindi_2021.csv")
OCR_FOLDER = os.path.join(DATA_RAW, "img_txt")

SAVE_PATH = os.path.join(DATA_PROCESSED, "train_final.csv")


# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\u0900-\u097F\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# LOAD OCR JSON
# =========================
def load_ocr(folder_path):
    ocr_data = {}

    if not os.path.exists(folder_path):
        print("⚠️ OCR folder not found, skipping...")
        return ocr_data

    files = os.listdir(folder_path)

    for i, file in enumerate(files):
        if file.endswith(".json"):
            try:
                with open(os.path.join(folder_path, file), encoding="utf-8") as f:
                    data = json.load(f)

                img_id = file.replace(".json", "")
                ocr_data[img_id] = data.get("img_text", "")

                if i % 500 == 0:
                    print(f"Loaded {i} OCR files...")

            except Exception:
                continue

    return ocr_data


# =========================
# PROCESS JIGSAW
# =========================
def process_jigsaw(path):
    df = pd.read_csv(path)

    toxic_cols = [
        "toxic", "severe_toxic",
        "obscene", "threat",
        "insult", "identity_hate"
    ]

    df["label"] = df[toxic_cols].max(axis=1)

    df = df[["id", "comment_text", "label"]]
    df.rename(columns={"comment_text": "text"}, inplace=True)

    return df


# =========================
# PROCESS HASOC
# =========================
def process_hasoc(path):
    df = pd.read_csv(path)

    def map_label(x):
        if pd.isna(x):
            return 0
        x = str(x)
        return 1 if ("HOF" in x or "OFFN" in x or "PRFN" in x) else 0

    df["label"] = df["task_1"].apply(map_label)

    return df[["_id", "text", "label"]].rename(columns={"_id": "id"})


# =========================
# MERGE OCR
# =========================
def merge_ocr(df, ocr_dict):
    if not ocr_dict:
        return df

    df["ocr_text"] = df["id"].astype(str).map(ocr_dict).fillna("")
    df["text"] = df["text"] + " " + df["ocr_text"]

    return df.drop(columns=["ocr_text"])


# =========================
# MAIN PIPELINE
# =========================
def run_preprocessing():

    print("📁 BASE DIR:", BASE_DIR)
    print("📂 RAW DATA:", DATA_RAW)

    all_dfs = []

    # JIGSAW
    if os.path.exists(TRAIN_CSV):
        print("✅ Processing Jigsaw...")
        all_dfs.append(process_jigsaw(TRAIN_CSV))

    # HASOC ENGLISH
    if os.path.exists(ENG_CSV):
        print("✅ Processing English HASOC...")
        all_dfs.append(process_hasoc(ENG_CSV))

    # HASOC HINDI
    if os.path.exists(HIN_CSV):
        print("✅ Processing Hindi HASOC...")
        all_dfs.append(process_hasoc(HIN_CSV))

    if not all_dfs:
        print("❌ No datasets found. Check paths.")
        return

    # MERGE
    df = pd.concat(all_dfs, ignore_index=True)

    # OCR
    print("📦 Loading OCR...")
    ocr_dict = load_ocr(OCR_FOLDER)
    df = merge_ocr(df, ocr_dict)

    # CLEAN
    print("🧹 Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    # REMOVE EMPTY
    df = df[df["text"].str.strip() != ""]

    # SAVE
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)

    print(f"\n✅ Saved to: {SAVE_PATH}")
    print(f"📊 Total samples: {len(df)}")
    print(df.head())


# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_preprocessing()