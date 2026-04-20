import pandas as pd
import numpy as np
import re, json, os
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data" / "raw"
OUT = BASE_DIR / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

LABEL_COLS = [
    "label_toxic","label_severe","label_obscene",
    "label_threat","label_insult","label_identity_hate","label_offensive"
]

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z0-9\u0900-\u097F\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# LANGUAGE DETECTION
# =========================
def detect_lang(text):
    has_hi = bool(re.search(r"[\u0900-\u097F]", text))
    has_en = bool(re.search(r"[a-zA-Z]", text))
    if has_hi and has_en:
        return "hi-en"
    elif has_hi:
        return "hi"
    return "en"

# =========================
# SPAN SEEDS
# =========================
SPAN_SEEDS = {
    "threat": [r"\bkill\b", r"\bmar\b", r"\bshoot\b"],
    "identity_hate": [r"\bmuslim\b", r"\bhindu\b", r"\bkafir\b"],
    "insult": [r"\bstupid\b", r"\bidiot\b", r"\bchutiya\b"],
    "obscene": [r"\bfuck\b", r"\bshit\b", r"\bgand\b"],
}

def extract_spans(text):
    spans = []
    for patterns in SPAN_SEEDS.values():
        for pat in patterns:
            for m in re.finditer(pat, text.lower()):
                spans.append([m.start(), m.end()])
    return spans

# =========================
# PROCESS JIGSAW
# =========================
def process_jigsaw():
    df = pd.read_csv(RAW / "train.csv")

    df["text"] = df["comment_text"].apply(clean_text)
    df["lang"] = df["text"].apply(detect_lang)

    df["label_toxic"] = df["toxic"]
    df["label_severe"] = df["severe_toxic"]
    df["label_obscene"] = df["obscene"]
    df["label_threat"] = df["threat"]
    df["label_insult"] = df["insult"]
    df["label_identity_hate"] = df["identity_hate"]
    df["label_offensive"] = df[["toxic","obscene","insult"]].max(axis=1)

    return df[["id","text","lang"] + LABEL_COLS]

# =========================
# PROCESS HASOC
# =========================
def process_hasoc(file, tag):
    df = pd.read_csv(file)

    df["text"] = df["text"].apply(clean_text)
    df["lang"] = df["text"].apply(detect_lang)

    df["label_toxic"] = (df["task_1"] == "HOF").astype(int)
    df["label_identity_hate"] = (df["task_2"] == "HATE").astype(int)
    df["label_offensive"] = df["task_2"].isin(["OFFN","PRFN"]).astype(int)

    df["label_insult"] = (df["task_2"] == "OFFN").astype(int)
    df["label_obscene"] = (df["task_2"] == "PRFN").astype(int)

    df["label_threat"] = 0
    df["label_severe"] = df["label_identity_hate"]

    df["id"] = tag + "_" + df.index.astype(str)

    return df[["id","text","lang"] + LABEL_COLS]

# =========================
# OCR MERGE
# =========================
def load_ocr():
    ocr_dict = {}
    path = RAW / "img_txt"

    if not path.exists():
        return ocr_dict

    for file in os.listdir(path):
        if file.endswith(".json"):
            with open(path / file, encoding="utf-8") as f:
                data = json.load(f)
                ocr_dict[file.replace(".json","")] = data.get("img_text","")

    return ocr_dict

# =========================
# MAIN
# =========================
def main():

    df1 = process_jigsaw()
    df2 = process_hasoc(RAW / "english_2021.csv","en")
    df3 = process_hasoc(RAW / "hindi_2021.csv","hi")

    df = pd.concat([df1, df2, df3], ignore_index=True)

    # OCR
    ocr = load_ocr()
    df["text"] = df["text"] + " " + df["id"].astype(str).map(ocr).fillna("")

    # spans
    df["spans"] = df["text"].apply(lambda x: json.dumps(extract_spans(x)))

    # split
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv(OUT/"train.csv", index=False)
    val.to_csv(OUT/"val.csv", index=False)
    test.to_csv(OUT/"test.csv", index=False)

    print("✅ Preprocessing complete")

if __name__ == "__main__":
    main()