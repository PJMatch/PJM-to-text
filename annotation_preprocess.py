import re
from collections import Counter

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def clean_text(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def canonicalize_gloss(gloss: str) -> str:
    gloss = clean_text(gloss).upper()
    gloss = gloss.replace("’", "'").replace("`", "'")
    gloss = re.sub(r"[.,!?;:\"()\[\]{}]", "", gloss)

    token_map = {
        "UDDER": "RUDDER",
    }

    prefixes_to_strip = ["x-", "X-", "DESC-", "IX-", "INDEX-", "CL-"]

    cleaned_tokens = []
    for token in gloss.split():
        token = token.strip("-")

        for prefix in prefixes_to_strip:
            if token.lower().startswith(prefix.lower()) and len(token) > len(prefix):
                token = token[len(prefix):]
                break

        token = token_map.get(token, token)
        token = re.sub(r"[^A-Z0-9\-]", "", token)
        token = token.strip("-")

        if token:
            cleaned_tokens.append(token)

    deduped_tokens = []
    for token in cleaned_tokens:
        if not deduped_tokens or deduped_tokens[-1] != token:
            deduped_tokens.append(token)

    return " ".join(deduped_tokens)


def translate_batch(sentences, tokenizer, model, device):
    max_source_length = 256
    max_target_length = 128

    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_target_length,
            num_beams=1,
            do_sample=False,
            early_stopping=True,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def build_vocab_table(gloss_series: pd.Series) -> pd.DataFrame:
    counter = Counter()

    for gloss in gloss_series.fillna(""):
        for token in str(gloss).split():
            if token:
                counter[token] += 1

    vocab_df = pd.DataFrame(
        [{"token": token, "count": count} for token, count in counter.items()]
    ).sort_values(["count", "token"], ascending=[False, True])

    vocab_df["rank"] = range(1, len(vocab_df) + 1)
    vocab_df = vocab_df[["rank", "token", "count"]]
    return vocab_df


def main():
    input_csv = "how2sign_val.csv"
    output_csv = "how2sign_val_gloss.csv"
    vocab_csv = "how2sign_val_gloss_vocab.csv"
    model_name = "AchrafAzzaouiRiceU/t5-english-to-asl-gloss"
    batch_size = 32


    df = pd.read_csv("how2sign_test.csv", sep="\t")

    df["SENTENCE"] = df["SENTENCE"].fillna("").map(clean_text)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    raw_glosses = []
    sentences = df["SENTENCE"].tolist()

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        predictions = translate_batch(batch, tokenizer, model, device)
        raw_glosses.extend(clean_text(pred) for pred in predictions)

    df["ASL_GLOSS_RAW"] = raw_glosses
    df["ASL_GLOSS"] = df["ASL_GLOSS_RAW"].map(canonicalize_gloss)

    vocab_df = build_vocab_table(df["ASL_GLOSS"])

    df.to_csv(output_csv, index=False)
    vocab_df.to_csv(vocab_csv, index=False)

    print(f"Saved converted file: {output_csv}")
    print(f"Saved vocabulary file: {vocab_csv}")
    print(f"Rows: {len(df)} | Vocab size: {len(vocab_df)}")


if __name__ == "__main__":
    main()
