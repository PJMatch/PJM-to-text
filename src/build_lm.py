import csv
import yaml


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ANN_TRAIN = config["data"]["train_ann"]
    print("Extracting text corpus and building lexicon")
    unique_glosses = set()
    sentences = []
    with open(ANN_TRAIN, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            sentence = row["orth"].strip()

            if not sentence:
                continue
            sentences.append(sentence)
            for gloss in sentence.split():
                unique_glosses.add(gloss)

    with open("corpus.txt", "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    with open("lexicon.txt", "w", encoding="utf-8") as f:
        for gloss in sorted(unique_glosses):
            f.write(f"{gloss} {gloss}\n")

    print(f"Wrote {len(sentences)} sentences to corpus.txt")
    print(f"Wrote {len(unique_glosses)} unique glosses to lexicon.txt")


if __name__ == "__main__":
    main()
