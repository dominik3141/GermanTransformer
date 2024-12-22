"""
Preprocesses the data by extracting nouns and their articles from the long CSV file and writing them to a new, more manageable CSV file.
"""

from nouns import Noun
import csv


def read_long_csv(file_path: str) -> list[Noun]:
    """Reads German nouns from a CSV file and converts them to Noun objects"""

    nouns: list[Noun] = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip entries that don't represent full words (like suffixes/affixes)
            if row["lemma"].startswith("-"):
                continue

            # Extract the genus (article) - can be 'm', 'f', or 'n'
            genus = row["genus"].lower()
            if genus in ["m", "f", "n"]:
                nouns.append(Noun(row["lemma"], genus))

    return nouns


def write_nouns_to_csv(nouns: list[Noun], file_path: str):
    """Writes a list of Noun objects to a CSV file"""

    with open(file_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "article"])
        for noun in nouns:
            writer.writerow([noun.word, noun.article])


def main():
    nouns = read_long_csv("data/nouns.csv")
    write_nouns_to_csv(nouns, "data/nouns_clean.csv")


if __name__ == "__main__":
    main()
