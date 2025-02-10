import argparse
import json

from ragatouille import RAGPretrainedModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Index Wikipedia articles for FEVER")
    parser.add_argument(
        "--in-files",
        type=str,
        nargs="+",
        help="Input files with Wikipedia articles from the FEVER dataset that need to be indexed",
    )
    return parser.parse_args()


def load_wiki_data(in_files):
    texts = []
    ids = []

    for file in in_files:
        with open(file) as f:
            for line in f:
                entry = json.loads(line)
                if entry["id"]:
                    texts.append(entry["text"])
                    ids.append(entry["id"])
    return texts, ids


def main():
    args = parse_arguments()
    texts, ids = load_wiki_data(args.in_files)

    model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    index_path = model.index(
        collection=texts,
        document_ids=ids,
        index_name="fever-articles-index",
    )
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()
