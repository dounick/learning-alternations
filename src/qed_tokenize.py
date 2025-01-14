'''Sentence tokenizes the input corpus.'''
import argparse
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def main(args):
    source = args.source
    target = args.target

    with open(source, "r") as f:
        content = f.read()

    # Replace all newlines with spaces to handle wrapped sentences
    # note: problematic, since these are subtitles for videos; removing newlines actually forms huge text blocks
    # content = content.replace('\n', '')

    sentences = []
    for sentence in tqdm(sent_tokenize(content)):
        if any(char.isalnum() for char in sentence):
            sentences.append(sentence.strip())  # Remove any initial tabs/spaces

    with open(target, "w") as f:
        for sent in sentences:
            f.write(sent.replace("_", "") + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize sentences from raw text files"
    )
    parser.add_argument("--source", type=str, help="source file", required=True)
    parser.add_argument("--target", type=str, help="output file", required=True)
    args = parser.parse_args()
    main(args)
