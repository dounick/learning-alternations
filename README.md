# learning-alternations

[add what this project is about]

## Data

Run the following to download and sentence tokenize the babylm dataset:

```bash
bash data/get_babylm.sh
```

This will create the following directory structure:

```
├── corpora
│   └── babylm
│       └── train_100M.sents
```

## Automatic Dative Detection

To automatically detect DOs and PPs, run the following code (we have passed in a test corpus that we created, please change the argument accordingly, and make sure the `--dative_path` argument is named appropriately to be organized)

```bash
python src/detect_datives.py --corpus_path data/corpora/test-corpus/corpus.sents --dative_path data/datives/test-corpus/ --batch_size 4
```

For babylm, you can use the following (after running the data download script) -- this has also been conveniently written in `scripts/detect_dative.sh`:

```bash
python src/detect_datives.py \
    --corpus_path data/corpora/babylm/train_100M.sents \
    --dative_path data/datives/babylm \
    --batch_size 8192
```

Note: to run the above code, you might need to use spacy that works with an older version of `transformers` -- you can do it by creating a separate environment just for spacy-operations. For convenience, we have created an example environment in `spacy_environment.yml`. To install this, run:

```bash
# creates an environment named "spacy"
# feel free to change this in the file.
conda env create -f spacy_environment.yml 

# activate it
conda activate spacy
```

But after that, make sure you run code on a separate environment with the latest transformers version.
