# babylm 2.0 if needed
DIR=data/corpora/babylm2
mkdir -p $DIR
wget https://osf.io/download/rduj2/ -O $DIR/train_100M.zip
unzip $DIR/train_100M.zip -d $DIR
rm $DIR/train_100M.zip
rm $DIR/train_100M/childes.train
rm $DIR/train_100M/simple_wiki.train
rm $DIR/train_100M/bnc_spoken.train
rm $DIR/train_100M/switchboard.train


wget https://osf.io/download/m48ed/ -O $DIR/dev.zip
unzip $DIR/dev.zip -d $DIR
rm $DIR/dev.zip
rm $DIR/dev/childes.dev
rm $DIR/dev/simple_wiki.dev
rm $DIR/dev/bnc_spoken.dev
rm $DIR/dev/switchboard.dev


wget https://osf.io/download/qj4a6/ -O $DIR/test.zip
unzip $DIR/test.zip -d $DIR
rm $DIR/test.zip
rm $DIR/test/childes.test
rm $DIR/test/simple_wiki.test
rm $DIR/test/bnc_spoken.test
rm $DIR/test/switchboard.test

#process gutenberg books separately for more natural linebreaks
python src/gutenberg_tokenize.py --source $DIR/train_100M/gutenberg.train --target $DIR/train_100M/gutenberg_sentences.train
rm $DIR/train_100M/gutenberg.train

python src/gutenberg_tokenize.py --source $DIR/dev/gutenberg.dev --target $DIR/dev/gutenberg_sentences.train
rm $DIR/dev/gutenberg.dev

python src/gutenberg_tokenize.py --source $DIR/test/gutenberg.test --target $DIR/test/gutenberg_sentences.train
rm $DIR/test/gutenberg.test

# concatenate all files within each dir
cat $DIR/train_100M/* > $DIR/train_100M.txt
cat $DIR/dev/* > $DIR/dev.txt
cat $DIR/test/* > $DIR/test.txt

# sentence-tokenize all files using src/sentence_tokenize.py
python src/sentence_tokenize.py --source $DIR/train_100M.txt --target $DIR/train.sents&
python src/sentence_tokenize.py --source $DIR/dev.txt --target $DIR/dev.sents&
python src/sentence_tokenize.py --source $DIR/test.txt --target $DIR/test.sents

# remove the original dirs
rm -r $DIR/train_100M
rm -r $DIR/dev
rm -r $DIR/test

rm $DIR/train_100M.txt
rm $DIR/dev.txt
rm $DIR/test.txt

