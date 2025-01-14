export CUDA_VISIBLE_DEVICES=2
python src/detect_ditransitive.py \
    --corpus_path data/corpora/babylm/train.sents \
    --dative_path data/datives/ditransitive \
    --batch_size 8192