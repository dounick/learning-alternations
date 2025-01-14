export CUDA_VISIBLE_DEVICES=0
python src/detect_datives_phrasal.py \
    --corpus_path data/corpora/babylm/test.sents \
    --dative_path data/datives/test \
    --batch_size 8192