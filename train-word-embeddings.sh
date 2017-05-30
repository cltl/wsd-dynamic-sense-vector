python3 -u process-gigaword.py > output/gigaword.txt 2> output/gigaword.log && \
    python3 -u train-word-embeddings.py output/gigaword.txt output/gigaword