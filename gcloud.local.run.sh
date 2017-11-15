gcloud ml-engine local train \
    --module-name word_rnn.train \
    --package-path ./word_rnn \
    -- \
    --data_dir ./data/rnn-c-data \
    --log_dir ./logs \
    --save_dir ./save
