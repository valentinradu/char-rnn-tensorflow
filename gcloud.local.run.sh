gcloud ml-engine local train \
    --module-name char_rnn.train \
    --package-path ./char_rnn \
    -- \
    --data_dir ./data \
    --save_dir ./save
