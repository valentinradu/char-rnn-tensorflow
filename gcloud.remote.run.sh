export BUCKET_NAME=char-rnn-data
export JOB_NAME="train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=europe-west1

gcloud ml-engine jobs submit training $JOB_NAME\
    --job-dir gs://$BUCKET_NAME/$JOB_NAME \
    --runtime-version 1.0 \
    --module-name char_rnn.train \
    --package-path ./char_rnn \
    --region $REGION \
    --config cloudml-gpu.yaml \
    -- \
    --data_dir gs://$BUCKET_NAME/data_test \
    --save_dir gs://$BUCKET_NAME/save \
    --batch_size 50 \
    --seq_length 50 \
    --save_every 5000 \
    --learning_rate 0.008 \
    --num_epochs 5
    
