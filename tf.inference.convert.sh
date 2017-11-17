PWD=$(pwd)
INPUT_FILE=$PWD/remote_save/latest/graph_raw.pb
OUTPUT_FILE=$PWD/remote_save/latest/graph_mobile.pb
CHECKPOINT_FILE=$PWD/remote_save/latest/model.ckpt-2047
cd ~/Playgrounds/tensorflow
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=$INPUT_FILE \
--input_checkpoint=$CHECKPOINT_FILE \
--output_node_names=data_out,state_out \
--input_binary \
--output_graph=$OUTPUT_FILE
#bazel-bin/tensorflow/python/tools/optimize_for_inference \
#--input=/tmp/graph_frozen.pb \
#--output=$OUTPUT_FILE \
#--input_names=data_in,state_in \
#--output_names=data_out,state_out \
#--frozen_graph=True
#bazel --bazelrc=/dev/null run --config=opt \
#//tensorflow/contrib/lite/toco:toco -- \
#--input_file=/tmp/inference.pb \
#--output_file=$OUTPUT_FILE \
#--input_format=TENSORFLOW_GRAPHDEF \
#--output_format=TFLITE \
#--input_type=FLOAT \
#--inference_type=FLOAT \
#--input_shapes=1,128:1,50,50 \
#--input_arrays=state_in,data_in \
#--output_arrays=state_out,data_out
cd $PWD
#batch_size, #seq_length for input shape, data_in/out is int32, state_in/out is double

