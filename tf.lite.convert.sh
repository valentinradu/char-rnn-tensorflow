PWD=$(pwd)
INPUT_FILE=$PWD/remote_save/latest/graph_frz.pb
OUTPUT_FILE=$PWD/remote_save/latest/graph.tflite
cd ~/Playgrounds/tensorflow
bazel --bazelrc=/dev/null run --config=opt \
//tensorflow/contrib/lite/toco:toco -- \
--input_file=$INPUT_FILE \
--output_file=$OUTPUT_FILE \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--input_type=FLOAT \
--inference_type=FLOAT \
--input_shapes=1,128:1,50,50 \
--input_arrays=state_in,data_in \
--output_arrays=state_out,data_out
cd $PWD
#batch_size, #seq_length for input shape, data_in/out is int32, state_in/out is double

