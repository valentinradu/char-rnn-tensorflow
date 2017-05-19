This is a fork of https://github.com/sherjilozair/char-rnn-tensorflow with modifications to enable the trained models to be used in other environments (e.g. [ofxMSATensorFlow](https://github.com/memo/ofxMSATensorFlow)). Reasons as to why these changes are nessecary are described [here](https://github.com/memo/ofxMSATensorFlow/wiki/Loading-and-using-trained-tensorflow-models-in-openFrameworks).

After training, run:

[sample.py](https://github.com/memo/char-rnn-tensorflow/blob/master/sample.py) with the `--freeze_graph` argument to prune, freeze and save the graph as a binary protobuf to be loaded in C++ (removing unnessecary nodes used in training, and replacing variables with consts). It also saves the character-index map as a text file.

[sample_frozen.py](https://github.com/memo/char-rnn-tensorflow/blob/master/test_frozen.py) demonstrates inference with the frozen graph from python. It also works in [C++/openFrameworks](https://github.com/memo/ofxMSATensorFlow/blob/master/example-char-rnn/src/example-char-rnn.cpp). 

---
# char-rnn-tensorflow
Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow 1.0](http://www.tensorflow.org)

# Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`.

To sample from a checkpointed model, `python sample.py`.
# Roadmap
- Add explanatory comments
- Expose more command-line arguments
- Compare accuracy and performance with char-rnn
