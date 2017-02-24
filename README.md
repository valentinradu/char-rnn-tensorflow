This is a fork of https://github.com/sherjilozair/char-rnn-tensorflow modified so I can use the trained model easily in C++.

TensorFlow used to manage RNN states with a single concatenated tensor (for all cells, both hidden and LSTM cell state). But that behaviour is being deprecated in favour of using a tuple of tensors (for each layer, cell, batch size etc). This makes it more complicated to manage from C++. In this version, I add nodes to stack and unstack all states to and from a single tensor which I can easily reference from C++. Also the graph is a bit more organised so other key nodes can easily be referenced by name.

Also, after training, run 

[save_graph.py](https://github.com/memo/char-rnn-tensorflow/blob/master/save_graph.py) to freeze and saves the graph as a protobuf to be loaded in C++ (removing unnessecary nodes used in training, and replacing variables with consts). It also saves the character-index map as a text file.

[test_frozen.py](https://github.com/memo/char-rnn-tensorflow/blob/master/test_frozen.py) demonstrates using the frozen graph from python. It also works in [C++/openFrameworks](https://github.com/memo/ofxMSATensorFlow/blob/master/example-char-rnn/src/example-char-rnn.cpp). 

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
