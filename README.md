This is a fork of https://github.com/sherjilozair/char-rnn-tensorflow modified so the trained models can easily be used in other environments (e.g. C++, [openFrameworks](http://openframeworks.cc/), [ofxMSATensorFlow](https://github.com/memo/ofxMSATensorFlow) etc.)

General problems and motivations for the mods:

* The default format that trained tensorflow models are saved in are checkpoint files, which don't contain architecture information, only parameter values, so loading them alone in C++ wouldn't be enough (AFAIK there is no C++ loader for ckpt files anyway).
* The file format which contains architecture information and can be loaded in C++ is protobuf (.pb), however saving a .pb from tensorflow saves the *untrained* model only, i.e. the trained model parameters are *not* saved in this file. There is a [utility](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) which prunes and 'freezes' graphs (replaces variables with consts of the same value), however...
* ... when we use *loaded* graphs, we need to access ops to feed and fetch by name. Most python tensorflow examples however, build the graph in python, save references to ops in variables, then during inference they access ops to feed and fetch via those variables, so the graphs often aren't setup to be easily accessible by name. 

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
