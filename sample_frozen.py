# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:25:16 2017

@author: memo

tests frozen graph
"""


from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import argparse
import os
from six.moves import cPickle
from six import text_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                       help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--model_dir', type=str, default='./save',
                       help='path to frozen graph pb file')

    args = parser.parse_args()

    with open(os.path.join(args.model_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
        
    with tf.Session() as sess:
        # load frozen graph
        with gfile.FastGFile(os.path.join(args.model_dir, 'graph_frz.pb'),'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')        
        
        print(sample(sess, chars, vocab, args.n, args.prime, args.sample))




def sample(sess, chars, vocab, num=200, prime='The ', sampling_type=1):
    data_in = 'data_in:0'
    data_out = 'data_out:0'
    state_in = 'state_in:0'
    state_out = 'state_out:0'
    
    state = sess.run(state_in)
    for char in prime[:-1]:
        x = np.zeros((1, 1))
        x[0, 0] = vocab[char]
        feed = {data_in: x, state_in:state}
        [state] = sess.run([state_out], feed)

    def weighted_pick(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        return(int(np.searchsorted(t, np.random.rand(1)*s)))

    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.zeros((1, 1))
        x[0, 0] = vocab[char]
        feed = {data_in: x, state_in:state}
        [probs, state] = sess.run([data_out, state_out], feed)
        p = probs[0]

        if sampling_type == 0:
            sample = np.argmax(p)
        elif sampling_type == 2:
            if char == ' ':
                sample = weighted_pick(p)
            else:
                sample = np.argmax(p)
        else: # sampling_type == 1 default:
            sample = weighted_pick(p)

        pred = chars[sample]
        ret += pred
        char = pred
    return ret


            
if __name__ == '__main__':
    main()
