# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 19:43:22 2017

@author: memo

freezes and saves graph
also saves vocabulary in text format
"""


from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

from six import text_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                       help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--out_dir', type=str, default='./frozen',
                       help='directory to store graph pb file')


    args = parser.parse_args()

    with open(os.path.join(args.ckpt_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.ckpt_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
        
    with open(os.path.join(args.out_dir, 'chars.txt'), 'w') as f:
        for c in chars:
            f.write("%s\n" % ord(c))
        
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)


            # only get required portion of graph and convert variables to constants
            graph_raw = sess.graph_def
            graph_frz = tf.graph_util.convert_variables_to_constants(sess, graph_raw, ['data_out', 'state_out'])
            
            ## save graph definition
#            as_text = False;
            for as_text in [True, False]:
                ext = '.txt' if as_text else '.pb'
                tf.train.write_graph(graph_raw, args.out_dir, 'graph_raw'+ext, as_text=as_text)
                tf.train.write_graph(graph_frz, args.out_dir, 'graph_frz'+ext, as_text=as_text)

            print(model.sample(sess, chars, vocab, args.n, args.prime, args.sample))

            
if __name__ == '__main__':
    main()
