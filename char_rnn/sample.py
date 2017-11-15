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
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                       help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--freeze_graph', dest='freeze_graph', action='store_true',
                       help='if true, freeze (replace variables with consts), prune (for inference) and save graph')

    args = parser.parse_args()
    sample(args)


def freeze_and_save_graph(sess, folder, out_nodes, as_text=False):
    ## save graph definition
    graph_raw = sess.graph_def
    graph_frz = tf.graph_util.convert_variables_to_constants(sess, graph_raw, out_nodes)
    ext = '.txt' if as_text else '.pb'
    #tf.train.write_graph(graph_raw, folder, 'graph_raw'+ext, as_text=as_text)
    tf.train.write_graph(graph_frz, folder, 'graph_frz'+ext, as_text=as_text)
    

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
        
    # save character <-> index map as text file for easy loading in other apps    
    with open(os.path.join(args.save_dir, 'chars.txt'), 'w') as f:
        for c in chars:
            f.write("%s\n" % ord(c))

    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            if(args.freeze_graph):
                freeze_and_save_graph(sess, args.save_dir, ['data_out', 'state_out'], False)
            print(model.sample(sess, chars, vocab, args.n, args.prime, args.sample))

if __name__ == '__main__':
    main()
