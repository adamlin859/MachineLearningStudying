import tensorflow as tf 
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.reset_default_graph()
word_ids_left = tf.placeholder(tf.float32, shape=[None, maxlen, word_emb_size])
word_ids_mid = tf.placeholder(tf.float32, shape=[None, maxlen, word_emb_size])
word_ids_left = tf.placeholder(tf.float32, shape=[None, maxlen, word_emb_size])