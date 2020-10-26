#Copyright (C) 2018  Seyed Mehran Kazemi, Licensed under the GPL V3; see: <https://www.gnu.org/licenses/gpl-3.0.en.html>
from tensor_factorizer import *
from reader import *

class ED_SimplE(TensorFactorizer):

	def __init__(self, params, dataset="wn18"):
		TensorFactorizer.__init__(self, model_name="ED_SimplE", loss_function="likelihood", params=params, dataset=dataset)

	def setup_weights(self):
		sqrt_size = 6.0 / math.sqrt(self.params.emb_size)
		self.rel_emb_re      = tf.get_variable(name="rel_emb_re",      initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.rel_emb_im      = tf.get_variable(name="rel_emb_im",      initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))

		self.rel_inv_emb_re  = tf.get_variable(name="rel_inv_emb_re",  initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.rel_inv_emb_im  = tf.get_variable(name="rel_inv_emb_im",  initializer=tf.random_uniform(shape=[self.num_rel, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))

		self.ent_head_emb_re = tf.get_variable(name="ent_head_emb_re", initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.ent_head_emb_im = tf.get_variable(name="ent_head_emb_im", initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))

		self.ent_tail_emb_re = tf.get_variable(name="ent_tail_emb_re", initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
		self.ent_tail_emb_im = tf.get_variable(name="ent_tail_emb_im", initializer=tf.random_uniform(shape=[self.num_ent, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))

		self.var_list = [self.rel_emb_re, self.rel_inv_emb_re, self.ent_head_emb_re, self.ent_tail_emb_re,self.rel_emb_im, self.rel_inv_emb_im, self.ent_head_emb_im, self.ent_tail_emb_im]


	def decompress(self,is_training):

		self.rel_emb_re       = tf.layers.dropout(self.rel_emb_re,rate=0.2, training=is_training)
		self.rel_emb_im     = tf.layers.dropout(self.rel_emb_im, rate=0.2, training=is_training)

		self.rel_inv_emb_re = tf.layers.dropout(self.rel_inv_emb_re, rate=0.2,training=is_training)
		self.rel_inv_emb_im = tf.layers.dropout(self.rel_inv_emb_im, rate=0.2, training=is_training)

		self.ent_head_emb_re = tf.layers.dropout(self.ent_head_emb_re,rate=0.2, training=is_training)
		self.ent_head_emb_im = tf.layers.dropout(self.ent_head_emb_im, rate=0.2, training=is_training)

		self.ent_tail_emb_re = tf.layers.dropout(self.ent_tail_emb_re, rate=0.2,training=is_training)
		self.ent_tail_emb_im = tf.layers.dropout(self.ent_tail_emb_im, rate=0.2, training=is_training)


	def define_regularization(self):
		self.regularizer = (tf.nn.l2_loss(self.ent_head_emb_re) + tf.nn.l2_loss(self.ent_tail_emb_re) + tf.nn.l2_loss(self.rel_emb_re) + tf.nn.l2_loss(self.rel_inv_emb_re)
							+ tf.nn.l2_loss(self.ent_head_emb_im) + tf.nn.l2_loss(self.ent_tail_emb_im) + tf.nn.l2_loss(self.rel_emb_im) + tf.nn.l2_loss(self.rel_inv_emb_im)) / self.num_batch

	def gather_train_embeddings(self):
		self.h1_emb_re = tf.gather(self.ent_head_emb_re, self.head)
		self.h1_emb_im = tf.gather(self.ent_head_emb_im, self.head)

		self.h2_emb_re = tf.gather(self.ent_head_emb_re, self.tail)
		self.h2_emb_im = tf.gather(self.ent_head_emb_im, self.tail)

		self.t1_emb_re = tf.gather(self.ent_tail_emb_re, self.tail)
		self.t1_emb_im = tf.gather(self.ent_tail_emb_im, self.tail)

		self.t2_emb_re = tf.gather(self.ent_tail_emb_re, self.head)
		self.t2_emb_im = tf.gather(self.ent_tail_emb_im, self.head)

		self.r1_emb_re = tf.gather(self.rel_emb_re, self.rel)
		self.r1_emb_im = tf.gather(self.rel_emb_im, self.rel)

		self.r2_emb_re = tf.gather(self.rel_inv_emb_re, self.rel)
		self.r2_emb_im = tf.gather(self.rel_inv_emb_im, self.rel)


	def gather_test_embeddings(self):
		self.gather_train_embeddings()

	def create_train_model(self):
		self.dot1 = tf.reduce_sum(tf.multiply(self.r1_emb_re, tf.multiply(self.h1_emb_re, self.t1_emb_re)), 1)
		self.dot2 = tf.reduce_sum(tf.multiply(self.r1_emb_re, tf.multiply(self.h1_emb_im, self.t1_emb_im)), 1)
		self.dot3 = tf.reduce_sum(tf.multiply(self.r1_emb_im, tf.multiply(self.h1_emb_re, self.t1_emb_im)), 1)
		self.dot4 = tf.reduce_sum(tf.multiply(self.r1_emb_im, tf.multiply(self.h1_emb_im, self.t1_emb_re)), 1)
		self.init_scores1 = self.dot1 + self.dot2 + self.dot3 - self.dot4

		self.dot1_inv = tf.reduce_sum(tf.multiply(self.r2_emb_re, tf.multiply(self.h2_emb_re, self.t2_emb_re)), 1)
		self.dot2_inv = tf.reduce_sum(tf.multiply(self.r2_emb_re, tf.multiply(self.h2_emb_im, self.t2_emb_im)), 1)
		self.dot3_inv = tf.reduce_sum(tf.multiply(self.r2_emb_im, tf.multiply(self.h2_emb_re, self.t2_emb_im)), 1)
		self.dot4_inv = tf.reduce_sum(tf.multiply(self.r2_emb_im, tf.multiply(self.h2_emb_im, self.t2_emb_re)), 1)
		self.init_scores2 = self.dot1_inv + self.dot2_inv + self.dot3_inv - self.dot4_inv

		self.init_scores = (self.init_scores1 + self.init_scores2) / 2.0
		# self.init_scores = (tf.reduce_sum(tf.multiply(tf.multiply(self.h1_emb, self.r1_emb), self.t1_emb), 1) + tf.reduce_sum(tf.multiply(tf.multiply(self.h2_emb, self.r2_emb), self.t2_emb), 1)) / 2.0
		self.scores = tf.clip_by_value(self.init_scores, -20, 20) #Without clipping, we run into NaN problems.
		self.labels = self.y

	def create_test_model(self):
		self.dot1 = tf.reduce_sum(tf.multiply(self.r1_emb_re, tf.multiply(self.h1_emb_re, self.t1_emb_re)), 1)
		self.dot2 = tf.reduce_sum(tf.multiply(self.r1_emb_re, tf.multiply(self.h1_emb_im, self.t1_emb_im)), 1)
		self.dot3 = tf.reduce_sum(tf.multiply(self.r1_emb_im, tf.multiply(self.h1_emb_re, self.t1_emb_im)), 1)
		self.dot4 = tf.reduce_sum(tf.multiply(self.r1_emb_im, tf.multiply(self.h1_emb_im, self.t1_emb_re)), 1)
		self.init_scores1 = self.dot1 + self.dot2 + self.dot3 - self.dot4

		self.dot1_inv = tf.reduce_sum(tf.multiply(self.r2_emb_re, tf.multiply(self.h2_emb_re, self.t2_emb_re)), 1)
		self.dot2_inv = tf.reduce_sum(tf.multiply(self.r2_emb_re, tf.multiply(self.h2_emb_im, self.t2_emb_im)), 1)
		self.dot3_inv = tf.reduce_sum(tf.multiply(self.r2_emb_im, tf.multiply(self.h2_emb_re, self.t2_emb_im)), 1)
		self.dot4_inv = tf.reduce_sum(tf.multiply(self.r2_emb_im, tf.multiply(self.h2_emb_im, self.t2_emb_re)), 1)
		self.init_scores2 = self.dot1_inv + self.dot2_inv + self.dot3_inv - self.dot4_inv

		self.init_scores = (self.init_scores1 + self.init_scores2) / 2.0
		# self.init_scores = (tf.reduce_sum(tf.multiply(tf.multiply(self.h1_emb, self.r1_emb), self.t1_emb), 1) + tf.reduce_sum(tf.multiply(tf.multiply(self.h2_emb, self.r2_emb), self.t2_emb), 1)) / 2.0
		self.dissims = -tf.clip_by_value(self.init_scores, -20, 20) #Without clipping, we run into NaN problems.
