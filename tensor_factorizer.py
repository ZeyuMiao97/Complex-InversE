#Copyright (C) 2018  Seyed Mehran Kazemi, Licensed under the GPL V3; see: <https://www.gnu.org/licenses/gpl-3.0.en.html>
import tensorflow as tf
import time
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import math
from reader import *
import os
import numpy as np

class TensorFactorizer:

	def __init__(self, model_name, params, loss_function="margin", dataset="wn18"):
		self.model_name = model_name
		self.params = params
		self.dataset = dataset
		self.loss_function = loss_function

	def setup_reader(self):
		self.reader = Reader()
		self.reader.read_triples(self.dataset + "/")
		self.reader.set_batch_size(self.params.batch_size)
		self.num_batch = self.reader.num_batch()
		self.num_ent = self.reader.num_ent()
		self.num_rel = self.reader.num_rel()

	def setup_loader(self):
		self.loader = tf.train.Saver(self.var_list)

	def setup_saver(self):
		self.saver = tf.train.Saver(max_to_keep=0)

	def create_session(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		# if self.model_name =="ComplEx":
		# 	ent_emb_img = self.sess.run(self.ent_emb_img)
		# 	ent_emb_real = self.sess.run(self.ent_emb_real)
		# 	rel_emb_img = self.sess.run(self.rel_emb_img)
		# 	rel_emb_real = self.sess.run(self.rel_emb_real)

		# elif self.model_name == "SimplE_avg":
		# 	rel_emb = self.sess.run(self.rel_emb)
		# 	rel_inv_emb = self.sess.run(self.rel_inv_emb)
		# 	ent_head_emb = self.sess.run(self.ent_head_emb)
		# 	ent_tail_emb = self.sess.run(self.ent_tail_emb)


		# dot1 = self.dot1
		# print(rel_emb_real)


	def load_session(self, itr):
		self.loader.restore(self.sess, self.model_name + "_weights/" + self.dataset + "/" + itr + ".ckpt")

	def close_session(self):
		self.sess.close()

	def create_train_placeholders(self):
		if self.loss_function == "margin":
			self.ph = tf.placeholder(tf.int32, [None])
			self.pt = tf.placeholder(tf.int32, [None])
			self.nh = tf.placeholder(tf.int32, [None])
			self.nt = tf.placeholder(tf.int32, [None])
			self.r  = tf.placeholder(tf.int32, [None])
		elif self.loss_function == "likelihood":
			self.head = tf.placeholder(tf.int32, [None])
			self.rel  = tf.placeholder(tf.int32, [None])
			self.tail = tf.placeholder(tf.int32, [None])
			self.y    = tf.placeholder(tf.float64, [None])
		else:
			print("Unrecognizable loss function.")
			exit()

	def create_test_placeholders(self):
		self.head = tf.placeholder(tf.int32, [None])
		self.rel  = tf.placeholder(tf.int32, [None])
		self.tail = tf.placeholder(tf.int32, [None])

	def create_optimizer(self):
		if self.loss_function == "margin":
			self.loss = tf.reduce_sum(tf.nn.relu(self.params.gamma + self.pos_dissims - self.neg_dissims)) + self.params.alpha * self.regularizer
		else:
			self.loss = tf.reduce_sum(tf.nn.softplus(-self.labels * self.scores)) + self.params.alpha * self.regularizer

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.optimizer = tf.train.AdagradOptimizer(self.params.learning_rate).minimize(self.loss)

	def save_model(self, itr):
		filename = self.model_name + "_weights/" + self.dataset + "/" + str(itr) + ".ckpt"
		if not os.path.exists(os.path.dirname(filename)):
			os.makedirs(os.path.dirname(filename))
		self.saver.save(self.sess, filename)

	def optimize(self):
		for itr in range(1, self.params.max_iterate + 1):
			total_loss = 0.0
			time_start = time.time()
			for b in range(self.num_batch):
				if self.loss_function == "margin":
					ph, pt, nh, nt, r = self.reader.next_batch(format="pos_neg")
					_, err = self.sess.run([self.optimizer, self.loss], feed_dict={self.ph: ph, self.pt: pt, self.nh: nh, self.nt: nt, self.r: r})
				else:
					h, r, t, y = self.reader.next_batch(format="triple_label", neg_ratio=self.params.neg_ratio)
					_, err = self.sess.run([self.optimizer, self.loss], feed_dict={self.head: h, self.rel: r, self.tail: t, self.y: y})
				total_loss += err
			time_end = time.time()
			time_c = time_end - time_start

			print("Loss in iteration", itr, "=", total_loss,"    ",'time cost', time_c, 's')
			if(itr % self.params.save_each == 0 and itr >= self.params.save_after):
				self.save_model(itr)

	def test(self, triples):
		r_mrr = f_mrr = r_hit1 = r_hit3 = r_hit10 = f_hit1 = f_hit3 = f_hit10 = 0.0
		time_1 = time.time()

		for i, triple in enumerate(triples):
			if i % 100 == 0:
				print(i)
			head_raw_h, head_raw_r, head_raw_t = self.reader.replace_raw(triple, "head")
			tail_raw_h, tail_raw_r, tail_raw_t = self.reader.replace_raw(triple, "tail")
			head_fil_h, head_fil_r, head_fil_t = self.reader.replace_fil(triple, "head")
			tail_fil_h, tail_fil_r, tail_fil_t = self.reader.replace_fil(triple, "tail")

			head_raw_preds = self.sess.run(self.dissims, feed_dict={self.head: head_raw_h, self.rel: head_raw_r, self.tail: head_raw_t})
			tail_raw_preds = self.sess.run(self.dissims, feed_dict={self.head: tail_raw_h, self.rel: tail_raw_r, self.tail: tail_raw_t})
			head_fil_preds = self.sess.run(self.dissims, feed_dict={self.head: head_fil_h, self.rel: head_fil_r, self.tail: head_fil_t})
			tail_fil_preds = self.sess.run(self.dissims, feed_dict={self.head: tail_fil_h, self.rel: tail_fil_r, self.tail: tail_fil_t})

			head_raw_rank = self.reader.get_rank(head_raw_preds[1:], head_raw_preds[0])
			tail_raw_rank = self.reader.get_rank(tail_raw_preds[1:], tail_raw_preds[0])
			head_fil_rank = self.reader.get_rank(head_fil_preds[1:], head_fil_preds[0])
			tail_fil_rank = self.reader.get_rank(tail_fil_preds[1:], tail_fil_preds[0])

			r_hit1  += float(head_raw_rank <= 1)  + float(tail_raw_rank <= 1)
			r_hit3  += float(head_raw_rank <= 3)  + float(tail_raw_rank <= 3)
			r_hit10 += float(head_raw_rank <= 10) + float(tail_raw_rank <= 10)

			f_hit1  += float(head_fil_rank <= 1)  + float(tail_fil_rank <= 1)
			f_hit3  += float(head_fil_rank <= 3)  + float(tail_fil_rank <= 3)
			f_hit10 += float(head_fil_rank <= 10) + float(tail_fil_rank <= 10)

			r_mrr += ((1.0 / head_raw_rank) + (1.0 / tail_raw_rank))
			f_mrr += ((1.0 / head_fil_rank) + (1.0 / tail_fil_rank))

		r_hit1  /= (2.0 * len(triples))
		r_hit3  /= (2.0 * len(triples))
		r_hit10 /= (2.0 * len(triples))
		f_hit1  /= (2.0 * len(triples))
		f_hit3  /= (2.0 * len(triples))
		f_hit10 /= (2.0 * len(triples))
		r_mrr   /= (2.0 * len(triples))
		f_mrr   /= (2.0 * len(triples))

		time_2 = time.time()
		time_cost = time_1 - time_2
		print("test total cost:",time_cost)
		return r_mrr, r_hit1, r_hit3, r_hit10, f_mrr, f_hit1, f_hit3, f_hit10



