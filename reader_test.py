import random
import math
from random import shuffle
import tensorflow as tf


class Reader:

    def __init__(self):
        self.ent2id = dict()
        self.rel2id = dict()
        self.triples = {"train": [], "valid": [], "test": []}
        self.start_batch = 0

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def train_triples(self):
        return self.triples["train"]

    def valid_triples(self):
        return self.triples["valid"]

    def test_triples(self):
        return self.triples["test"]

    def all_triples(self):
        return self.triples["train"] + self.triples["valid"] + self.triples["test"]

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def num_batch(self):
        return int(math.ceil(float(len(self.triples["train"])) / self.batch_size))

    def next_pos_batch(self):
        if self.start_batch + self.batch_size > len(self.triples["train"]):
            ret_triples = self.triples["train"][self.start_batch:]
            self.start_batch = 0
        else:
            ret_triples = self.triples["train"][self.start_batch: self.start_batch + self.batch_size]
            self.start_batch += self.batch_size
        return ret_triples

    def get_add_ent_id(self, entity):
        if entity in self.ent2id:
            entity_id = self.ent2id[entity]
        else:
            entity_id = len(self.ent2id)
            self.ent2id[entity] = entity_id
        return entity_id

    def get_add_rel_id(self, relation):
        if relation in self.rel2id:
            relation_id = self.rel2id[relation]
        else:
            relation_id = len(self.rel2id)
            self.rel2id[relation] = relation_id
        return relation_id

    def read_triples(self, directory="wn18/"):
        for file in ["train", "valid", "test"]:
            with open(directory + file + ".txt", "r") as f:
                for line in f.readlines():
                    head, rel, tail = line.strip().split("\t")
                    head_id = self.get_add_ent_id(head)
                    rel_id = self.get_add_rel_id(rel)
                    tail_id = self.get_add_ent_id(tail)
                    self.triples[file].append((head_id, rel_id, tail_id))

    def rand_ent_except(self, except_ent):
        rand_ent = random.randint(0, self.num_ent() - 1)
        while rand_ent == except_ent:
            rand_ent = random.randint(0, self.num_ent() - 1)
        return rand_ent

    def generate_neg_triples(self, batch_pos_triples):
        neg_triples = []
        for head, rel, tail in batch_pos_triples:
            head_or_tail = random.randint(0, 1)
            if head_or_tail == 0:  # head
                new_head = self.rand_ent_except(head)
                neg_triples.append((new_head, rel, tail))
            else:  # tail
                new_tail = self.rand_ent_except(tail)
                neg_triples.append((head, rel, new_tail))
        return neg_triples

    def shred_triples(self, triples):
        h_idx = [triples[i][0] for i in range(len(triples))]
        r_idx = [triples[i][1] for i in range(len(triples))]
        t_idx = [triples[i][2] for i in range(len(triples))]
        return h_idx, r_idx, t_idx

    def shred_triples_and_labels(self, triples_and_labels):
        heads = [triples_and_labels[i][0][0] for i in range(len(triples_and_labels))]
        rels = [triples_and_labels[i][0][1] for i in range(len(triples_and_labels))]
        tails = [triples_and_labels[i][0][2] for i in range(len(triples_and_labels))]
        labels = [triples_and_labels[i][1] for i in range(len(triples_and_labels))]
        return heads, rels, tails, labels

    def next_batch(self, format="pos_neg", neg_ratio=1):
        if format == "pos_neg":
            bp_triples = self.next_pos_batch()
            bn_triples = self.generate_neg_triples(bp_triples)
            ph_idx, pr_idx, pt_idx = self.shred_triples(bp_triples)
            nh_idx, nr_idx, nt_idx = self.shred_triples(bn_triples)
            return ph_idx, pt_idx, nh_idx, nt_idx, pr_idx
        elif format == "triple_label":
            bp_triples = self.next_pos_batch()
            bp_triples_and_labels = [(bp_triples[i], 1.0) for i in range(len(bp_triples))]
            bn_triples_and_labels = []

            for _ in range(neg_ratio):
                bn_triples = self.generate_neg_triples(bp_triples)
                bn_triples_and_labels += [(bn_triples[i], -1.0) for i in range(len(bn_triples))]
            all_triples_and_labels = bp_triples_and_labels + bn_triples_and_labels
            shuffle(all_triples_and_labels)
            mzy = self.shred_triples_and_labels(all_triples_and_labels)
            return self.shred_triples_and_labels(all_triples_and_labels)
        else:
            print("Unrecognizeable format in reader.next_batch")
            exit()

    def next_example(self):
        ph_idx, pr_idx, pt_idx, nh_idx, nr_idx, nt_idx = self.next_batch()
        return ph_idx[0], pt_idx[0], nh_idx[0], nt_idx[0], pr_idx[0]

    def replace_fil(self, triple, head_or_tail):
        raw_triples = self.replace_raw_unshreded(triple, head_or_tail)
        ret_triples = [triple] + list(set(raw_triples) - set(self.all_triples()))
        return self.shred_triples(ret_triples)

    def replace_raw_unshreded(self, triple, head_or_tail):
        ret_triples = []
        head, rel, tail = triple
        for i in range(self.num_ent()):
            if head_or_tail == "head" and i != head:
                ret_triples.append((i, rel, tail))
            elif head_or_tail == "tail" and i != tail:
                ret_triples.append((head, rel, i))
        return [triple] + ret_triples

    def replace_raw(self, triple, head_or_tail):
        return self.shred_triples(self.replace_raw_unshreded(triple, head_or_tail))

    def get_rank(self, triple_dissims, dissim_of_actual_triple):
        rank = 1.0
        for dissim in triple_dissims:
            if dissim < dissim_of_actual_triple:
                rank += 1
        return rank




reader = Reader()
reader.read_triples("fb15k" + "/")
reader.set_batch_size(4832)

h, r, t, y = reader.next_batch(format="triple_label", neg_ratio=10)

num_batch = reader.num_batch()
num_ent = reader.num_ent()
num_rel = reader.num_rel()

sqrt_size = 6.0 / math.sqrt(200)
rel_emb = tf.get_variable(name="rel_emb", initializer=tf.random_uniform(shape=[num_rel, 200],minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
rel_inv_emb = tf.get_variable(name="rel_inv_emb", initializer=tf.random_uniform(shape=[num_rel, 200], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
ent_head_emb = tf.get_variable(name="ent_head_emb", initializer=tf.random_uniform(shape=[num_ent, 200], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))
ent_tail_emb = tf.get_variable(name="ent_tail_emb", initializer=tf.random_uniform(shape=[num_ent, 200], minval=-sqrt_size, maxval=sqrt_size, dtype=tf.float64))


# h1_emb = tf.gather(ent_head_emb, h)
# h2_emb = tf.gather(ent_head_emb, t)
# t1_emb = tf.gather(ent_tail_emb, t)
# t2_emb = tf.gather(ent_tail_emb, h)
# r1_emb = tf.gather(rel_emb, r)
# r2_emb = tf.gather(rel_inv_emb, r)

h_emb = tf.concat([tf.gather(ent_head_emb, h), tf.gather(ent_head_emb, t)], 0)
t_emb = tf.concat([tf.gather(ent_tail_emb, t), tf.gather(ent_tail_emb, h)], 0)
r_emb = tf.concat([tf.gather(rel_emb, r), tf.gather(rel_inv_emb, r)], 0)

hr = tf.multiply(h_emb, r_emb)
hrt = tf.multiply(tf.multiply(h_emb, r_emb), t_emb)
# init_scores = (tf.reduce_sum(tf.multiply(tf.multiply(h1_emb, r1_emb), t1_emb), 1) + tf.reduce_sum(tf.multiply(tf.multiply(h2_emb, r2_emb), t2_emb), 1)) / 2.0
init_scores = tf.reduce_sum(tf.multiply(tf.multiply(h_emb, r_emb), t_emb), 1)
scores = tf.clip_by_value(init_scores, -20, 20)




sess = tf.Session()
sess.run(tf.global_variables_initializer())
a = init_scores
print(sqrt_size)
# triples = reader.triples["valid"]
# for i, triple in enumerate(triples):
#     head_raw_h, head_raw_r, head_raw_t = reader.replace_raw(triple, "head")
#     tail_raw_h, tail_raw_r, tail_raw_t = reader.replace_raw(triple, "tail")
#     head_fil_h, head_fil_r, head_fil_t = reader.replace_fil(triple, "head")
#     tail_fil_h, tail_fil_r, tail_fil_t = reader.replace_fil(triple, "tail")
#     print(head_raw_h)
