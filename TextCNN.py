import tensorflow as tf
import numpy as np

class TextCNN:
	def __init__(self, sess, window_size, filters, num_classes, pad_idx, lr,
					voca_size, embedding_size, embedding_mode='rand', word_embedding=None):
		self.sess = sess
		self.window_size = window_size # like [3, 4, 5]
		self.filters = filters # the number of out filter
		self.num_classes = num_classes
		self.pad_idx = pad_idx
		self.lr = lr
		self.voca_size = voca_size
		self.embedding_size = embedding_size
		self.embedding_mode = embedding_mode # mode: rand static nonstatic multichannel
		self.word_embedding = word_embedding # only if embedding_mode is not 'rand'


		with tf.name_scope("placeholder"):
			self.idx_input = tf.placeholder(tf.int32, [None, None], name="idx_input")
			self.target = tf.placeholder(tf.int32, [None], name="target")
			self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
			self.weight_scale = tf.placeholder(tf.float32, name="weight_scale")
			self.cost_weight = tf.placeholder(tf.float32, [None], name="cost_weight")

		with tf.name_scope("embedding_table"):
			if self.embedding_mode == 'rand':
				self.embedding_table = self.rand_embedding_table(self.voca_size, self.pad_idx, self.embedding_size)
			
			elif self.embedding_mode == 'static':
				self.embedding_table = tf.constant(self.word_embedding, dtype=tf.float32)
			
			elif self.embedding_mode == 'nonstatic':
				self.embedding_table = tf.Variable(self.word_embedding, dtype=tf.float32) # trainable
			
			elif self.embedding_mode == 'multichannel':
				self.static_embedding_table = tf.constant(self.word_embedding, dtype=tf.float32)
				self.nonstatic_embedding_table = tf.Variable(self.word_embedding, dtype=tf.float32) # trainable
				

		with tf.name_scope("embedding_lookup"): 
			if self.embedding_mode != 'multichannel':
				embedding = tf.nn.embedding_lookup(self.embedding_table, self.idx_input) # [N, self.idx_input_length, self.embedding_size]
				self.embedding = tf.expand_dims(embedding, axis=-1) # [N, self.idx_input_length, self.embedding_size, 1]
			else:
				static_embedding = tf.nn.embedding_lookup(self.static_embedding_table, self.idx_input) # [N, self.idx_input_length, self.embedding_size]
				static_embedding = tf.expand_dims(static_embedding, axis=-1) # [N, self.idx_input_length, self.embedding_size, 1]
				nonstatic_embedding = tf.nn.embedding_lookup(self.nonstatic_embedding_table, self.idx_input) # [N, self.idx_input_length, self.embedding_size]
				nonstatic_embedding = tf.expand_dims(nonstatic_embedding, axis=-1) # [N, self.idx_input_length, self.embedding_size, 1]
				self.embedding = tf.concat((static_embedding, nonstatic_embedding), axis=-1) # [N, self.idx_input_length, self.embedding_size, 2]
			

		with tf.name_scope("model_architecture"):
			self.convolved_features = self.convolution(self.embedding, self.embedding_size, self.window_size, self.filters)
			self.pooled_features = self.max_pooling(self.convolved_features)
			self.concat_and_flatten_features = self.concat_and_flatten(self.pooled_features)
			self.pred, self.w_l2 = self.dropout_and_dense(self.concat_and_flatten_features, self.num_classes, self.keep_prob)


		with tf.name_scope('train'): 
			target_one_hot = tf.one_hot(
					self.target, # [N]
					depth=self.num_classes,
					on_value = 1., # tf.float32
					off_value = 0., # tf.float32
				) # [N, self.num_classes]
			# calc train_cost
			self.cost = tf.nn.softmax_cross_entropy_with_logits(
					labels = target_one_hot, 
					logits = self.pred
				) # [N]
			self.cost = tf.reduce_mean(self.cost_weight * self.cost)
				
			s = 3.0
			l2_cost_scale = 0.1 
			self.l2_cost = ((self.w_l2 - s)**2)/2 # l2_loss를 3으로 고정시킴. 
			#optimizer = tf.train.AdadeltaOptimizer(self.lr)
			optimizer = tf.train.AdamOptimizer(self.lr)
			self.minimize = optimizer.minimize(self.cost + l2_cost_scale * self.l2_cost)
			
			
		with tf.name_scope('metric'):
			self.pred_argmax = tf.argmax(self.pred, 1, output_type=tf.int32) # [N]	
			self.correct_check_beform_sum = tf.equal( self.pred_argmax, self.target )
			self.correct_check = tf.reduce_sum(tf.cast(self.correct_check_beform_sum, tf.int32 ))

		self.sess.run(tf.global_variables_initializer())


	def rand_embedding_table(self, voca_size, pad_idx, embedding_size=300): 
		zero = tf.zeros([1, embedding_size], dtype=tf.float32) # for padding 
		embedding_table = tf.Variable(tf.random_uniform([voca_size-1, embedding_size], -1, 1)) 
		front, end = tf.split(embedding_table, [pad_idx, voca_size-1-pad_idx])
		embedding_table = tf.concat((front, zero, end), axis=0)
		return embedding_table



	def convolution(self, embedding, embedding_size, window_size, filters):
		convolved_features = []
		for i in range(len(window_size)):
			convolved = tf.layers.conv2d(
					inputs = embedding, 
					filters = filters[i], 
					kernel_size = [window_size[i], embedding_size], 
					strides=[1, 1], 
					padding='VALID', 
					activation=tf.nn.relu
				) # [N, ?, 1, filters]
			convolved_features.append(convolved) # [N, ?, 1, filters] 이 len(window_size) 만큼 존재.
		return convolved_features


	def max_pooling(self, convolved_features):		
		pooled_features = []
		for convolved in convolved_features: # [N, ?, 1, self.filters]
			max_pool = tf.reduce_max(
  					input_tensor = convolved,
					axis = 1,
					keep_dims = True
				) # [N, 1, 1, self.filters]
			pooled_features.append(max_pool) # [N, 1, 1, self.filters] 이 len(window_size) 만큼 존재.
		return pooled_features
		

	def concat_and_flatten(self, pooled_features):
		concat = tf.concat(pooled_features, axis=-1) # [N, 1, 1, np.sum(self.filters)]
		concat_and_flatten_features = tf.reshape(concat, [-1, np.sum(self.filters)]) # [N, np.sum(self.filters)]
		return concat_and_flatten_features


	def dropout_and_dense(self, concat_and_flatten_features, num_classes, keep_prob):
		dropout = tf.nn.dropout(concat_and_flatten_features, keep_prob = keep_prob)
		W = tf.get_variable('w2', shape = [np.sum(self.filters), num_classes], initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.Variable(tf.constant(0.0, shape = [num_classes]))
		
		dense = tf.matmul(dropout, self.weight_scale*W) + bias
		w_l2 = tf.sqrt( tf.reduce_sum(tf.square(W)) )
		return dense, w_l2