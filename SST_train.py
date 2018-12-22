import TextCNN
import SST_preprocess as pr
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

path_list = [
		'SST_data_extraction-master/sst_train_sentences.csv', 
		'SST_data_extraction-master/sst_train_phrases.csv',
		'SST_data_extraction-master/sst_dev.csv',
		'SST_data_extraction-master/sst_test.csv'
	]

data_type_dict = {'SST1':(True, 5), 'SST2':(False, 2)}
data_type = ['SST1', 'SST2'][0]
print(data_type)

dataset, word2idx, maximum_length = pr.get_dataset(
		path_list, 
		is_phrase_tarin=True, 
		#False,
		is_SST1=data_type_dict[data_type][0],
		max_length_margin=10
	)

tensorboard_path = data_type+'_tensorboard/'

window_size = [3, 4, 5] 
filters = [100, 100, 100] 
num_classes = data_type_dict[data_type][1]
pad_idx = word2idx['</p>']
lr = 0.001 * 10
voca_size = len(word2idx)
embedding_size = 300
embedding_mode = 'rand'
word_embedding = None	
batch_size = 50


def cost_weight_for_imbalanced_label(target):
	#target: [1,2,1,0,2,2]

	count = np.zeros(num_classes)
	for i in target:
		count[i] += 1
	#count: [1, 2, 3]

	weight = sum(count) / (count+0.00001) # avoid divide by zero
	#weight: [6, 3, 2] 

	weight = weight/np.min(weight)
	#weight: [3, 1.5, 1]

	# cross entropy 결과에 이 값들 곱해주면 됨. (부족한 데이터에 gradient 증폭)
	cost_weight = [weight[i] for i in target]
	#cost_weight: [1.5, 1, 1.5, 3, 1, 1]
	return np.asarray(cost_weight, np.float32)

def train(model, dataset):
	loss = 0
	total_data_length = len(dataset['source'])
	acc = 0

	# shuffle
	indices = np.arange(total_data_length)
	np.random.shuffle(indices)
	source = dataset['source'][indices]
	target = dataset['target'][indices]

	for i in tqdm(range( int(np.ceil(total_data_length/batch_size)) ), ncols=50):
		batch_source = source[batch_size * i: batch_size * (i + 1)] # [N, maximum_source_length]
		batch_target = target[batch_size * i: batch_size * (i + 1)] # [N]
		cost_weight = cost_weight_for_imbalanced_label(batch_target)
	
		train_loss, _, correct = sess.run([model.cost, model.minimize, model.correct_check],
				{
					model.idx_input:batch_source, 
					model.target:batch_target, 
					model.keep_prob:0.5,
					model.weight_scale:1,
					model.cost_weight:cost_weight
				}
			)
		acc += correct
		loss += train_loss
	print(acc/total_data_length)
	return loss / int(np.ceil(total_data_length/batch_size))


def valid(model, dataset):
	loss = 0
	total_data_length = len(dataset['source'])

	source = dataset['source']
	target = dataset['target']

	for i in tqdm(range( int(np.ceil(total_data_length/batch_size)) ), ncols=50):
		batch_source = source[batch_size * i: batch_size * (i + 1)] # [N, maximum_source_length]
		batch_target = target[batch_size * i: batch_size * (i + 1)] # [N]
		cost_weight = cost_weight_for_imbalanced_label(batch_target)
	
		valid_loss = sess.run(model.cost,
				{
					model.idx_input:batch_source, 
					model.target:batch_target, 
					model.keep_prob:1,
					model.weight_scale:0.5, # train drop_out rate
					model.cost_weight:cost_weight
				}
			)
		loss += valid_loss
	return loss / int(np.ceil(total_data_length/batch_size))


def test(model, dataset):
	accuracy = 0
	total_data_length = len(dataset['source'])

	source = dataset['source']
	target = dataset['target']

	for i in tqdm(range( int(np.ceil(total_data_length/batch_size)) ), ncols=50):
		batch_source = source[batch_size * i: batch_size * (i + 1)] # [N, maximum_source_length]
		batch_target = target[batch_size * i: batch_size * (i + 1)] # [N]
	
		correct, pred_argmax = sess.run([model.correct_check, model.pred_argmax],
				{
					model.idx_input:batch_source, 
					model.target:batch_target, 
					model.keep_prob:1,
					model.weight_scale:0.5 # train drop_out rate
				}
			)
		#print(np.array(list(zip(pred_argmax, batch_target))))
		accuracy += correct
	return accuracy / total_data_length


def run(model, dataset):

	with tf.name_scope("tensorboard"):
		train_loss_tensorboard = tf.placeholder(tf.float32, name='train_loss')
		vali_loss_tensorboard = tf.placeholder(tf.float32, name='vali_loss')
		test_accuracy_tensorboard = tf.placeholder(tf.float32, name='test')

		train_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
		vali_summary = tf.summary.scalar("vali_loss", vali_loss_tensorboard)
		test_summary = tf.summary.scalar("test_accuracy", test_accuracy_tensorboard)
				
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

	trainset, validset, testset = dataset['train'], dataset['valid'], dataset['test']
	for epoch in range(1, 5001):
		train_loss = train(model, trainset)
		valid_loss = valid(model, validset)
		test_accuracy = test(model, testset)
		print("epoch:", epoch, 'train_loss:', train_loss, 'valid_loss:', valid_loss, 'test_accuracy:', test_accuracy, '\n')
		
		# tensorboard
		summary = sess.run(merged, {
					train_loss_tensorboard:train_loss, 
					vali_loss_tensorboard:valid_loss,
					test_accuracy_tensorboard:test_accuracy, 
				}
		)
		writer.add_summary(summary, epoch)


sess = tf.Session()
model = TextCNN.TextCNN(
		sess = sess,
		window_size = window_size,
		filters = filters,
		num_classes = num_classes,
		pad_idx = pad_idx,
		lr = lr,
		voca_size = voca_size,
		embedding_size = embedding_size,
		embedding_mode = embedding_mode,
		word_embedding = word_embedding
	)

run(model, dataset)
