import os
import numpy as np


def _split_sentence_score(data):
	split_data = data.split(',')
	score = float(split_data[0])
	sentence = ','.join(split_data[1:])
	sentence = sentence.split()	
	return sentence, score

def _get_voca_and_max_length(path_list):
	word2idx = {'</p>':0, '</unk>':1}
	dict_value = 2
	max_length = 0

	for path in path_list:
		with open(path, 'r', encoding='utf-8') as f:
			for line, data in enumerate(f):
				if data == '\n' or data == ' ' or data == '':
					break

				# 뉴라인 제거
				if data[-1] == '\n':
					data = data[:-1]
				
				sentence, _ = _split_sentence_score(data)
				max_length = max(max_length, len(sentence))

				for word in sentence:
					if word not in word2idx:
						word2idx[word] = dict_value
						dict_value += 1
	return word2idx, max_length


def _get_label(score):
	label_bound = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
	for label, bound in enumerate(label_bound):
		if score <= bound[1]:
			return label


def _sentence2idx(sentence, word2idx, max_length):
	idx = []
	for word in sentence:
		if word in word2idx:
			idx.append(word2idx[word])
		else:
			idx.append(word2idx['</unk>'])
	idx = np.pad(idx, (0, max_length-len(idx)), 'constant', constant_values=word2idx['</p>'])
	return idx


def _get_dataset(data_path, word2idx, max_length, is_SST1):
	dataset = {
		'train':{'source':[], 'target':[]},
		'valid':{'source':[], 'target':[]},
		'test':{'source':[], 'target':[]}
	}

	for mode in data_path:
		for path in data_path[mode]:		
			with open(path, 'r', encoding='utf-8') as f:
				for data in f:
					if data == '\n' or data == ' ' or data == '':
						break

					# 뉴라인 제거
					if data[-1] == '\n':
						data = data[:-1]
					
					sentence, score = _split_sentence_score(data)
					label = _get_label(score)

					if is_SST1 is False:
						if label == 2: #중립
							continue
						elif label < 2: 
							label = 0
						else:
							label = 1

					idx = _sentence2idx(
							sentence, 
							word2idx, 
							max_length
						)
					dataset[mode]['source'].append(idx)
					dataset[mode]['target'].append(label)		

	print('data result')
	for mode in dataset:
		for key in dataset[mode]:
			dataset[mode][key] = np.asarray(dataset[mode][key], np.int32)
			print(mode, key, dataset[mode][key].shape)
	return dataset	


def get_dataset(path_list, is_phrase_tarin=True, is_SST1=True, max_length_margin=10):
	
	if is_phrase_tarin:
		train_path = [path_list[0], path_list[1]]
	else:
		train_path = [path_list[0]]
	valid_path = [path_list[2]]
	test_path = [path_list[3]]

	data_path = {
			'train':train_path,
			'valid':valid_path,
			'test':test_path
		}

	word2idx, max_length = _get_voca_and_max_length(train_path)
	dataset = _get_dataset(data_path, word2idx, max_length+max_length_margin, is_SST1)
	return dataset, word2idx, max_length

