import os
import numpy as np


'''
1.datasetSentences.txt와 datasetSplit.txt를 이용해서 valid, test 분리하고 문장을 key로 하고 value를 -1로 하는 딕셔너리 생성.
2.sentiment_labels.txt는 pharase-level임. 이것을 phrase ids를 key, sentiment values를 value로 딕셔너리 생성.
3.dictionary.txt를 읽고 2에서 만든 딕셔너리 이용해서 label을 구하고, 문장이 valid, test 딕셔너리에 있는지 확인
   => 있으면 valid, test에 Label을 value로 달아주고.
   => 없으면 train에 삽입.(리스트 형태)
4.valid, test의 value에 -1이 없는지 확인(전부 라벨 할당 잘 됐는지 체크임)하고 리스트 형태로 변환.
5.데이터 생성 완료.
'''
def _get_mode(split):
	if split[-1] == '\n':
		split = split[:-1]
	_mode = int(split.split(',')[1])
	return _mode


def _get_label(label):
	if label[-1] == '\n':
		label = label[:-1]
	label = label.split('|')
	
	score = float(label[1])
	phrase_id = label[0]

	label_bound = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
	for label, bound in enumerate(label_bound):
		if score <= bound[1]:
			return phrase_id, label


def _get_phrase(phrase):
	if phrase[-1] == '\n':
		phrase = phrase[:-1]
	phrase = phrase.split('|')
	return phrase


#1.datasetSentences.txt와 datasetSplit.txt를 이용해서 valid, test 분리하고 문장을 key로 하고 value를 -1로 하는 딕셔너리 생성.
def _valid_test_split_dictionary_form(sentence_path, split_path):
	# 중복 sentence 제거
	valid_dict = {}
	test_dict = {}
	train_dict = {}

	with open(sentence_path, 'r', encoding='utf-8') as f_sentence, open(split_path, 'r') as f_split:
		for line, data in enumerate(zip(f_sentence, f_split)):
			if line != 0:
				_sentence, _split = data
				# 종료 조건
				if _sentence == '\n' or _sentence == ' ' or _sentence == '':
					break
				
				# 뉴라인 제거
				if _sentence[-1] == '\n':
					_sentence = _sentence[:-1]
				_sentence = _sentence.split('\t')[1]

				# 데이터 split, sentence를 key로, -1을 value로
				split = _get_mode(_split)
				if split == 3: # dev
					valid_dict[_sentence] = -1
				elif split == 2:
					test_dict[_sentence] = -1
				elif split == 1:
					train_dict[_sentence] = -1

	return train_dict, valid_dict, test_dict


#2.sentiment_labels.txt는 pharase-level임. 이것을 phrase ids를 key, sentiment values를 value로 딕셔너리 생성.
def _get_sentiment_label_dictionary_form(label_path):
	label_dict = {}
	with open(label_path, 'r') as f_label:
		for line, _label in enumerate(f_label):
			if line != 0:
				if _label == '\n' or _label == ' ' or _label == '':
					break
				
				# 구문 id와 정답 추출.
				phrase_id, label = _get_label(_label)
				label_dict[phrase_id] = label

	return label_dict


def _remove_no_label(dictionary):
	for key, value in list(dictionary.items()):
		if value == -1:
			dictionary.pop(key)

#3.dictionary.txt를 읽고 2에서 만든 딕셔너리 이용해서 label을 구하고, 문장이 valid, test 딕셔너리에 있는지 확인
   #=> 있으면 valid, test에 Label을 value로 달아주고.
   #=> 없으면 train에 삽입.(리스트 형태)
def _set_label_and_get_phrase_dictionary_form(dictionary_path, train_dict, valid_dict, test_dict, label_dict):
	phrase_dict = {}
	with open(dictionary_path, 'r', encoding='utf-8') as f_phrase:
		for line, _phrase in enumerate(f_phrase):
			if _phrase == '\n' or _phrase == ' ' or _phrase == '':
				break
			phrase, phrase_id = _get_phrase(_phrase)
			

			if phrase_id not in label_dict:
				print(phrase_id, 'not in label_dict')
				continue
			
			# phrase_id가 label_dict에 있는 경우.
   				#=> 있으면 train_dict valid_dict, test_dict에 Label을 value로 달아줌.
  				#=> 없으면 phrase_dict에 삽입
			label = label_dict[phrase_id]
			if phrase in valid_dict:
				valid_dict[phrase] = label
			elif phrase in test_dict:
				test_dict[phrase] = label
			elif phrase in train_dict:
				train_dict[phrase] = label
			else:
				phrase_dict[phrase] = label
	
	_remove_no_label(train_dict)
	_remove_no_label(valid_dict)
	_remove_no_label(test_dict)

	dictionary = {'train':train_dict, 'valid':valid_dict, 'test':test_dict, 'phrase':phrase_dict}
	return dictionary


def _make_txt_from_dctionary(out_path, dictionary):
	# make directory
	if not os.path.exists(out_path):
		print("create save directory")
		os.makedirs(out_path)

	for mode in dictionary:
		print('mode:', mode, 'length:', len(dictionary[mode]))

		with open(out_path+mode+'.txt', 'w', encoding='utf-8') as o:
			for sentence, label in dictionary[mode].items():
				data = sentence + '|' + str(label) + '\n'
				o.write(data)


def _get_voca(path_list):
	word2idx = {'</p>':0, '</unk>':1}
	dict_value = 2
	maximum_length = 0

	for path in path_list:
		with open(path, 'r', encoding='utf-8') as f:
			for line, data in enumerate(f):
				if data == '\n' or data == ' ' or data == '':
					break

				# 뉴라인 제거
				if data[-1] == '\n':
					data = data[:-1]
				
				sentence, label = data.split('|')
				sentence = sentence.split()

				maximum_length = max(maximum_length, len(sentence))

				for word in sentence:
					if word not in word2idx:
						word2idx[word] = dict_value
						dict_value += 1
	return word2idx, maximum_length


def _get_maximum_length_from_path(path_list):
	maximum_length = 0
	for path in path_list:
		with open(path, 'r', encoding='utf-8') as f:
			for data in f:
				if data == '\n' or data == ' ' or data == '':
					break

				# 뉴라인 제거
				if data[-1] == '\n':
					data = data[:-1]
				
				sentence = data.split('|')[0]
				sentence = sentence.split()

				maximum_length = max(maximum_length, len(sentence))			
	return maximum_length


def _sentence2idx(sentence, word2idx, maximum_length):
	idx = []
	for word in sentence:
		if word in word2idx:
			idx.append(word2idx[word])
		else:
			idx.append(word2idx['</unk>'])

	idx = np.pad(idx, (0, maximum_length-len(idx)), 'constant', constant_values=word2idx['</p>'])
	return idx


def _word2dix_padding_split_source_target(data_path, word2idx, maximum_length, is_SST1=True):
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
					
					sentence, label = data.split('|')
					label = int(label)

					if is_SST1 is False:
						if label == 2: #중립
							continue
						elif label < 2: 
							label = 0
						else:
							label = 1

					idx = _sentence2idx(
							sentence.split(), 
							word2idx, 
							maximum_length
						)
					dataset[mode]['source'].append(idx)
					dataset[mode]['target'].append(label)		

	print('data result')
	for mode in dataset:
		for key in dataset[mode]:
			dataset[mode][key] = np.asarray(dataset[mode][key], np.int32)
			print(mode, key, dataset[mode][key].shape)

	return dataset


def split_dataset_to_txt(path_list, out_path):
	train_dict, valid_dict, test_dict = _valid_test_split_dictionary_form(path_list[0], path_list[1])

	label_dict = _get_sentiment_label_dictionary_form(path_list[2])
	dictionary = _set_label_and_get_phrase_dictionary_form(
			path_list[3], 
			train_dict, 
			valid_dict, 
			test_dict, 
			label_dict
		) # dictionary: {'train':dict, 'valid':dict, 'test':dict, 'phrase':dict}
	_make_txt_from_dctionary(out_path, dictionary)
	print('완료')




def get_dataset(path, is_phrase_tarin=True, is_SST1=True, maximum_length_margin=10):
	#maximum_length_margin => trainset_maximum_length + maximum_length_margin == maximum_length	
	train_path = path+'train.txt'
	valid_path = path+'valid.txt'
	test_path = path+'test.txt'
	phrase_path = path+'phrase.txt'

	if is_phrase_tarin:
		train_set_path = [train_path, phrase_path]
	else:
		train_set_path = [train_path]
	
	data_path = {
			'train':train_set_path,
			'valid':[valid_path],
			'test':[test_path]
		}

	# voca 추출
	word2idx, trainset_maximum_length = _get_voca(train_set_path)
	maximum_length = maximum_length_margin + trainset_maximum_length
	
	# make_dataset
	dataset = _word2dix_padding_split_source_target(
			data_path, 
			word2idx, 
			maximum_length,
			is_SST1=is_SST1
		)

	
	return dataset, word2idx, maximum_length


'''
out_path = 'preprocess/'
path_list = [
		'stanfordSentimentTreebank/datasetSentences.txt', # validset에 Hey Arnold ! 가 2개 있음.
		'stanfordSentimentTreebank/datasetSplit.txt',
		'stanfordSentimentTreebank/sentiment_labels.txt',
		'stanfordSentimentTreebank/dictionary.txt'
	]

split_dataset_to_txt(path_list, out_path)
get_dataset(out_path, is_phrase_tarin=True, is_SST1=True, maximum_length_margin=10)
get_dataset(out_path, is_phrase_tarin=True, is_SST1=False, maximum_length_margin=10)
'''