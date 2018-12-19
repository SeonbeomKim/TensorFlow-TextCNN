import os
import csv
import numpy as np

path_list = [
		'stanfordSentimentTreebank/datasetSentences.txt',
		'stanfordSentimentTreebank/datasetSplit.txt',
		'stanfordSentimentTreebank/sentiment_labels.txt',
		'stanfordSentimentTreebank/dictionary.txt'
	]

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
	valid_dict = {}
	test_dict = {}

	with open(sentence_path, 'r') as f_sentence, open(split_path, 'r') as f_split:
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
					if _sentence not in valid_dict:
						valid_dict[_sentence] = 0
					else:
						valid_dict[_sentence] += 1

					#valid_dict[_sentence] = -1
				elif split == 2:				
					test_dict[_sentence] = -1

	return valid_dict, test_dict


#2.sentiment_labels.txt는 pharase-level임. 이것을 phrase ids를 key, sentiment values를 value로 딕셔너리 생성.
def _get_sentiment_label_dictionary_form(label_path, dataset='SST-1'):
	label_dict = {}
	with open(label_path, 'r') as f_label:
		for line, _label in enumerate(f_label):
			if line != 0:
				if _label == '\n' or _label == ' ' or _label == '':
					break
				
				# 구문 id와 정답 추출.
				phrase_id, label = _get_label(_label)
				if dataset == 'SST-2':
					if label == 2: # neutral
						continue
					elif label < 2: # very negative, negative
						label = 0
					else: # positive, very positive
						label = 1 

				label_dict[phrase_id] = label

	return label_dict


#3.dictionary.txt를 읽고 2에서 만든 딕셔너리 이용해서 label을 구하고, 문장이 valid, test 딕셔너리에 있는지 확인
   #=> 있으면 valid, test에 Label을 value로 달아주고.
   #=> 없으면 train에 삽입.(리스트 형태)
def _make_train_valid_test(dictionary_path, valid_dict, test_dict, label_dict):
	train_dict = {}
	with open(dictionary_path, 'r') as f_phrase:
		for line, _pharse in enumerate(f_phrase):
			if _pharse == '\n' or _pharse == ' ' or _pharse == '':
				break
			phrase, phrase_id = _get_phrase(_pharse)
			
			# 'SST-2'는 중립 문장 제거하므로 없는 경우가 있음.
			if phrase_id not in label_dict:
				continue

			# phrase_id가 label_dict에 있는 경우.
   				#=> 있으면 valid, test에 Label을 value로 달아주고.
  				#=> 없으면 train에 삽입.(리스트 형태)
			label = label_dict[phrase_id]
			if phrase in valid_dict:
				valid_dict[phrase] = label
			elif phrase in test_dict:
				test_dict[phrase] = label
			else:
				train_dict[phrase] = label
	return train_dict, valid_dict, test_dict



valid_dict, test_dict = _valid_test_split_dictionary_form(path_list[0], path_list[1])
#label_dict = _get_sentiment_label_dictionary_form(path_list[2])
#train, valid, test = _make_train_valid_test(path_list[3], valid_dict, test_dict, label_dict)

for i in valid_dict:
	if valid_dict[i] == 1:
		print(i)

'''
print(len(valid_dict))
print(len(test_dict))
print(len(label_dict))

print(len(train))
print(len(valid))
print(len(test))
'''
#for i in train:
#	print(i, train[i])