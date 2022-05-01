from __init__ import *
from transformers import BertTokenizer
from transformers import BertModel

def encode_data(data):
	encoding = BertTokenizer.from_pretrained('bert-base-cased')
	data_encode = encoding(data,padding='max_length',max_length=400,truncation=True,return_tensors='pt')
	return data_encode

def encode_data_file(data):
	pass


files = ['reviews_Electronics_5',
	'reviews_Electronics_5_1',
	'reviews_Electronics_5_2',
	'reviews_Electronics_5_3',
	'reviews_Electronics_5_4',
	'reviews_Electronics_5_5',
	'reviews_Electronics_5_6',
	'reviews_Electronics_5_7',
	'reviews_Electronics_5_8',
	'reviews_Electronics_5_9',
	'reviews_Electronics_5_10',
	'reviews_Electronics_5_11',
	'reviews_Electronics_5_12',
	'reviews_Electronics_5_13',
	'reviews_Electronics_5_14',
	'reviews_Electronics_5_15'
]

dir = '/mnt/ramdisk/Electronics/5-core'

for i in files:
	data = unpickle_data(f'{dir}/{i}.pickle')
	data['reviewText'] = encode_data(data['reviewText'])
	pickle_data(data, f'{dir}/{i}_tokenized.pickle')