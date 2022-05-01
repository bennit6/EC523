# from __init__ import *
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertModel

def tokenize_data_BERT(data):
	encoding = BertTokenizer.from_pretrained('bert-base-cased')
	data_encode = encoding(data,padding='max_length',max_length=400,truncation=True,return_tensors='pt')
	return data_encode

def encode_data_BERT(input_ids,attention_mask,batch_size=5):
	index_base = torch.tensor(list(range(batch_size)))
	model = BertModel.from_pretrained('bert-base-cased').cuda()

	size = input_ids.shape[0]
	num_words = input_ids.shape[1]

	pool_out = torch.zeros((size,768))

	# loops through whole batches
	for i in range(size//batch_size):
		torch.cuda.empty_cache()

		index = index_base + (batch_size* i)
		input_ids_sample = input_ids[index,:].cuda()
		attention_mask_sample = attention_mask[index,:].cuda()
		out,pool = model(input_ids=input_ids_sample , attention_mask=attention_mask_sample,return_dict=False)

		# delete these to free up cuda
		del input_ids_sample
		del attention_mask_sample

		pool_out[index,:]   = pool.detach().cpu()

		del out
		del pool


		print(i*batch_size,size)

	if size//batch_size != size/batch_size:
		torch.cuda.empty_cache()

		input_ids_sample = input_ids[size:,:].cuda()
		attention_mask_sample = attention_mask[size:,:].cuda()
		out,pool = model(input_ids=input_ids_sample , attention_mask=attention_mask_sample,return_dict=False)

		# delete these to free up cuda
		del input_ids_sample
		del attention_mask_sample

		pool_out[size:,:]   = pool.detach().cpu()

		del out
		del pool
	
	return pool_out



# files = ['reviews_Electronics_5',
# 	'reviews_Electronics_5_1',
# 	'reviews_Electronics_5_2',
# 	'reviews_Electronics_5_3',
# 	'reviews_Electronics_5_4',
# 	'reviews_Electronics_5_5',
# 	'reviews_Electronics_5_6',
# 	'reviews_Electronics_5_7',
# 	'reviews_Electronics_5_8',
# 	'reviews_Electronics_5_9',
# 	'reviews_Electronics_5_10',
# 	'reviews_Electronics_5_11',
# 	'reviews_Electronics_5_12',
# 	'reviews_Electronics_5_13',
# 	'reviews_Electronics_5_14',
# 	'reviews_Electronics_5_15'
# ]

# dir = '/mnt/ramdisk/Electronics/5-core'

# for i in files:
# 	data = unpickle_data(f'{dir}/{i}.pickle')
# 	data['reviewText'] = encode_data(data['reviewText'])
# 	pickle_data(data, f'{dir}/{i}_tokenized.pickle')