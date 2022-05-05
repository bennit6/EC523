# from __init__ import *
import torch
import numpy as np
from transformers import BertTokenizer, FNetModel
from transformers import BertModel
from transformers import FNetTokenizer
from progress.bar import Bar

class CustomBar(Bar):
	fill='█'
	suffix=r'%(percent).1f%% - %(remaining_hours)dh  %(remaining_mins)dm %(remaining_seconds)ds'

	@property
	def remaining_hours(self):
		return self.eta // 3600
	@property
	def remaining_mins(self):
		return (self.eta - self.remaining_hours * 3600) // 60
	
	@property
	def remaining_seconds(self):
		return (self.eta - self.remaining_hours*3600 - self.remaining_mins*60) // 3600


def tokenize_data(data, model_name="fnet"):
	if model_name == "fnet":
		encoding = FNetTokenizer.from_pretrained('google/fnet-base')
		data_encode = encoding(data, padding='max_length', max_length=400, truncation=True, return_tensors='pt')
	elif model_name == "bert":
		encoding = BertTokenizer.from_pretrained('bert-base-cased')
		data_encode = encoding(data, padding='max_length', max_length=400, truncation=True, return_tensors='pt')
	return data_encode

def encode_data(input_ids,attention_mask,batch_size=5, model_name="fnet"):

	if model_name == "fnet":
		model = FNetModel.from_pretrained('google/fnet-base').eval().cuda()
	elif model_name == "bert":
		model = BertModel.from_pretrained('bert-base-cased').eval().cuda()

	index_base = torch.tensor(list(range(batch_size)))
	model = BertModel.from_pretrained('bert-base-cased').cuda()

	size = input_ids.shape[0]
	num_words = input_ids.shape[1]

	pool_out = torch.zeros((size,768))

	bar = CustomBar('Loading',max=size)

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

		bar.next(n=batch_size)

	if size//batch_size != size/batch_size:

		index = (size//batch_size) * batch_size

		torch.cuda.empty_cache()

		input_ids_sample = input_ids[index:,:].cuda()
		attention_mask_sample = attention_mask[index:,:].cuda()
		out,pool = model(input_ids=input_ids_sample , attention_mask=attention_mask_sample,return_dict=False)

		# delete these to free up cuda
		del input_ids_sample
		del attention_mask_sample

		pool_out[index:,:]   = pool.detach().cpu()

		del out
		del pool

		bar.finish()
	
	return pool_out