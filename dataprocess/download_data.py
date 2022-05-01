import requests
from os.path import isdir
import os
import gzip
import shutil
import lzma
import json
import pickle
import pandas as pd
from multiprocessing import Process

categories = [
	# 'Books',
	'Electronics',
	# 'Movies and TV',
	# 'CDs and Vinyl',
	# 'Clothing, Shoes and Jewelry',
	# 'Home and Kitchen',
	# 'Kindle Store',
	# 'Sports and Outdoors',
	# 'Cell Phones and Accessories',
	# 'Health and Personal Care',
	# 'Toys and Games',
	# 'Video Games',
	# 'Tools and Home Improvement',
	# 'Beauty',
	# 'Apps for Android',
	# 'Office Products',
	# 'Pet Supplies',
	# 'Automotive',
	# 'Grocery and Gourmet Food',
	# 'Patio, Lawn and Garden',
	# 'Baby',
	# 'Digital Music',
	# 'Musical Instruments',
	# 'Amazon Instant Video'
]

data_modes = ['5-core', 'ratings only']

def pickle_data(data,file):
	with lzma.open(file,mode='wb',preset=9) as store_file:
		 pickle.dump(data, store_file,protocol=4)


class AmazonReveiws():
	def __init__(self, category, root_dir='.', data_mode='5-core' ,transform=None, split=(80,10,10)):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""

		# creates root dir if does not exist
		if isdir(root_dir):
			self.root_dir = root_dir
		else:
			os.makedirs(root_dir)
			self.root_dir = root_dir

		# ensures a valid category is selected
		if category in categories:
			self.category = category.replace(',','').replace(' ', '_')
		else:
			raise ValueError(f'{category} is not a valid category')
		
		# ensures correct data mode is selected
		if data_mode in data_modes:
			self.data_mode = data_mode
		else:
			raise ValueError(f'{data_mode} is not a valid mode \n valid modes are {data_modes[0]} & {data_modes[1]}')
		
		self.__data_dir_base = f'{self.root_dir}/{self.category}/{self.data_mode}'
		
		# checks if data is downloaded
		if not isdir(self.__data_dir_base):
			os.makedirs(self.__data_dir_base)
			self.__download_data()
		

	def __download_data(self):

		if self.data_mode == '5-core':
			uri = f'reviews_{self.category}_5'
		else:
			uri = f'reviews_{self.category}'

		url = f'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{uri}.json.gz'

		request = requests.get(url)

		with open(f'{self.__data_dir_base}/{uri}.json.gz', 'wb') as file1:
			file1.write(request.content)

		with gzip.open(f'{self.__data_dir_base}/{uri}.json.gz', 'rb') as f_in:
			with open(f'{self.__data_dir_base}/{uri}.json', 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)

		json_file = open(f'{self.__data_dir_base}/{uri}.json', 'r')

		data = []

		keys = ['helpful','overall','reviewText']

		for line in json_file.readlines():
			data.append(json.loads(line))

		# converts list of dictionaries to dictionary of lists
		data = {k: [dic[k] for dic in data] for k in keys}

		pickle_data(data, f'{self.__data_dir_base}/{uri}.pickle')

# if __name__ == '__main__':

proceesses = []

# for i in categories:
# 	AmazonReveiws(i,root_dir='/mnt/ramdisk')

for i in categories:
	func = lambda x : AmazonReveiws(x,root_dir='/mnt/ramdisk')
	process = Process(target=func,args=[i])
	process.start()
	proceesses.append(process)

for i in proceesses:
	i.join()


print('test')