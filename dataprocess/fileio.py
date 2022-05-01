import pickle
import lzma

def pickle_data(data,file):
	with lzma.open(file,mode='wb',preset=9) as store_file:
		 pickle.dump(data, store_file,protocol=4)
def unpickle_data(file):
	with lzma.open(file,mode='rb') as store_file:
		data = pickle.load(store_file)
	return data