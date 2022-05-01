import __init__ as dp
import copy
from multiprocessing import Process

def func (data,num):
	data['reviewText'] = [dp.preprocessing(text,i) for text in data['reviewText']]
	dp.pickle_data(data,f'{path}/{fname}_{i}.pickle')
	pass

path = '/mnt/ramdisk/Electronics/5-core'
fname = 'reviews_Electronics_5'

data = dp.unpickle_data(f'{path}/{fname}.pickle')

print(type(list(data['reviewText'])))


processes = []

for i in range(1,0b1111 + 1):
	process = Process(target=func,args=[data,i])
	process.start()
	processes.append(process)

for process in processes:
	process.join()
