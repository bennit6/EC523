import torch
import torch.nn as nn

# TODO:
#	- rewrite and test in torch

m = nn.Softmax(dim=1).cuda()

def gen_MSE_Vec(labels, scale=3, n=5):
	size = len(labels)
	vec = torch.rand((size,n),device=torch.device('cuda'))
	vec[list(range(size)),labels] = scale
	return m(vec)


class MSE_Vec_matrix():
	def __init__(self,edge=0.1,n=5):
		if edge >1 or edge < 0:
			ValueError(f'Mid should be in the range [0,1], but got {mid}')
		if n < 0:
			ValueError(f'N must be a positive number, but got {n}')
		elif not isinstance(n, int):
			ValueError(f'N must be an integer, but got {n}')
		
		indices = torch.tensor(range(n))
		index_mat = torch.eye(n)
		index_mat[indices[1:],indices[:-1]] = edge
		index_mat[indices[:-1],indices[1:]] = edge
		
		self.__matrix = index_mat
	
	def __getitem__(self,index):
		return self.__matrix[index,:]
	
	def __repr__(self):
		return str(self.__matrix)
	
	def to(self,device):
		self.__matrix = self.__matrix.to(device)
	
	def cpu(self):
		self.__matrix = self.__matrix.cpu()
	
	def cuda(self):
		self.__matrix = self.__matrix.cuda()
		

if __name__ == '__main__':
	mat = MSE_Vec_matrix()
	print(mat[1])
	print('\n\n')
	print(mat)
