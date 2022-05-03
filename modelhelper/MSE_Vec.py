import torch

# TODO:
#	- rewrite and test in torch

class MSE_Vec_matrix():
	def __init__(self,mid=0.8,n=5):
		if mid >1 or mid < 0:
			ValueError(f'Mid should be in the range [0,1], but got {mid}')
		if n < 0:
			ValueError(f'N must be a positive number, but got {n}')
		elif not isinstance(n, int):
			ValueError(f'N must be an integer, but got {n}')
		
		indices = torch.tensor(range(n))
		edge = (1-mid)/2
		index_mat = mid * torch.eye(n)
		index_mat[[0,-1],[0,-1]] = mid + edge
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
