from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

def F_score(net, inputs, labels):
	y_pred = []
	y_test = []
	
	output = net(inputs)
	output = torch.argmax(output).cpu().numpy()
	y_pred.extend(output)
	
	labels = labels.cpu().numpy()
	y_true.extend(labels)

	classes = ('1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars')
	cf_matrix = confusion_matrix(y_true, y_pred)
	df_cm = pd.Dataframe(cf_matrix/np.sum(cf_matrix)*10, index=[i for i in classes], columns = [i for i in classes])
	plt.figure(figsize = (12,7))
	sn.heatmap(df_cm, annot=True)
	plt.savefig('output.png')
