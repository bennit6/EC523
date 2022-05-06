from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
import numpy as np

def F_score(net, inputs, labels):
    y_pred = []
    y_true = []

    output = net(inputs)
    output = torch.argmax(output, dim=1).cpu().numpy()
    y_pred.extend(output)
    
    y_true.extend(labels.cpu().numpy())
    
    classes = ('1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars')
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    print(np.sum(cf_matrix,axis=1))
    df_cm = pd.DataFrame((cf_matrix.T/np.sum(cf_matrix,axis=1)).T, index=[i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")