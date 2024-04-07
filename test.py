
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
import numpy as np

path = 'data/log.txt'
def test(model, dataset_name, data):

	out = model(data['x'])
	pred = out.argmax(dim=1)
	data['pred']  = pred
	pickle.dump(data,open('data/'+dataset_name+'/'+dataset_name+'.pl', 'wb'))
	correct = pred==data['y']
	accuracy = correct.sum()/correct.shape[0]
	with open(path, 'a') as f:
		f.write(f'Experiment {dataset_name}\n')
		f.write(f'accuracy = {accuracy}\n')

	return accuracy
