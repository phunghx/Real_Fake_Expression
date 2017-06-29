import glob
import numpy as np
import os
import pickle
import sys
#sys.path.append('/data4/xgboost/python-package')
from xgboost import XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt

facials = ['anger','contentment','disgust','happy','sadness','surprise']
result = {}

pairsImage = pickle.load(open('vallidPairs.pkl','rb'))
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

def evalauc(preds, dtrain):
    
    labels = dtrain.get_label()
    
    return 'roc', - roc_auc_score(labels, preds)
def majority(pred):
	pred = pred.mean()
	if pred > 0.5: return 1
	else:	return 0
	'''
	return pred.mean(axis=0).argmax()
	count = [0,0]
	for i in range(len(pred)):
	   count[pred[i]] =count[pred[i]] + 1
	if count[0] > count[1]:
		return 0
	else:
		return 1

	'''

for facial in facials:
	model = xgb.Booster()
	
	model.load_model(facial + '/model2.dat')
	subjects = os.listdir(facial + '/dataVal')
	
	sub = []
	while(len(subjects) > 0):
		s1 = subjects[0]
		for i in range(1,len(subjects)):
			if subjects[i] == pairsImage[s1]:
				break
		s2 = pairsImage[s1]
		subjects.pop(i)
		subjects.pop(0)
		name1 = '' + s1 + '.mp4'
		filenames = os.listdir(facial + '/dataVal/' + s1)
		X = []
		for filename in filenames:
			X.append(np.load(facial + '/dataVal/'+ s1 + '/' + filename).flatten().tolist())
		X = np.array(X)
		pred_f = model.predict(xgb.DMatrix(X))
		pred1 = pred_f.mean()

		name2 = '' + s2 + '.mp4'
		filenames = os.listdir(facial + '/dataVal/' + s2)
		X = []
		for filename in filenames:
			X.append(np.load(facial + '/dataVal/'+ s2 + '/' + filename).flatten().tolist())
		X = np.array(X)
		pred_f = model.predict(xgb.DMatrix(X))
		pred2 = pred_f.mean()
		if pred2> pred1:
			result[name2] = 'fake'
			result[name1] = 'true'
		else:
			result[name2] = 'true'
			result[name1] = 'fake'
		#showFaces(s1,s2,'/server/dataset/Real_Fake/data/validImage/')


		
	'''
				
	for subject in subjects:
		X = []
		pred_files = []
		name = '' + subject + '.mp4'
		filenames = os.listdir(facial + '/dataVal/' + subject)
		for filename in filenames:
			X.append(np.load(facial + '/dataVal/'+ subject + '/' + filename).flatten().tolist())
		X = np.array(X)
		pred_f = model.predict(xgb.DMatrix(X))
		pred = majority(pred_f)
		if pred == 0: result[name] = 'true'
		else:	result[name] = 'fake'
	'''
pickle.dump(result, open("valid_prediction.pkl","wb"))



