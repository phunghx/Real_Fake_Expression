import glob
import numpy as np
import os
import pickle
import sys
#sys.path.append('/data4/xgboost/python-package')
from xgboost import XGBClassifier

facials = ['anger','contentment','disgust','happy','sadness','surprise']
result = {}
for facial in facials:
	X = []
	pred_files = []
	filenames = os.listdir(facial + '/data')
	for filename in filenames:
		parts = filename.split('_')
		name = '' + parts[0] + '_' + parts[1] + '.mp4'
		pred_files.append(name)
		X.append(np.load(facial + '/data/' + filename).flatten().tolist())
	X = np.array(X)
	if len(X) <=0:	continue
	model = pickle.load(open(facial + '/model.dat','rb'))
	pred_f = model.predict(X)
	for i in range(len(pred_f)):
		
		if pred_f[i] == 0: result[pred_files[i]] = 'true'
		else:	result[pred_files[i]] = 'fake'

pickle.dump(result, open("test_prediction.pkl","wb"))


