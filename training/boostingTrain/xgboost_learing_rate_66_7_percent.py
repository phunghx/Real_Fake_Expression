import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sys
#sys.path.append('/client/data4/xgboost/python-package')
from xgboost import XGBClassifier
import sklearn
import random
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.ensemble import AdaBoostClassifier
subjects = range(1,41)
train,test = cross_validation.train_test_split(subjects, train_size = 0.8, random_state=1)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation   import StratifiedKFold

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

def evalauc(preds, dtrain):
    
    labels = dtrain.get_label()
    
    return 'roc',  roc_auc_score(labels, preds)

def do_train(X, Y,subjectIndex, initial_eta, min_eta, verbose=True):

    np.random.seed( 1 )
    random.seed(    1 )
    subjects = np.array(range(1,41))
    cv_scores    = []
    train_scores = []
    
    split = StratifiedKFold(subjects, 5, shuffle=True )
    
    fold = 0
    
    for train_index, cv_index in split:
    
        fold = fold + 1
                    
        y_pred              = []
    	sTrain = subjects[train_index]
	sVal = subjects[cv_index]
	
	train_index = np.zeros(len(Y),dtype=np.bool)
	val_index = np.zeros(len(Y),dtype=np.bool)
	for ii in range(len(subjectIndex)):
		if subjectIndex[ii] in sTrain:
			train_index[ii] = True
		else:
			val_index[ii] = True

        X_train, X_valid    = X[train_index,:], X[val_index,:]
        y_train, y_valid    = Y[train_index],   Y[val_index]
        
        params = {
                "max_depth"             : 2, 
                "eta"                   : initial_eta,
                "min_eta"               : min_eta,
                "eta_decay"             : 0.5,
                "max_fails"             : 3,
                "early_stopping_rounds" : 500,
                "objective"             : 'binary:logistic',
                "subsample"             : 0.8, 
                "colsample_bytree"      : 1,
                "n_jobs"                : -1,
                "n_estimators"          : 5000, 
                "silent"                : 1,
		"lambda"		: 1000
        }
    
        num_round       = params["n_estimators"]
        eta             = params["eta"]
        min_eta         = params["min_eta"]
        eta_decay       = params["eta_decay"]
        early_stop      = params["early_stopping_rounds"]
        max_fails       = params["max_fails"]
        
        params_copy     = dict(params)
        
        dtrain          = xgb.DMatrix( X_train, label=y_train ) 
        dvalid          = xgb.DMatrix( X_valid, label=y_valid )  
    
        total_rounds        = 0
        best_rounds         = 0
        pvalid              = None
        model               = None
        best_train_score    = None
        best_cv_score       = None
        fail_count          = 0
        best_rounds         = 0
        best_model          = None
        
        while eta >= min_eta:           
            
            model        = xgb.train( params_copy.items(), 
                                      dtrain, 
                                      num_round, 
                                      [(dtrain, 'train'), (dvalid,'valid')], 
                                      early_stopping_rounds=early_stop,
                                      feval=evalauc )
    
            rounds          = model.best_iteration + 1
            total_rounds   += rounds
            
            train_score = roc_auc_score( y_train, model.predict(dtrain, ntree_limit=rounds) )
            cv_score    = roc_auc_score( y_valid, model.predict(dvalid, ntree_limit=rounds) )
    
            if best_cv_score is None or cv_score > best_cv_score:
                fail_count = 0
                best_train_score = train_score
                best_cv_score    = cv_score
                best_rounds      = rounds
                best_model       = model

                ptrain           = best_model.predict(dtrain, ntree_limit=rounds, output_margin=True)
                pvalid           = best_model.predict(dvalid, ntree_limit=rounds, output_margin=True)
                
                dtrain.set_base_margin(ptrain)
                dvalid.set_base_margin(pvalid)
            else:
                fail_count += 1

                if fail_count >= max_fails:
                    break
    
            eta                 = eta_decay * eta
            params_copy["eta"]  = eta
    
        train_scores.append(best_train_score)
        cv_scores.append(best_cv_score)

        print("Fold [%2d] %9.6f : %9.6f" % ( fold, best_train_score, best_cv_score ))
        
    print("-------------------------------")
    print("Mean      %9.6f : %9.6f" % ( np.mean(train_scores), np.mean(cv_scores) ) )
    print("Stds      %9.6f : %9.6f" % ( np.std(train_scores),  np.std(cv_scores) ) )
    print("-------------------------------")
    return  best_model   
# ----------------------------f----------------------------------------------------
#



classesList = {'anger':['N2A.npy','H2N2A.npy'],
		'contentment':['N2C.npy','H2N2C.npy'],
		'disgust':['N2D.npy','H2N2D.npy'],
		'happy':['N2H.npy','S2N2H.npy'],
		'sadness':['N2S.npy','H2N2S.npy'],
		'surprise':['N2Sur.npy','D2N2Sur.npy']}

LRs = {'anger':0.55,
       'contentment':0.1,
       'disgust':0.5,
       'happy':0.1,
       'sadness':0.4,
       'surprise':0.01}
ntree = {'anger':1500,
       'contentment':1000,
       'disgust':1000,
       'happy':1000,
       'sadness':1000,
       'surprise':1200}


for facial in classesList:
	#if facial != 'surprise': continue
	X = []
	y = []
	X_test = []
	y_test = []
	trainPath = '../results_' + facial + '/'

	valPath = '../../' + facial + '/dataVal'
	classes = classesList[facial] #['N2A.npy','H2N2A.npy']
	classesName = {}
	classesName[classes[0]] = 0
	classesName[classes[1]] = 1

	X_val = []
	y_val = []
	subjectIndex = []
	trainSubjects = os.listdir(trainPath)
	for subject in trainSubjects:
		x = np.load(trainPath + subject)
		filename = subject.split("_")[2]
		su = int(subject.split("_")[0])
		testing = subject.split("_")[1]
		#if su in train:
		X.append(x.flatten().tolist())
		y.append(classesName[filename])
		subjectIndex.append(su)				
		if (su in test):# and (testing=='test'):
			X_test.append(x.flatten().tolist())
			y_test.append(classesName[filename])
	'''
	objectVals = os.listdir(valPath)
	for ob in objectVals:
		obs = ob + '.mp4'
		if val_labels[obs]=='true': la = 0
		else:	la = 1
		filenames = os.listdir(valPath + '/' + ob)
		for filename in filenames:
			x = np.load(valPath + '/' + ob + '/'+ filename)
			X_val.append(x.flatten().tolist())
			y_val.append(la)
	X_val = np.array(X_val)
	y_val = np.array(y_val)
	'''
	X = np.array(X)
	y = np.array(y)
	subjectIndex = np.array(subjectIndex)
	X_test = np.array(X_test)
	y_test = np.array(y_test)
	eval_set = [(X_test, y_test)]
	#xgdmat = xgb.DMatrix(X,y)
	#testdmat = xgb.DMatrix(X_val)
	model = do_train(X,y,subjectIndex,LRs[facial],0.00001)
	#y_pred = model.predict(testdmat)
	#thresh=0.5;y_p = y_pred.copy();y_p[y_p>thresh] = 1;y_p[y_p<=thresh] = 0;accuracy_score(y_p, y_val);
	
	#print(facial + ":" + str(accuracy_score(y_p, y_val)))
	model.save_model("../../" +facial + "/model2.dat")
	#import pdb;pdb.set_trace()
	#model1 = AdaBoostClassifier(n_estimators=1000)
	#model2 = RandomForestClassifier(n_estimators=1000)
	#model = VotingClassifier(estimators=[('lr', model1), ('rf', model2)], voting='hard')
	#model = XGBClassifier(learning_rate=LRs[facial],silent=True,n_estimators=ntree[facial],seed=1)
	#model = XGBClassifier()
	#ver = False
	#if facial=='surprise':
	#	ver = True
	#model.fit(X, y,early_stopping_rounds=500,  eval_metric="auc", eval_set=eval_set, verbose=ver)
	#model.fit(X,y)
	#pickle.dump(model, open("../../" +facial + "/model.dat","wb"))
        #import pdb;pdb.set_trace()
	'''
	learning_rates = []
	lr = LRs[facial]
	for i in range(ntree[facial]):
		learning_rates.append(lr)
		lr = lr * 0.99

	#import pdb;pdb.set_trace()
	our_params = {'eta':LRs[facial],'seed':0,'subsample':0.9,'colsample_bytree':1,'objective':'binary:logistic','max_depth':3,'min_child_weight':1,
	'silent':1}

	#cv_xgb = xgb.cv(params=our_params,dtrain=xgdmat,num_boost_round=1200,nfold=5,metrics=['error'])
	#import pdb;pdb.set_trace()
	
	model = xgb.train(our_params, xgdmat,num_boost_round=ntree[facial], callbacks=[xgb.callback.reset_learning_rate(learning_rates)])
	testdmat = xgb.DMatrix(X_test)
	y_pred = model.predict(testdmat)
	thresh=0.5;y_p = y_pred.copy();y_p[y_p>thresh] = 1;y_p[y_p<=thresh] = 0;accuracy_score(y_p, y_test);
	#import pdb;pdb.set_trace()
	print(facial + ":" + str(accuracy_score(y_p, y_test)))
	#model.save_model("../../" +facial + "/model.dat")
	'''	


