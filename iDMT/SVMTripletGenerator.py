# SVM Triplet generator module for iDMT
# DMT is originally by: Jedrzej
# Author: Kemal Berk Kocabagli 

from misc import *
from sklearn import svm # for building Support Vector Machines to use in DMT
import h5py # numerical data stored in binary, can be manipulated by NumPy easily
# detailed info here: http://docs.h5py.org/en/latest/quick.html

# generates a classification boundary using linear SVM given positive and negative data and a regularization parameter
def generateSVM(pos_data, neg_data, regularization_param):
	pos_labels = np.ones(len(pos_data[:,0])) # generate labels for positive instances
	neg_labels = -1*np.ones(len(neg_data[:,0])) # generate labels for negative instances
	train_X = np.concatenate((pos_data,neg_data),axis=0) # combine all training data
	train_Y = np.concatenate((pos_labels, neg_labels), axis=0) # combine all labels

    # Scikit Support Vector Classification (SVC)
	clf = svm.SVC(regularization_param, kernel='linear')
	clf.fit(train_X,train_Y)  # train the SVM

	n,p = np.shape(train_X) # n = number of data points, p = number of features for each data point
	#print(n,p)

	SV_indices = clf.support_ # indices of support vectors
	SVectors = clf.support_vectors_ # support vectors
	weights = clf.coef_[0] # (available if linear kernel) - weights assigned to features
	bias = clf.intercept_[0] # constants in the decision fct = b

	hyperplane=np.append(bias, weights) # DECISION BOUNDARY
	print("Support vectors: " + str(SVectors))
	#print(bias, weights)
	print("Hyperplane"+ str(hyperplane))
	#test = [[1,1],[4,5],[1,2],[2.34,-2],[1,1.5],[0,0]]
	#print(clf.predict(test))

	return hyperplane

########################################################

# returns the result of the binary classification:
# sign(xw+b)
def SVMpredict(testSet,weights,bias):
	return np.sign(np.dot(testSet, weights)+bias*np.ones(len(testSet)))

########################################################

def getNegativeData(negative_class_set, classes, howManyClasses, perClassCount, dataset_name, image_fv_dictionaries_PATH, model_name):
	neg_data = []
	#print("Negative classes used are:")
	for i in range(howManyClasses):
		neg_class = random.choice(negative_class_set) # PICK CLASS (do it howManyClasses times)
		negative_class_set.remove(neg_class)
		#print(classes[neg_class])
		with(open(image_fv_dictionaries_PATH+"/" + model_name + "_on_" + dataset_name + "_FV_dict_class_" + str(neg_class), "rb")) as openfile:
			neg_dict=pickle.load(openfile) # get the feature vectors for that class
		for j in range(perClassCount): neg_data.append(random.choice(neg_dict.values())) # get perClassCount random feature vectors
	print("Negative data gathered.")
	return neg_data


# generates SVM triplets [w0,ws,w*] to be used in Model Regression Network
def generateSVMTriplets(dataset_name, classes_PATH, whichClasses, image_fv_dictionaries_PATH, model_name, S):
	print("Generating SVM triplets to be used in Model Regression Network...")
	with(open(classes_PATH)) as openfile:
			classes=pickle.load(openfile)
	numClasses= len(classes) # number of classes in source data

	SVM_triplets = [] # [[w0,ws,w*], [] ... []]
	K = 6 # positive data amount for few shot learning
	#regularization_param = 0.1 # regularization parameter for linear SVM
	regularizationParams = [0.01,0.1,1]

	# GENERATE S SVM TRIPLETS FOR EACH CLASS IN THE TOP N LIST
	class_count = 0
	for class_no in whichClasses:
		print("SVM triplet creation began for class: " + str(classes[class_no]))
		with(open(image_fv_dictionaries_PATH+"/" + model_name + "_on_" + dataset_name + "_FV_dict_class_" + str(class_no), "rb")) as openfile:
			image_fv_dict=pickle.load(openfile)
		
		'''
		FOR POSITIVE DATA, WE NEED TO CONSIDER 3 SETTINGS:
		1: ONE-SHOT - PICK ONLY ONE POSITIVE AND TRAIN SVM (do it S times)
		2: FEW-SHOT - PICK K POSITIVES AND TRAIN SVM (do it S times)
		3: CLASSIC - PICK ALL THE POSITIVES AND TRAIN SVM (do it once)
		'''
		
		# FOR NEGATIVE DATA, PICK 10 DIFFERENT CLASSES AND FROM EACH CLASS PICK 50 IMAGES = 500 NEGATIVES IN TOTAL
		neg_set = [i for j in (range(class_no), range(class_no+1, numClasses)) for i in j]
		neg_data = getNegativeData(neg_set,classes,10, 50,dataset_name, image_fv_dictionaries_PATH, model_name)

		for i in range(S):
			print("subsampling " + str(i+1) + " for class " + str(classes[class_no]))

			pos_data_fewshot = []
			for j in range(K): pos_data_fewshot.append(random.choice(image_fv_dict.values()))
			#print("Positive data gathered for few shot SVM.")
			for regularization_param in regularizationParams:
				# GENERATE THE SVM TRIPLET
				w_zero = generateSVM(np.asarray([random.choice(image_fv_dict.values())]), np.asarray(neg_data), regularization_param)
				w_s = generateSVM(np.asarray(pos_data_fewshot), np.asarray(neg_data), regularization_param)
				w_star = generateSVM(np.asarray(image_fv_dict.values()), np.asarray(neg_data), regularization_param)
				# APPEND THE TRIPLET TO THE LIST
				SVM_triplets.append([w_zero, w_s, w_star])

		print("SVM triplets generated for class: " + str(classes[class_no]))
		
		'''
		# TEST FOR HOW SIMILAR GENERATED CLASSIFIERS ARE TO EACH OTHER, WE DO NOT WANT THEM TO BE TOO SIMILAR	
		CS = 0
		for i in range(S* len(regularizationParams)):
			for j in range(S*len(regularizationParams)):
				if (i!=j):
					CS += (1 - sp.distance.cosine(SVM_triplets[S* len(regularizationParams)* class_count + i][1], SVM_triplets[j][1]))
		print(1.0*CS/(count*(count-1)))
		'''
	
	print(len(SVM_triplets))

		# SAVE SVM TRIPLETS AS PICKLES/H5 files networkInput = 4097 dim, label = 4097 dim, networkOutput = 4097 dim


# CODE FOR DEBUG PURPOSES
#posData= np.array([[1,2],[2,2]])
#negData= np.array([[1,1],[2,1]])
#generateSVM(posData, negData, 0.1)




    