# Informed DMT implementation for one-shot-one-class learning
# DMT is originally by: Jedrzej
# Author: Kemal Berk Kocabagli 

from misc import *
from SVMTripletGenerator import *
from similarityJudge import *

# extract features from the images in source data using featureExtractor.py before starting
# once you're done with feature extraction, you should have a feature vector representation for each image in your dataset

##########
##### MAIN #####
##########

def main():
	if (len(sys.argv)<3):
		print("Usage: python iDMT.py <positive_img_path> <N>")
		return 0

	PI_PATH= sys.argv[1] # THE SINGLE POSITIVE IMAGE IN ONE-SHOT-ONE-CLASS SETTING
	POSITIVE_IMG = cv2.imread(PI_PATH) # read the positive image
	#displayImage(PI_PATH,POSITIVE_IMG, 0.4) # optionally, display it

	N = int(sys.argv[2]) # param to determine top N most similar classes to be considered in iDMT

	# PRE-CALCULATED AVERAGE BGR VALUES OF SOME DATASETS
	Caltech256_avg = np.array([ 128.78144029, 136.06747341, 140.76367044]) 
	ILSVRC2012_avg = np.array([93.5940,104.7624,129.1863])  

	# MODIFY IF NECESSARY
	# SOURCE DATA
	source_dataset_PATH = "/Users/berkkocabagli/Desktop/UCSB_Research_Summer2017/Datasets/256_ObjectCategories" 
	source_dataset_avg_BGR = Caltech256_avg # This is for Caltech-256, can calculate using in featureExtractor.py if using another dataset
	source_dataset_name = "Caltech-256"
	numClasses = 256
	# FEATURE EXTRACTION MODEL TO BE USED IN TRANSFER LEARNING
	FE_model_name = "AlexNet"
	FE_model= "/Users/berkkocabagli/Desktop/UCSB_Research_Summer2017/Models/AlexNet/deploy.prototxt" 
	FE_weights= "/Users/berkkocabagli/Desktop/UCSB_Research_Summer2017/Models/AlexNet/bvlc_alexnet.caffemodel"
	FE_layer= "fc6"
	FE_layer_numFeatures = 4096
	similarity_FE_layer = "fc6" # choose an intermediate layer (not close to the final) to increase possibility of feature sharing
	similarity_FE_layer_numFeatures = 4096
	network_input_width = 227
	network_input_height = 227
	# subsampling amount in SVM triplet creation
	S = max(10,int(np.ceil(2*numClasses/N))) # as N increases, S decreases
	S = 1

	# UNCOMMENT IF YOU NEED TO CREATE THE CLASS DICTIONARY AND FEATURE VECTORS USING A NEW SOURCE DATA AND/OR FEATURE EXTRACTION MODEL
	
	# createClassDictionary(source_dataset_PATH, source_dataset_name, numClasses)
	#getFeatureVectors(source_dataset_PATH,source_dataset_name, numClasses, source_dataset_avg_BGR, FE_model_name, FE_model, FE_weights,FE_layer, FE_layer_numFeatures, network_input_width, network_input_height)
	# feature vectors for similarity judge (no need if similarity_FE_layer = FE_layer)
	#getFeatureVectors(source_dataset_PATH,source_dataset_name, numClasses, source_dataset_avg_BGR, FE_model_name, FE_model, FE_weights,similarity_FE_layer, similarity_FE_layer_numFeatures, network_input_width, network_input_height)

	classes_PATH = "./" + source_dataset_name + "_classes" # PATH FOR THE LIST OF CLASSES OF SOURCE DATA
	image_fv_dictionaries_PATH = "./FeatureVectors/" + FE_layer # PATH FOR THE FOLDER THAT CONTAINS IMAGE-FEATURE VECTOR DICTIONARIES
	similarity_image_fv_dictionaries_PATH = "./FeatureVectors/" + similarity_FE_layer # PATH FOR THE FOLDER THAT CONTAINS IMAGE-FEATURE VECTOR DICTIONARIES

	# LOAD CLASSES IN SOURCE DATASET
	with(open(classes_PATH)) as openfile:
			classes=pickle.load(openfile)

	# FIND MOST SIMILAR N CLASSES TO OUR NEW OBJECT (the list should contain integers, 0 is the first class, 1 is the second class and so on...)
	most_similar_N_classes = findMostSimilarClasses(PI_PATH, source_dataset_name, source_dataset_PATH, classes, similarity_image_fv_dictionaries_PATH, source_dataset_avg_BGR, N, FE_model_name, FE_model, FE_weights, similarity_FE_layer, network_input_width,network_input_height) 
	#most_similar_N_classes = [0,3,5]

	# CREATE SVM TRIPLETS USING ONLY THE MOST SIMILAR N CLASSES, WHICH WILL THEN BE USED TO TRAIN MODEL REGRESSION NETWORK
	#generateSVMTriplets(source_dataset_name,classes_PATH, most_similar_N_classes, image_fv_dictionaries_PATH, FE_model_name, S)


	# CREATE DECISION BOUNDARY FOR THE NOVEL IMAGE: W0
	# POSITIVE SET: ONE GIVEN IMAGE
	# NEGATIVE SET: RANDOM IMAGES FROM SOURCE CLASSES OTHER THAN THE TOP N MOST SIMILAR
	#print("Creating w_zero for the novel image...")
	#novel_image_FV = getFeatureVector(RI_PATH,model,weights,target_layer, network_input_width, network_input_height, avgBGR)
	#auxiliaryNegativeSet = range(0,numClasses)
	#for c in most_similar_N_classes: auxiliaryNegativeSet.remove(c)

	#print(auxiliaryNegativeSet)
	#print(len(auxiliaryNegativeSet))

	#novel_image_w_zero = generateSVM(np.asarray(novel_image_FV), ,regularization_param=0.01)
	#print(novel_image_w_zero)

	# USING THE T0, TS, FIND WS, W* FOR THE NOVEL OBJECT

	#with(open("./FeatureVectors/fc6/AlexNet_on_Caltech-256_FV_dict_class_1")) as openfile:
	#		c=pickle.load(openfile)
	#print(c['002_0005'])
	#print(c.keys())
	#print(c['005_0001'])
if __name__ == '__main__':
   main()