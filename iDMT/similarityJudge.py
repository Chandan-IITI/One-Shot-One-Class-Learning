# Similarity judge module for iDMT
# DMT is originally by: Jedrzej
# Author: Kemal Berk Kocabagli 

from misc import *
from featureExtractor import * # to extract the features for the novel image

# finds the average cosine similarity between the feature vectors of a reference image and randomly selected K images from a class

'''
# THE ON-THE-RUN VERSION, DOES NOT USE THE ALREADY CREATED FEATURE VECTORS
def findAverageCosineSimilarity(reference_image_FV, class_path, model,weights,target_layer, network_input_width, network_input_height, avgBGR, howManyInstances=10):
	CS=0
	class_images =[]
	for img_path in glob.glob(class_path+"/*"): class_images.append(img_path)
	for i in range(howManyInstances):
		randomImg_path = random.choice(class_images)
		randomFV = getFeatureVector(randomImg_path,model,weights,target_layer, network_input_width, network_input_height, avgBGR)
		CS += (1 - sp.distance.cosine(reference_image_FV, randomFV))
	return 1.0*CS/howManyInstances

'''
########################################################

def findAverageCosineSimilarity(reference_image_FV, class_no, class_FV, howManyInstances=100):
	CS=0
	zeros=''
	for i in range (3- int(np.ceil(np.log10(class_no+1+1)))): zeros += '0'
	classkeys = class_FV.keys()
	for i in range(howManyInstances):
		# IF INTERESTED IN A SPECIFIC IMAGE (WROTE FOR SANITY CHECK), TAILORED TO CALTECH-256
		#ind = zeros+str(class_no+1)+'_0007'
		ind = random.choice(classkeys)
		# OPTIONALLY, do the randomization without replacement
		#classkeys.remove(ind)
		#if(len(classkeys)==0): classkeys=class_FV.keys()
		CS += (1 - sp.distance.cosine(reference_image_FV, class_FV[ind]))
	return 1.0*CS/howManyInstances


########################################################
'''
@params:
	RI_PATH: path of the reference image
	dataset_name: name of the source dataset
	dataset_PATH: path of the source dataset
	classes: list of classes in the source data
	image_fv_dictionaries_PATH: lists of feature vectors of images in the source data (path to /FeatureVectors)
	avgBGR: the average [blue,green,red] value of the source dataset
	N: the number of most similar classes to be found
	model_name: name of the feature extraction model
	model: path of the deploy.prototxt of the Caffe model
	weights: path of the .caffemodel of the Caffe model
	target_layer: the layer that will be used to extract features for similarity comparison
	network_input_width: data input width of the feature extraction model
	network_input_height: data input height of the feature extraction model
@return
	top N most similar classes from the source dataset to the reference image
'''

def findMostSimilarClasses(RI_PATH, dataset_name, dataset_PATH, classes, image_fv_dictionaries_PATH, avgBGR, N, model_name, model, weights, target_layer, network_input_width, network_input_height):
	print("Finding the most similar " + str(N) + " classes...")
	numClasses = len(classes)

	# get the feature vector for the reference image
	reference_image_FV = getFeatureVector(RI_PATH,model,weights,target_layer, network_input_width, network_input_height, avgBGR)
	# initialize the cosine similarities list where element i represents the
	# average cosine similarity between the feature vector of the reference image and
	# feature vectors of randomly selected images from class i.
	cosine_similarities = np.zeros(numClasses);

	# print the name of the novel image for debug purposes
	print("Novel image: " + RI_PATH.replace("./",""))

	'''
	# CODE FOR THE ON-THE-RUN VERSION, TAKES MUCH LONGER TIME
	class_count = 0
	for class_path in glob.glob(dataset_PATH+"/*"):
		print("Processing class " + str(class_count) + "...")
		cosine_similarities[class_count]=findAverageCosineSimilarity(reference_image_FV, class_path, model,weights,target_layer, network_input_width, network_input_height, avgBGR)
		class_count +=1
		if class_count==5: break
	'''

	# increasing repeat might give better results by decreasing variance caused by random sampling but takes more time
	repeat = 1
	for i in range(repeat):
		for class_ in range(0,numClasses): # for each class in the source dataset
			print("Similarity judge processing class " + str(class_) + "...")
			with(open(image_fv_dictionaries_PATH+ "/"+ model_name + "_on_" + dataset_name + "_FV_dict_class_" + str(class_), "rb")) as openfile:
					class_FV=pickle.load(openfile) # get the feature vectors for that class
			cosine_similarities[class_]+=findAverageCosineSimilarity(reference_image_FV, class_, class_FV)	

	#print(cosine_similarities)
	topN = cosine_similarities.argsort()[-N:][::-1] # sort arguments in ascending order and then reverse list
	print(topN) # topN should contain N indices of classes that are most similar to our reference image
	for arg in topN: print(classes[arg])
	#for arg in topN: print(cosine_similarities[arg]),
	print("Found the most similar " + str(N) + " classes. Layer used: " + target_layer +  " of " + model_name)
	print("Source dataset: " + dataset_name + ", which contains " + str(numClasses) +  " classes in total.")

	return topN