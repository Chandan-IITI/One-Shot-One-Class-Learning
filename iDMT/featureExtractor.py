# Feature Extractor module for informed DMT
# DMT is originally by: Jedrzej
# Author: Kemal Berk Kocabagli 

from misc import *
import caffe  # for feature extraction using deepCNNs

# Returns a list of feature vectors given 
# a dataset (of images), 
# a Caffe model 
# and the name of the layer to be extracted from the model
def getFeatureVectors(dataset_PATH, dataset_name, numClasses, dataset_avg_BGR, model_name, model, weights, target_layer, numFeatures, network_input_width, network_input_height):
	start_time = time.clock()
	print("Getting feature vectors for dataset: ", dataset_name)

	batch_size=40 # batch size to be given to deepCNN
	image_dict = dict() # keys: image names, values: feature vectors (refreshed for each class)
	image_names = [] # the list of the names of all images (refreshed for each class)

	CLASS_LIMIT = numClasses # limit for number of classes to process
	IMAGE_LIMIT = 1000 # limit for number of images to process per class
	IMAGE_LIMIT_ON = 0 # is image limit on (if off, process all images per class)

	net = caffe.Net(model,weights,caffe.TEST) # Caffe network
	feature_vectors = np.zeros((0,numFeatures))  # the list for all the feature vectors (refreshed for each class)
	#[[numFeatures zeros], [numFeatures zeros] ... [numFeatures zeros]]

	class_count=0
	image_count=0
	image_in_class_count=0

	if(not os.path.exists("./FeatureVectors/"+ target_layer)):
		os.mkdir("./FeatureVectors/"+ target_layer)
	else:
		print("Features have already been extracted. Check the /FeatureVectors directory.")
		response = raw_input('If you want to overwrite, press y: ')
		if response != 'y': # if user wants to overwrite, run the code. Otherwise, return.
			return

	# PROCESS THE IMAGES CLASS BY CLASS

	for class_path in glob.glob(dataset_PATH+"/*"): # for each class in the dataset
		images_in_class = [] # list of images in the class 
		class_path += "/"
		class_count+=1
		if class_count>min(CLASS_LIMIT,numClasses): # disregard clutter in Caltech-256
				break
		#print("CLASS PATH" + class_path)
		for img_path in glob.glob(class_path+"*.jpg"): # for each image in the class
			image_in_class_count+=1
			img = processImage(img_path, network_input_width, network_input_height, dataset_avg_BGR) # process the image
			images_in_class.append(img) # add to images list
			image_names.append(img_path.replace(class_path,"").replace(".jpg","")) # add the image name
			if image_in_class_count == IMAGE_LIMIT and IMAGE_LIMIT_ON==1: # OPTIONAL - only process first K images per class
				break
			#if (image_in_class_count==1): # OPTIONAL - display the first preprocessed images for each class
			#	displayImage(img_path, img, 1)
		image_count+=image_in_class_count
		print("Class " + str(class_count-1) + " preprocessed. (" + str(image_in_class_count) + " images)")

		# GET FEATURE VECTORS FOR THE IMAGES USING THE GIVEN MODEL

		# blob = N-dimensional array in Caffe, convenient for both CPU and GPU computations (provides synchronization when needed)
		# BATCH PROCESSING yields BETTER THROUGHPUT
		'''
		 example 4D blob: (N,K,H,W)
		 where N = batch size
		 K = # of features (= 3 for RGB)
		 H = image height 
		 W = image width

	 	for a conv layer with S filters of K x K and N inputs,
		 the blob is S x N x K x K
		'''

		no_of_batches = int(np.ceil(1.0*image_in_class_count / batch_size)) # how many batches
		for batch_no in range(0,no_of_batches):
			size_of_current_batch = min(batch_size,(image_in_class_count-batch_no*batch_size))
			net.blobs['data'].reshape(size_of_current_batch, 3,network_input_width,network_input_height)
			batch = np.asarray(images_in_class[:size_of_current_batch])
			# ! ATTENTION: in Caffe, fully connected layers and rectified linear units are coupled. If you
			# go through the end, fc outputs will be relu outputs, not fc outputs!
			out = net.forward(blobs=[target_layer],start='data',end=target_layer,data=batch) # output of fc6 layer
			#print("OUTPUT FOR BATCH " + str(batch_no+1) + " for class " + str(class_count-1))
			#print(out[target_layer])
			feature_vectors = np.concatenate((feature_vectors,out[target_layer]),axis=0)  # add the feature vectors of the batch to the fv list
			del images_in_class[:size_of_current_batch] # delete the images that have been already processed

		print("Class " + str(class_count-1) + " :features extracted.")

		print("FEATURE VECTORS:")
		print(feature_vectors)

		# Fill the image dictionary with feature vectors
		for image in range(image_in_class_count):
			image_dict[image_names[image]]=feature_vectors[image]
		#print("IMAGE DICT:")
		#print(image_dict)

		# Save the image dictionary
		print("Saving the image-feature vector dictionary...")
		with open("./FeatureVectors/" + target_layer + "/" + model_name + "_on_" + dataset_name + "_FV_dict_class_" + str(class_count-1), 'wb') as outfile:
			pickle.dump(image_dict, outfile)

		image_in_class_count=0 # REFRESH VARIABLES AND LISTS
		image_dict = dict()
		image_names = []
		feature_vectors = np.zeros((0,numFeatures))

	print("Total # of processed images", image_count)

	print("Feature extraction complete. Time spent: " + str((time.clock()-start_time)) + " seconds")

########################################################

# Returns a feature vector given 
# a dataset (of images), 
# a Caffe model 
# and the name of the layer to be extracted from the model
def getFeatureVector(img_path, model,weights,target_layer, network_input_width, network_input_height, avgBGR):
	start_time = time.clock()
	net = caffe.Net(model,weights, caffe.TEST) # Caffe network

	#print("Getting feature vector...")

	#print("Processing image...")
	img = processImage(img_path, network_input_width, network_input_height, avgBGR)

	#displayImage(img_path, img, 1)
			
	net.blobs['data'].reshape(1, 3,network_input_width,network_input_height)
	# ! ATTENTION: in Caffe, fully connected layers and rectified linear units are coupled. If you
	# go through the end, fc outputs will be relu outputs, not fc outputs!
	out = net.forward(blobs=[target_layer],start='data',end=target_layer,data=np.asarray([img])) # get output of the target layer
	feature_vector = out[target_layer]  # retrieve the feature vector
	#print("Feature extraction complete. Time spent: " + str((time.clock()-start_time)) + "seconds")
	#print(feature_vector)
	print("Feature extraction complete. Time spent: " + str((time.clock()-start_time)) + " seconds")
	return feature_vector[0]

########################################################

def processImage(img_path, network_input_width, network_input_height, avgBGR):
	img = cv2.imread(img_path) # read
	img = img - avgBGR # center
	img = cv2.resize(img, (network_input_width+28,network_input_height+28)) # resize
	img = img[14:network_input_width+14, 14:network_input_height+14] # crop the center w-28 x h-28
	img = img.transpose((2,0,1)) # w x h x 3 into 3 x w x h
	return img

########################################################

def createClassDictionary(dataset_PATH, dataset_name, numClasses):
	start_time = time.clock()
	classes = []
	count = 0
	for class_path in glob.glob(dataset_PATH+"/*"):
		classes.append(class_path.replace(dataset_PATH+"/",""))
		count +=1
		if(count==numClasses): break
	with open("./" + dataset_name + "_classes", 'wb') as outfile:
		pickle.dump(classes, outfile)
	print("Class dictionary created in " + str(time.clock()-start_time) + " seconds")

########################################################

def findAvgBGR(dataset_PATH, dataset_name, numClasses):
	start_time = time.clock()
	print("Finding average BGR for dataset: " + dataset)
	class_count=0
	image_count=0
	BGR_sum = 0
	for class_path in glob.glob(dataset_PATH+"/*"): # for each class in the dataset
		class_path +="/"
		class_count+=1
		#if class_count>numClasses: # disregard clutter in Caltech-256
		#	break
		#print("CLASS PATH", class_path)
		for img_path in glob.glob(class_path+"*.jpg"): # for each image in the class
			image_count+=1
			img = cv2.imread(img_path)
			img_Avg_BGR = np.average(np.average(img,axis=0), axis=0) # average BGR for current image
			BGR_sum += img_Avg_BGR								# cumulative BGR sum
	print("Total # of images in the dataset:", image_count)
	print("Average BGR values found in " + str(time.clock()-start_time) + " seconds.")
	return 1.0*BGR_sum/image_count

########################################################

def readAndDisplayPickle(PATH):
	print("Reading image dictionary...")
	with(open(PATH, "rb")) as openfile:
		print(pickle.load(openfile))
