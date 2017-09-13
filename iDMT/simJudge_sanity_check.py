# Python script to conduct sanity check on similarityJudge

from misc import *
from similarityJudge import *

Caltech256_avg = np.array([ 128.78144029, 136.06747341, 140.76367044]) 
ILSVRC2012_avg = np.array([93.5940,104.7624,129.1863])  

# Source dataset
source_dataset_PATH = "/Users/berkkocabagli/Desktop/UCSB_Research_Summer2017/Datasets/256_ObjectCategories" 
source_dataset_name = "Caltech-256"
classes_PATH = "./" + source_dataset_name + "_classes" # PATH FOR THE LIST OF CLASSES OF SOURCE DATA
# LOAD CLASSES IN SOURCE DATASET
with(open(classes_PATH)) as openfile:
	classes=pickle.load(openfile)
numClasses = len(classes)

source_dataset_avg_BGR = Caltech256_avg
N = numClasses
FE_model_name = "AlexNet"
FE_model= "/Users/berkkocabagli/Desktop/UCSB_Research_Summer2017/Models/AlexNet/deploy.prototxt" 
FE_weights= "/Users/berkkocabagli/Desktop/UCSB_Research_Summer2017/Models/AlexNet/bvlc_alexnet.caffemodel"
similarity_FE_layer= "fc6"
similarity_image_fv_dictionaries_PATH = "./FeatureVectors/" + similarity_FE_layer
network_input_width = 227
network_input_height = 227

my_dict = dict()
most_similar_N_classes_dict = dict()

top_3_count = 0
top_5_count = 0

for class_no in range (0):
	print("Check for class " + str(class_no))
	#rand_img_path = random.choice([img_path for img_path in glob.glob(source_dataset_PATH + "/" + classes[class_no] + "/*")])
	rand_img_path = [img_path for img_path in glob.glob(source_dataset_PATH + "/" + classes[class_no] + "/*")][0]
	print(rand_img_path)
	#displayImage(rand_img_path, cv2.imread(rand_img_path),0.5)
	most_similar_N_classes = findMostSimilarClasses(rand_img_path, source_dataset_name, source_dataset_PATH, classes, similarity_image_fv_dictionaries_PATH, source_dataset_avg_BGR, N, FE_model_name, FE_model, FE_weights, similarity_FE_layer, network_input_width,network_input_height) 
	most_similar_N_classes_dict[classes[class_no]]=most_similar_N_classes
	#print(most_similar_N_classes)
	for i in range(N):
		if(most_similar_N_classes[i]==class_no): 
			my_dict[classes[class_no]]=i
			if(i<3): top_3_count += 1
		 	if(i<5): top_5_count +=1 

print(my_dict)
print("Top-3 count: " + str(top_3_count))
print("Top-5 count: " + str(top_5_count))

# Save the most_similar_N_classes dictionary
print("Saving the most_similar_N dictionary...")
if(not os.path.exists("./"+ similarity_FE_layer + "_" + FE_model_name + "_on_" + source_dataset_name + "_topN_dict")):
	with open("./"+ similarity_FE_layer + "_" + FE_model_name + "_on_" + source_dataset_name + "_topN_dict", 'wb') as outfile:
		pickle.dump(most_similar_N_classes_dict, outfile)

if(not os.path.exists("./"+ similarity_FE_layer + "_" + FE_model_name + "_on_" + source_dataset_name + "_topN_performance_dict")):
	with open("./"+ similarity_FE_layer + "_" + FE_model_name + "_on_" + source_dataset_name + "_topN_performance_dict", 'wb') as outfile:
		pickle.dump(my_dict, outfile)

######################################################
######### THIS PART IS AFTER RUNNING THE ABOVE CODE AND GENERATING PERFORMANCE DICTIONARIES
######### WILL NEED TO BE MODIFIED TO COMPARE DIFFERENT MODELS/LAYERS
######################################################

with open("./simJudgePerformance_random/fc6_AlexNet_on_Caltech-256_topN_performance_dict", 'rb') as openfile:
	perf_dict_fc6= pickle.load(openfile)
with open("./simJudgePerformance_random/fc8_AlexNet_on_Caltech-256_topN_performance_dict", 'rb') as openfile:
	perf_dict_fc8= pickle.load(openfile)

top_3_count_fc6=0
top_3_count_fc8=0
top_5_count_fc6=0
top_5_count_fc8=0

for c in classes:
	if(perf_dict_fc6[c]<3): 
		top_3_count_fc6 +=1
	if(perf_dict_fc6[c]<5):
		top_5_count_fc6 +=1
	if(perf_dict_fc8[c]<3): 
		top_3_count_fc8 +=1
	if(perf_dict_fc8[c]<5):
		top_5_count_fc8 +=1
	print(c,perf_dict_fc6[c],perf_dict_fc8[c])
print("FC6", top_3_count_fc6, top_5_count_fc6)
print("FC8", top_3_count_fc8, top_5_count_fc8)

