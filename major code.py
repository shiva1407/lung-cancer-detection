
import numpy as np
import pandas as pd
​
import matplotlib.pyplot as plt
import seaborn as sns 
​
import os
import glob
​
import SimpleITK as sitk
​
from PIL import Image
​
#from scipy.misc import imread
#from scipy.misc.pilutil import imread
​
import cv2
%matplotlib inline
from IPython.display import clear_output
pd.options.mode.chained_assignment = None
import sklearn





import pandas as pd
annotations = pd.read_csv('../input/lidcswsub/annotations.csv')
candidates = pd.read_csv('../input/lidcswsub/candidates.csv')





annotations.head()





candidates['class'].sum()





len(annotations)





candidates.info()





len(candidates[candidates['class'] == 1])





len(candidates[candidates['class'] == 0])





import multiprocessing
num_cores = multiprocessing.cpu_count()
print (num_cores)





class CTScan(object):
    def __init__(self, filename = None, coords = None):
        self.filename = filename
        self.coords = coords
        self.ds = None
        self.image = None
​
    def reset_coords(self, coords):
        self.coords = coords
​
    def read_mhd_image(self):
        path = glob.glob('../input/lidcswsub/seg-lungs-LUNA16/seg-lungs-LUNA16/'+ self.filename + '.mhd')
        self.ds = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.ds)
​
    def get_resolution(self):
        return self.ds.GetSpacing()
​
    def get_origin(self):
        return self.ds.GetOrigin()
​
    def get_ds(self):
        return self.ds
​
    def get_voxel_coords(self):
        origin = self.get_origin()
        resolution = self.get_resolution()
        voxel_coords = [np.absolute(self.coords[j]-origin[j])/resolution[j] \
            for j in range(len(self.coords))]
        return tuple(voxel_coords)
    
    def get_image(self):
        return self.image
    
    def get_subimage(self, width):
        self.read_mhd_image()
        x, y, z = self.get_voxel_coords()
        subImage = self.image[z, y-width/2:y+width/2, x-width/2:x+width/2]
        return subImage   
    
    def normalizePlanes(self, npzarray):
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray
    
    def save_image(self, filename, width):
        image = self.get_subimage(width)
        image = self.normalizePlanes(image)
        Image.fromarray(image*255).convert('L').save(filename)





positives = candidates[candidates['class']==1].index  
negatives = candidates[candidates['class']==0].index





import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
candidates = pd.read_csv('../input/lidcswsub/candidates.csv')
negatives = candidates[candidates['class']==0].index
scan = CTScan(np.asarray(candidates.iloc[negatives[600]])[0], \
              np.asarray(candidates.iloc[negatives[600]])[1:-1])
scan.read_mhd_image()
x, y, z = scan.get_voxel_coords()
image = scan.get_image()
dx, dy, dz = scan.get_resolution()
x0, y0, z0 = scan.get_origin()





filename = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100398138793540579077826395208'
coords = (70.19, -140.93, 877.68)#[877.68, -140.93, 70.19]
scan = CTScan(filename, coords)
scan.read_mhd_image()
x, y, z = scan.get_voxel_coords()
image = scan.get_image()
dx, dy, dz = scan.get_resolution()
x0, y0, z0 = scan.get_origin()





positives = candidates[candidates['class']==1].index 
positives





np.random.seed(42)
positives = candidates[candidates['class']==1].index 
negIndexes = np.random.choice(negatives, len(positives)*5, replace = False)





candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]





#from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
X = candidatesDf.iloc[:,:-1]
y = candidatesDf.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)





X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 42)





len(X_train)





X_train.to_pickle('traindata')
X_test.to_pickle('testdata')
X_val.to_pickle('valdata')





def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray





print('number of positive cases are ' + str(y_train.sum()))
print ('total set size is ' + str(len(y_train)))
print ('percentage of positive cases are ' + str(y_train.sum()*1.0/len(y_train)))





from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
​
candidates = pd.read_csv('../input/lidcswsub/candidates.csv')
positives = candidates[candidates['class']==1].index 
negatives = candidates[candidates['class']==0].index
negIndexes = np.random.choice(negatives, len(positives)*5, replace = False)
candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]
X = candidatesDf.iloc[:,:-1]
y = candidatesDf.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
tempDf = X_train[y_train == 1]
tempDf = tempDf.set_index(X_train[y_train == 1].index + 1000000)
​
X_train_new = X_train.append(tempDf)
tempDf = tempDf.set_index(X_train[y_train == 1].index + 2000000)
X_train_new = X_train_new.append(tempDf)
​
ytemp = y_train.reindex(X_train[y_train == 1].index + 1000000)
ytemp.loc[:] = 1
y_train_new = y_train.append(ytemp)
ytemp = y_train.reindex(X_train[y_train == 1].index + 2000000)
ytemp.loc[:] = 1
y_train_new = y_train_new.append(ytemp)
​
print (len(X_train_new), len(y_train_new))





X_train_new.index





#!pip install pillow
#from scipy.misc import imresize
#import skimage.transform.resize
from PIL import ImageEnhance
from PIL import Image
import cv2
​
class PreProcessing(object):
    def __init__(self, image = None):
        self.image = image
    
    def subtract_mean(self):
       # self.image = cv2.resize(self.image,(255,255))
        self.image = (self.image/255.0 - 0.25)*255
        return self.image
    
    def downsample_data(self):
        self.image = imresize(self.image, size = (40, 40), interp='bilinear', mode='L')
        return self.image
    
    def enhance_contrast(self):
        self.image = ImageEnhance.Contrast(self.image)
        return self.image





# import matplotlib.pyplot as plt
# import cv2
# dirName = '../input/lidcswsub/data/train/'
# plt.figure(figsize = (10,10))
# inp = cv2.imread(dirName + 'image_'+ str(30517) + '.jpg')
# plt.subplot(221)
# cv2.imshow('inp',inp)
# plt.show(inp)
​
# plt.grid(False)
​
# Pp = PreProcessing(inp)
​
# inp2 = Pp.subtract_mean()
# plt.subplot(222)
# plt.imshow(inp2)
# plt.grid(False)
​
# #inp4 = Pp.enhance_contrast()
# #plt.subplot(224)
# #plt.imshow(inp4)
# #plt.grid(False)
​
# inp3 = Pp.downsample_data()
# plt.subplot(223)
# plt.imshow(inp3)
# plt.grid(False)
​
# #inp4 = Pp.enhance_contrast()
# #plt.subplot(224)
# #plt.imshow(inp4)
# #plt.grid(False)
​





dirName = '../input/lidcswsub/data/train/'
dirName
dirName = '../input/lidcswsub/data/train/'
dirName





from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
​
candidates = pd.read_csv('../input/lidcswsub/candidates.csv')
positives = candidates[candidates['class']==1].index 
negatives = candidates[candidates['class']==0].index
negIndexes = np.random.choice(negatives, len(positives)*5, replace = False)
candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]
X = candidatesDf.iloc[:,:-1]
y = candidatesDf.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
tempDf = X_train[y_train == 1]
tempDf = tempDf.set_index(X_train[y_train == 1].index + 1000000)
​
X_train_new = X_train.append(tempDf)
tempDf = tempDf.set_index(X_train[y_train == 1].index + 2000000)
X_train_new = X_train_new.append(tempDf)
​
ytemp = y_train.reindex(X_train[y_train == 1].index + 1000000)
ytemp.loc[:] = 1
y_train_new = y_train.append(ytemp)
ytemp = y_train.reindex(X_train[y_train == 1].index + 2000000)
ytemp.loc[:] = 1
y_train_new = y_train_new.append(ytemp)
#coo
y_train_new.values.astype(int)





train_filenames =\
X_train_new.index.to_series().apply(lambda x:\
                                    '../input/lidcswsub/data/train/image_'+str(x)+'.jpg')
train_filenames.values.astype(str)
train_filenames =\
X_train_new.index.to_series().apply(lambda x:\
                                    '../input/lidcswsub/data/train/image_'+str(x)+'.jpg')
train_filenames.values.astype(str)





dataset_file = 'traindatalabels.txt'
​
# train_filenames =\
# X_train_new.index.to_series().apply(lambda x:\
           
filenames = train_filenames.values.astype(str) 
​
labels = y_train_new.values.astype(int) 
​
traindata = np.zeros(filenames.size,\
                     dtype=[('var1', 'S36'), ('var2', int)])
​
traindata['var1'] = filenames
​
traindata['var2'] = labels
​
np.savetxt(dataset_file, traindata, fmt="%10s %d")





# from tflearn.data_utils import build_hdf5_image_dataset
# dataset_file = '../input/lidcswsub/traindatalabels.txt'
# build_hdf5_image_dataset(dataset_file, image_shape=(50, 50), mode='file', output_path='/kaggle/working/traindataset.h5',categorical_labels=True, normalize=True)





# Load HDF5 dataset
import h5py
h5f = h5py.File('../input/lidcswsub/traindataset.h5', 'r')
X_train_images = h5f['X']
Y_train_labels = h5f['Y']
​
h5f2 = h5py.File('../input/lidcswsub/valdataset.h5', 'r')
X_val_images = h5f2['X']
Y_val_labels = h5f2['Y']





h5f3 = h5py.File('../input/lidcswsub/testdataset.h5', 'r')
X_test_images = h5f3['X']
Y_test_labels = h5f3['Y']
h5f3 = h5py.File('../input/lidcswsub/testdataset.h5', 'r')
X_test_images = h5f3['X']
Y_test_labels = h5f3['Y']





X_train_images = np.array(X_train_images).reshape(X_train_images.shape[0], 50, 50, 1)
Y_train_labels = np.array(Y_train_labels)
X_val_images = np.array(X_val_images).reshape(X_val_images.shape[0], 50, 50, 1)
Y_val_labels = np.array(Y_val_labels)
X_train_images = np.array(X_train_images).reshape(X_train_images.shape[0], 50, 50, 1)
Y_train_labels = np.array(Y_train_labels)
X_val_images = np.array(X_val_images).reshape(X_val_images.shape[0], 50, 50, 1)
Y_val_labels = np.array(Y_val_labels)





X_test_images = np.array(X_test_images).reshape(X_test_images.shape[0], 50, 50, 1)

Y_test_labels = np.array(Y_test_labels)
X_test_images = np.array(X_test_images).reshape(X_test_images.shape[0], 50, 50, 1)
​
Y_test_labels = np.array(Y_test_labels)





 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten





#create model
from tensorflow import keras
model = Sequential()
#add model layers
model.add(Conv2D(50, kernel_size=3, activation="relu", input_shape=(50,50,1)))
keras.layers.MaxPooling2D()
model.add(Conv2D(64, kernel_size=3, activation="relu"))
model.add(Conv2D(64, kernel_size=3,activation="relu"))
keras.layers.MaxPooling2D()
model.add(Flatten())
model.add(Dense(512, activation="relu"))
keras.layers.Dropout(0.5)
model.add(Dense(2, activation="softmax"))





from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
​
# Compatible with tensorflow backend
​
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed





#compile model using accuracy to measure model performance

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#compile model using accuracy to measure model performance
​
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])





#model.compile(optimizer ='adam' ,loss=[focal_loss(alpha=2.5, gamma=0.7)] , metrics=['accuracy'])
#model.compile(optimizer ='adam' ,loss=[focal_loss(alpha=2.5, gamma=0.7)] , metrics=['accuracy'])





#train the model
​
 model.fit(X_train_images, Y_train_labels, validation_data=(X_val_images, Y_val_labels), epochs=2,batch_size=96)





# Plot training & validation accuracy values
import matplotlib.pyplot as plt
history = model.fit(X_train_images, Y_train_labels, validation_data=(X_val_images, Y_val_labels), epochs=8,batch_size=96,verbose =1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
​
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()





Y_test_labels_classes = []
​
for sample in Y_test_labels:
    if sample[0] == 1.0:
        Y_test_labels_classes.append(0)
    else:
        Y_test_labels_classes.append(1)





# make class predictions with the model
predictions = model.predict_classes(X_test_images)





print("The testing accuracy is: ", sklearn.metrics.accuracy_score(Y_test_labels_classes, predictions))





#import numpy as np
#y_pred = np.argmax(predictions, axis=0)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    plt.imshow(cm, interpolation='nearest', cmap=cmap) 
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
​
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
​
    print(cm)
​
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
​
    plt.tight_layout()
    #plt.grid('off')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')





from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools
def get_metrics(Y_test_labels_classes, label_predictions):
  """
  Args:
  -----
  Y_test_labels, label_predictions
  Returns:
  --------
  precision, recall and specificity values and cm
  """
  cm = confusion_matrix(Y_test_labels[:,1], label_predictions[:,1])
​
  TN = cm[0][0]
  FP = cm[0][1]
  FN = cm[1][0]
  TP = cm[1][1]
​
  precision = TP*1.0/(TP+FP)
  recall = TP*1.0/(TP+FN)
  specificity = TN*1.0/(TN+FP)
​
  return precision, recall, specificity, cm





import numpy as np
print (Y_test_labels)
print(predictions)
#label_predictions = np.reshape(predictions,(-1,2))
#print(label_predictions)
#label_predictions = np.zeros_like(predictions)
#print(label_predictions)
#label_predictions[np.arange(len(predictions)), predictions] = 1
#print(label_predictions)
#np.concatenate((np.arange(len(predictions)),predictions))
labels_serial = np.arange(len(predictions))
label_predictions = np.vstack((labels_serial , predictions)).T
#combined = np.vstack((tp, fp)).T
print (label_predictions)





precision, recall, specificity, cm =\
   get_metrics(Y_test_labels, label_predictions)





cm = confusion_matrix(Y_test_labels[:,1], label_predictions[:,1])
​





plt.figure()
plot_confusion_matrix(cm, classes=['no-nodule', 'nodule'], \
    title='Confusion matrix')





def get_roc_curve(Y_test_labels, predictions):
  """
  Args:
  -------
  hdfs datasets: Y_test_labels and predictions
  
  Returns:
  --------
  fpr: false positive Rate
  tpr: true posiive Rate
  roc_auc: area under the curve value
  """
  fpr, tpr, thresholds = roc_curve(Y_test_labels[:,1], predictions, pos_label=1)
    #fpr, tpr, thresholds = roc_curve(Y_test_labels[:,1], predictions[:,1], pos_label=1)
    
    
  roc_auc = auc(fpr, tpr)
  return fpr, tpr, roc_auc
​





​
def plot_roc_curve(fpr, tpr, roc_auc):
  """
  Plots ROC curve
  Args:
  -----
  FPR, TPR and AUC
  """
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
    lw=lw, label='(AUC = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.axis('equal')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend(loc="lower right")
  plt.savefig('roc1.png', bbox_inches='tight')
​
​





​
  print(Y_test_labels)
  fpr, tpr, roc_auc = get_roc_curve(Y_test_labels, predictions)
  plot_roc_curve(fpr, tpr, roc_auc)
​
  precision, recall, specificity, cm =\
   get_metrics(Y_test_labels, label_predictions)
​
  print (precision, recall, specificity )
​
  # Plot non-normalized confusion matrix
  plt.figure()





def create_mosaic(image, nrows, ncols):
  """
  Tiles all the layers in nrows x ncols
  Args:
  ------
  image = 3d numpy array of M * N * number of filters dimensions
  nrows = integer representing number of images in a row
  ncol = integer representing number of images in a column
  returns formatted image
  """
​
  M = image.shape[1]
  N = image.shape[2]
​
  npad = ((0,0), (1,1), (1,1))
  image = np.pad(image, pad_width = npad, mode = 'constant',\
    constant_values = 0)
  M += 2
  N += 2
  image = image.reshape(nrows, ncols, M, N)
  image = np.transpose(image, (0,2,1,3))
  image = image.reshape(M*nrows, N*ncols)
  return image
​





def format_image(image, num_images):
  """
  Formats images
  """
  idxs = np.random.choice(image.shape[0], num_images)
  M = image.shape[1]
  N = image.shape[2]
  imagex = np.squeeze(image[idxs, :, :, :])
  print (imagex.shape)
  return imagex





​





def plot_predictions(images, filename):
  """
  Plots the predictions mosaic
  """
  imagex = format_image(images, 4)
  mosaic = create_mosaic(imagex, 2, 2)
  plt.figure(figsize = (12, 12))
  plt.imshow(mosaic, cmap = 'gray')
  plt.axis('off')
  plt.savefig(filename + '.png', bbox_inches='tight')
​





  TP_images = X_test_images[(Y_test_labels[:,1] == 1) & (label_predictions[:,1] == 1), :,:,:]
  FP_images = X_test_images[(Y_test_labels[:,1] == 0) & (label_predictions[:,1] == 1), :,:,:]
  TN_images = X_test_images[(Y_test_labels[:,1] == 0) & (label_predictions[:,1] == 0), :,:,:]
  FN_images = X_test_images[(Y_test_labels[:,1] == 1) & (label_predictions[:,1] == 0), :,:,:]
​





  plot_predictions(TP_images, 'preds_tps')
  plot_predictions(TN_images, 'preds_tns')
  plot_predictions(FN_images, 'preds_fns')
  plot_predictions(FP_images, 'preds_fps') 





print("Below shows the overall classification report: \n")
​
print(sklearn.metrics.classification_report(Y_test_labels_classes, predictions))





# Save model when training is complete to a file
model.save("nodule-classifier.tfl")
print("Network trained and saved as nodule-classifier.tfl!")





​
