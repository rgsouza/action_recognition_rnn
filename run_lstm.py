# Last modification: 15 May 2019
# Author: Rayanne Souza

import numpy as np
import skvideo.io 
import glob 
import matplotlib.pyplot as plt
import cv2


from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, LSTM
from keras.models import Model
from keras import applications
from keras.preprocessing.image import img_to_array
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import classification_report


# fix random seed for reproducibility
np.random.seed(7)



# Function used to build the training and test set.
# You need to define the feature vector length per frame produced by the CNN.
def dataset_build(data_path, classes, set_):
  set_file_count=len(glob.glob(data_path+set_+'/**/*.avi',recursive=True))
  print(set_+" file count:",set_file_count)
  X_filenames=[]
  Y=np.zeros(set_file_count)
  sample_id=0
  for clss_name,clss_idx in zip(classes,range(len(classes))):
    clss_file_list=glob.glob(data_path+set_+'/'+clss_name+'/*.avi')
    X_filenames+=clss_file_list
    print("debug: sample_id={}, len(clss_file_list)={}".format(sample_id,clss_file_list))
    Y[sample_id:sample_id+len(clss_file_list)]=clss_idx
    sample_id+=len(clss_file_list)
  return X_filenames, Y



def get_cnn_model():
  
  # Defines the CNN model for extracting frames features #######
  vgg_model = applications.VGG16(weights='imagenet',
                                      include_top=False, input_shape=(224,224,3))
  x = vgg_model.output
  x = GlobalAveragePooling2D()(x)
  

  model = Model(inputs=vgg_model.input, outputs=x)
  model.trainable = False
  model.summary()
  
  return model

def extract_features(video_path, model, num_frames=40):
     
  # Video iterator for reading videos
  videogen = skvideo.io.vreader(video_path)
  vgg16_feature = []
  for i in range(num_frames):
      frame = next(videogen)
      
      """
      Resize the frame to a fixed dimension, 224x224 for instance
      """
      frame = cv2.resize(frame,(224,224))
      frame = img_to_array(frame)
      
      vgg16_feature.append(frame)
   
   
  
  # Preprocessing data and extract features using the CNN model###
  vgg16_feature = np.array(vgg16_feature, dtype="float32") / 255.0
  cnn_feature = model.predict(vgg16_feature)
      
  return np.squeeze(np.array(cnn_feature))


def show_result(acc, loss):
   
  
  # Shows the train history
  plt.figure(1)
  plt.plot(loss, label='train')
  plt.xlabel('epoch')
  plt.ylabel('Loss')
  plt.legend()
 
  plt.figure(2)
  plt.plot(acc, label='train')
  plt.xlabel('epoch')
  plt.ylabel('Accuracy')
  plt.savefig("Acc_loss.png")
  plt.legend()
  plt.show()



# Create a function for extracting features for all videos training and testing  
def extract_features_from_set(model,X_filenames,X):
  for filename,idx in zip(X_filenames,range(len(X_filenames))):
    X[idx,:,:] = extract_features(video_path=filename, model=model)
    if idx%50==0:
      print("Extracting features from videos. ID:",idx)
  return X



if __name__ == "__main__":
    
    data_path='data/Videos/'
    classes=['Basketball','Diving','GolfSwing','Skiing']
    t_len = 40 # number of frames per video
    feature_len = 512# feature vector length per frame
    class_n = len(classes) # number of classes

    X_filenames_train,Y_train=dataset_build(data_path, classes, set_='Testing')
    X_filenames_test,Y_test=dataset_build(data_path, classes, set_='Training')

    print("Train class distribution",np.unique(Y_train,return_counts=True))
    print("Test class distribution",np.unique(Y_test,return_counts=True))
    print("Classes:",classes)
    
    Y_train=np_utils.to_categorical(Y_train)
    Y_test=np_utils.to_categorical(Y_test)


    # Load the CNN model 
    model = get_cnn_model()

    try:
     X_train = np.load(open("xtrain.npy"))
     X_test = np.load(open("xtest.npy"))
    except:
  
     print("Extracting features")
  
     X_train = np.zeros((Y_train.shape[0],t_len,feature_len))
     X_test = np.zeros((Y_test.shape[0],t_len,feature_len))
     print("X train shape:{}, X test shape:{}".format(X_train.shape,X_test.shape))

     X_train = extract_features_from_set(model,X_filenames_train,X_train)
     print("Train features extracted")
     print(X_train.shape,X_test.shape)
     
     X_test = extract_features_from_set(model,X_filenames_test,X_test)
     print("Test features extracted")
  
     np.save(open("xtrain.npy","wb"),X_train)
     np.save(open("xtest.npy","wb"),X_test)
  
     # Verifying the train and test shapes 
    print("X_train.shape={}, Y_train.shape={}".format(X_train.shape,Y_train.shape))
    print("X_test.shape={}, Y_test.shape={}".format(X_test.shape,Y_test.shape))


    # Defines the LSTM Model 
    model_in = Input(shape=(X_train.shape[1], X_train.shape[2])) 
 
    model_out = LSTM(256, return_sequences=False)(model_in)
    model_out = Dense(512, activation='relu')(model_out)
    model_out = Dropout(0.5)(model_out)
    model_out = Dense(class_n, activation='softmax')(model_out)
  
    modelrnn = Model(model_in, model_out)
    np.asarray

    opt = Adam(lr=0.0001, decay=1e-6)
    modelrnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
 
    modelrnn.summary()


    # Train the LSTM Model  
    h = modelrnn.fit(X_train, Y_train,  epochs=15, batch_size=16, verbose=2)

    show_result(h.history['acc'], 
                h.history['loss'])

    # Reporting the accuracy, recall,precision and F1-score   
    predictions = modelrnn.predict(X_test)

    pred_bool = np.argmax(predictions, axis=1)
    Y_test_bool = np.argmax(Y_test, axis=1)

    print(classification_report(Y_test_bool, pred_bool))
