
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import datetime
import itertools
import random
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
  except RuntimeError as e:
    print(e)
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix,accuracy_score
from plot_keras_history import show_history, plot_history
import random
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.metrics import precision_score,recall_score, classification_report
import cv2
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing




# Walk through the directory and list number of files
for dirpath, dirnames, filenames in os.walk('./Data'):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")



# Another way to find out how many images are in a class
num_converse_images_train = len(os.listdir("./Data/train/converse"))
num_converse_images_test = len(os.listdir("./Data//test/converse"))
print(f"Total number of test images of converse is {num_converse_images_test} \nTotal number of train images of converse is {num_converse_images_train}")




# Define a function to view a random image
def viewRandomImage(targetDir, targetClass,savename):
    # Setup target directory
    targetFolder = targetDir + targetClass
    
    # Get a random image path
    randomImage = random.sample(os.listdir(targetFolder), 1)

    # Read in the image and plot it using matplotlib
    img = mpimg.imread(targetFolder + '/' + randomImage[0])
    plt.imshow(img)
    plt.title(targetClass)
    plt.axis(False)
    plt.savefig('./imgs/'+savename+'.png')
    
    # Show the shape of the image
    print(f"Image Shape: {img.shape}")
    
    return img



# View a random image from the training dataset 
img = viewRandomImage(targetDir='./Data/train/',targetClass='adidas',savename='EX-Adidas-0') 
img = viewRandomImage(targetDir='./Data/train/',targetClass='nike',savename='EX-Nike-0') 
img = viewRandomImage(targetDir='./Data/train/',targetClass='converse',savename='EX-Converse-0') 
img = viewRandomImage(targetDir='./Data/train/',targetClass='adidas',savename='EX-Adidas-1') 
img = viewRandomImage(targetDir='./Data/train/',targetClass='nike',savename='EX-Nike-1') 
img = viewRandomImage(targetDir='./Data/train/',targetClass='converse',savename='EX-Converse-1') 
                     





# Setup the training and the test directory paths
trainDir = "./Data/train"
testDir = "./Data/test"

# Define the hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

#----------------------------------Xception------------------------------------------

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input,zoom_range=0.1,brightness_range=[0.1,0.2],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)


test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)




train_images = train_datagen.flow_from_directory(directory = trainDir, target_size = IMG_SIZE, class_mode = 'categorical',batch_size = BATCH_SIZE,subset='training')


val_images = train_datagen.flow_from_directory(directory = trainDir, target_size = IMG_SIZE, class_mode = 'categorical',batch_size = BATCH_SIZE,subset='validation')

test_images=train_datagen.flow_from_directory(directory=testDir,target_size = IMG_SIZE,class_mode = 'categorical', batch_size = BATCH_SIZE,shuffle = False) 
                                                               
                                                            
inputs = tf.keras.layers.Input((224,224,3))
base_model=tf.keras.applications.xception.Xception(include_top=False, weights="imagenet",input_shape=(224,224,3), pooling='avg') 
x=base_model(inputs)
x=layers.Dense(256, activation='relu')(x)
x=layers.Dense(64, activation='relu')(x)
output=layers.Dense(3, activation='softmax')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
optimizer=tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, epochs=50,validation_data=val_images)

model.save("./model/Xception.h5")

show_history(history)
plot_history(history, path="./imgs/Training_history Xception.png",title="Xception Training history")
plt.close()

test_images.reset()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]

#----------------Classification_report-------------------------------

print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(real,predictions)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(real,predictions,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(real,predictions,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(real,predictions,average='weighted')))
print('---------------------------------------------------------')
print(classification_report(real,predictions))
conf_matrix = confusion_matrix(real, predictions)
sns.heatmap(conf_matrix,xticklabels = ["0","1","2"], yticklabels =["0","1","2"],annot=True,fmt='g')
plt.title('Xception Confusion Matrix')
plt.savefig("./imgs/Confusion Matrix Xception.png")
plt.show()






#----------------------------------EfficientNetV2S------------------------------------------

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,zoom_range=0.1,brightness_range=[0.1,0.2],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)


test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input)




train_images = train_datagen.flow_from_directory(directory = trainDir, target_size = IMG_SIZE, class_mode = 'categorical',batch_size = BATCH_SIZE,subset='training')


val_images = train_datagen.flow_from_directory(directory = trainDir, target_size = IMG_SIZE, class_mode = 'categorical',batch_size = BATCH_SIZE,subset='validation')

test_images=train_datagen.flow_from_directory(directory=testDir,target_size = IMG_SIZE,class_mode = 'categorical', batch_size = BATCH_SIZE,shuffle = False) 
                                                               
                                                            
inputs = tf.keras.layers.Input((224,224,3))
base_model=tf.keras.applications.EfficientNetV2S(include_top=False, weights="imagenet",input_shape=(224,224,3),pooling='avg') 
x=base_model(inputs)
x=layers.Dense(256, activation='relu')(x)
x=layers.Dense(64, activation='relu')(x)
output=layers.Dense(3, activation='softmax')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
optimizer=tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, epochs=50,validation_data=val_images)

model.save("./model/EfficientNetV2S.h5")

show_history(history)
plot_history(history, path="./imgs/Training_history EfficientNetV2S.png",title="EfficientNetV2S Training history")
plt.close()

test_images.reset()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]

#----------------Classification_report-------------------------------

print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(real,predictions)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(real,predictions,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(real,predictions,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(real,predictions,average='weighted')))
print('---------------------------------------------------------')
print(classification_report(real,predictions))
conf_matrix = confusion_matrix(real, predictions)
sns.heatmap(conf_matrix,xticklabels = ["0","1","2"], yticklabels =["0","1","2"],annot=True,fmt='g')
plt.title('EfficientNetV2S Confusion Matrix')
plt.savefig("./imgs/Confusion Matrix EfficientNetV2S.png")
plt.show()





#----------------------------------InceptionResNetV2------------------------------------------

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input,zoom_range=0.1,brightness_range=[0.1,0.2],
                                   width_shift_range=0.1,height_shift_range=0.1,validation_split=0.1)


test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)




train_images = train_datagen.flow_from_directory(directory = trainDir, target_size = IMG_SIZE, class_mode = 'categorical',batch_size = BATCH_SIZE,subset='training')


val_images = train_datagen.flow_from_directory(directory = trainDir, target_size = IMG_SIZE, class_mode = 'categorical',batch_size = BATCH_SIZE,subset='validation')

test_images=train_datagen.flow_from_directory(directory=testDir,target_size = IMG_SIZE,class_mode = 'categorical', batch_size = BATCH_SIZE,shuffle = False) 
                                                               
                                                            
inputs = tf.keras.layers.Input((224,224,3))
base_model=tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet",input_shape=(224,224,3),pooling='avg') 

x=base_model(inputs)
x=layers.Dense(256, activation='relu')(x)
x=layers.Dense(64, activation='relu')(x)
output=layers.Dense(3, activation='softmax')(x)
model=tf.keras.models.Model(inputs=inputs, outputs=output)
optimizer=tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, epochs=50,validation_data=val_images)

model.save("./model/InceptionResNetV2.h5")

show_history(history)
plot_history(history, path="./imgs/Training_history InceptionResNetV2.png",title="InceptionResNetV2 Training history")
plt.close()

test_images.reset()
preds=model.predict(test_images)
preds = np.argmax(preds,axis=1)
labels = (test_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in preds]
real=[labels[k] for k in test_images.labels]

#----------------Classification_report-------------------------------

print('---------------------------------------------------------')
print('Accuracy:'+str(accuracy_score(real,predictions)))
print('---------------------------------------------------------')
print('F1-Score:'+str(f1_score(real,predictions,average='weighted')))
print('---------------------------------------------------------')
print('Recall:'+str(recall_score(real,predictions,average='weighted')))
print('---------------------------------------------------------')
print('Precision:'+str(precision_score(real,predictions,average='weighted')))
print('---------------------------------------------------------')
print(classification_report(real,predictions))
conf_matrix = confusion_matrix(real, predictions)
sns.heatmap(conf_matrix,xticklabels = ["0","1","2"], yticklabels =["0","1","2"],annot=True,fmt='g')
plt.title('InceptionResNetV2 Confusion Matrix')
plt.savefig("./imgs/Confusion Matrix InceptionResNetV2.png")
plt.show()



# Define a function to view a random image
def viewRandomImagePredict(model_path,targetDir, targetClass,savename):
    # Setup target directory
    targetFolder = targetDir + targetClass
    model = keras.models.load_model(model_path)
    
    # Get a random image path
    randomImage = random.sample(os.listdir(targetFolder), 1)

    # Read in the image and plot it using matplotlib
    img=cv2.imread(targetFolder + '/' + randomImage[0])
    img=cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img_model=img.reshape(1,224,224,3)
    preds=model.predict(img_model,batch_size=1)
    preds = np.argmax(preds,axis=1)
    
    labels ={0:'adidas',1:'converse',2:'nike'}
    
    predictions = [labels[k] for k in preds]
    predictions = ''.join(predictions)
    
    plt.imshow(img)
    plt.title('Real:'+targetClass+'Predict:'+str(predictions))
    plt.axis(False)
    plt.savefig('./imgs/'+savename+'.png')
    
    return img


for i in range(3):
# View a random image from the training dataset 
    img = viewRandomImagePredict(model_path="./model/Xception.h5",targetDir='./Data/train/',targetClass='adidas',savename='XMEX-Adidas'+str(i)) 

for i in range(3):
# View a random image from the training dataset 
    img = viewRandomImagePredict(model_path="./model/InceptionResNetV2.h5",targetDir='./Data/train/',targetClass='adidas',savename='IMEX-Adidas'+str(i)) 

for i in range(3):
# View a random image from the training dataset 
    img = viewRandomImagePredict(model_path="./model/EfficientNetV2S.h5",targetDir='./Data/train/',targetClass='adidas',savename='NMEX-Adidas'+str(i)) 



for i in range(3):
# View a random image from the training dataset 
    img = viewRandomImagePredict(model_path="./model/Xception.h5",targetDir='./Data/train/',targetClass='converse',savename='XMEX-Converse'+str(i)) 

for i in range(3):
# View a random image from the training dataset 
    img = viewRandomImagePredict(model_path="./model/InceptionResNetV2.h5",targetDir='./Data/train/',targetClass='converse',savename='IMEX-Converse'+str(i)) 

for i in range(3):
# View a random image from the training dataset 
    img = viewRandomImagePredict(model_path="./model/EfficientNetV2S.h5",targetDir='./Data/train/',targetClass='converse',savename='NMEX-Converse'+str(i)) 




for i in range(3):
# View a random image from the training dataset 
    img = viewRandomImagePredict(model_path="./model/Xception.h5",targetDir='./Data/train/',targetClass='nike',savename='XMEX-Nike'+str(i)) 

for i in range(3):
# View a random image from the training dataset 
    img = viewRandomImagePredict(model_path="./model/InceptionResNetV2.h5",targetDir='./Data/train/',targetClass='nike',savename='IMEX-Nike'+str(i)) 

for i in range(3):
# View a random image from the training dataset 
    img = viewRandomImagePredict(model_path="./model/EfficientNetV2S.h5",targetDir='./Data/train/',targetClass='nike',savename='NMEX-Nike'+str(i)) 


















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

