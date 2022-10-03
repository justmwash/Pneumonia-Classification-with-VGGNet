import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# check the folders
import os
print("Working Directory Content(s): ", os.listdir("C:/Machine Learning DataSet/Chest_xray/Datasets/"))

# create  images folders
train_dir = "C:/Machine Learning DataSet/Chest_xray/Datasets/train/"
val_dir = "C:/Machine Learning DataSet/Chest_xray/Datasets/val/"
test_dir = "C:/Machine Learning DataSet/Chest_xray/Datasets/test/"

# check what is inside one of the folders
train_dir = "C:/Machine Learning DataSet/Chest_xray/Datasets/train/"
print("Train Dir has ", os.listdir(train_dir))

# check the size for each sub dir
#we are combining both normal and pnumonia images folders
train_len = len(os.listdir(train_dir+"NORMAL/")) + len(os.listdir(train_dir+"PNEUMONIA/"))
train_normal = len(os.listdir(train_dir+"NORMAL/"))
train_pneu = len(os.listdir(train_dir+'PNEUMONIA/'))
print(f"There are {train_len}  Images in Traing where {train_pneu} are Pneumonia and {train_normal} are normal")


test_len = len(os.listdir(test_dir+"NORMAL/")) + len(os.listdir(test_dir+"PNEUMONIA/"))
test_normal = len(os.listdir(test_dir+"NORMAL/"))
test_pneu = len(os.listdir(test_dir+'PNEUMONIA/'))
print(f"There are {test_len}  Images in testing where {test_pneu} are Pneumonia and {test_normal} are normal")


val_len = len(os.listdir(val_dir+"NORMAL/")) + len(os.listdir(val_dir+"PNEUMONIA/"))
val_normal = len(os.listdir(val_dir+"NORMAL/"))
val_pneu = len(os.listdir(val_dir+'PNEUMONIA/'))
print(f"There are {val_len}  Images in Validation where {val_pneu} are Pneumonia and {val_normal} are normal")




#training the vgg16 model
IMAGE_SIZE = [224, 224]

train_path = 'Datasets/train'
valid_path = 'Datasets/test'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
    folders = glob('Datasets/train/*')
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
# view the structure of the model
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)




# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')




test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=1,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
import tensorflow as tf
from keras.models import load_model
model.save('chest_xray.h5')
print("Model is ready...") 

from tensorflow.keras.applications.vgg16 import VGG16
vgg16_model = VGG16(weights='imagenet')
vgg16_model.summary()

from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
# test the model
print("Start Evaluating...")

print("\t\t **********VGG16 EVALUATION *****************\n\n")
preds = vgg16_model.predict(Xval, batch_size=batch)
# get the max predicted output
preds = np.argmax(preds, axis=1)

print(f"Accuracy is   {accuracy_score(yval.argmax(axis=1) ,preds)*100}%\n\n")
# classifciation reports
print(classification_report(yval.argmax(axis=1) ,preds))
print("\n\n")

c_matrix = confusion_matrix(yval.argmax(axis=1) ,preds)
# plot confusion matrix for better view
plt.figure(figsize=(4,4))
sns.heatmap(c_matrix , annot= True ,fmt="" ,  annot_kws={"size": 10})
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.title(f"Confusion Matrix Plot for VGG16 classifier")
plt.savefig(f"mobinet.png")
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
model=load_model('chest_xray.h5')
img=tf.keras.utils.load_img('C:\\Machine Learning DataSet\\Cheast_xray\\Datasets\\val\\NORMAL\\NORMAL2-IM-1431-0001.jpeg',target_size=(224,224))
x=image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data)
result=int(classes[0][0])
if result==0:
    print("Person is Affected By PNEUMONIA")
else:
    print("Result is Normal")
    import streamlit as st