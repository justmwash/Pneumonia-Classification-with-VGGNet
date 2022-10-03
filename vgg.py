from tensorflow.keras.applications.vgg16 import VGG16
vgg16_model = VGG16(weights='imagenet')
vgg16_model.summary()
#img_width , img_height , channel = X_train.shape[1:]
#img_width , img_height , channel
# create Inputs with the 150 * 150 * 3 dimensions to adjust from vgg16
input_vgg_layer = input #(shape=(img_width,img_height,channel))
# set our model with the new inputs dimension
vgg_base=VGG16(weights='imagenet',input_tensor=input_vgg_layer,include_top=False)

# get the last layer of the model
vgg_head=vgg_base.output

vgg_head = Conv2D(8, 3, 3, activation='elu')(vgg_head)

#flatten the last layer and add other layers
#Here our architecture needs to have 2 as the number of classes
vgg_head=Flatten()(vgg_head) 

vgg_head = Dense(128, activation="relu")(vgg_head)
vgg_head = Dense(64, activation="relu")(vgg_head)
vgg_head = Dropout(0.5)(vgg_head)
# add a dense layer for the output with the 2 classes
vgg_head = Dense(2, activation="softmax")(vgg_head)
# create a model that maps the inputs and output layers together
vgg16_model = Model(inputs=vgg_base.input, outputs=vgg_head)


# check the model summary
vgg16_model.summary()

tf.keras.utils.plot_model(
    vgg16_model, to_file='VGG16.png', show_shapes=True,
    show_layer_names=True,
)

# time
import time
start_time = time.time()
# freeze all layers from the base to avoid update during traing
for layer in vgg_base.layers:
    layer.trainable = False

# compile 
LR = 1e-5
epochs = 40
batch =32

print("***** Started Compiling VGG16 model and optimization...................")
vgg16_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=LR, decay=LR / epochs),metrics=["accuracy"])

# train 
print("Begin training ......")
history = vgg16_model.fit(aug.flow(X_train, y_train, batch_size=batch),
                    steps_per_epoch=len(Xval) // batch,
                    validation_data=(Xval, yval),
                    validation_steps=len(Xval) // batch,
                    epochs=epochs)

print(f"The VGG16 Model Took   {(time.time()-start_time)/60}  to run to completion")

# evaluate
loss, acc = vgg16_model.evaluate(Xval, yval, verbose=0)
print(f"********VGG16 Model Score**********\n")
print(f'Accuracy  {(acc * 100.0)}')
print(f'Loss  {(loss * 100.0)}')

create_review_plots(history , "VGG16")

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

print("Saving the model for future use...")
vgg16_model.save("vgg16_model.model" , save_format='h5')
print(f"VGG16  model had   {len(vgg16_model.layers)} layers")

print("Our model is ready!")