#!/usr/bin/env python
# coding: utf-8

# Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

# ### Importing Skin Cancer Data
# #### To do: Take necessary actions to read the data

# ### Importing all the important libraries

# In[4]:


from collections import Counter
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[5]:


## If you are using the data by mounting the google drive, use the following :
## from google.colab import drive
## drive.mount('/content/gdrive')

##Ref:https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166


# This assignment uses a dataset of about 2357 images of skin cancer types. The dataset contains 9 sub-directories in each train and test subdirectories. The 9 sub-directories contains the images of 9 skin cancer types respectively.

# In[111]:


# Defining the path for train and test images
## Todo: Update the paths of the train and test dataset
data_dir_train = pathlib.Path("Skin cancer ISIC The International Skin Imaging Collaboration/Train")
data_dir_test = pathlib.Path("Skin cancer ISIC The International Skin Imaging Collaboration/Test")


# In[8]:


image_count_train = len(list(data_dir_train.glob('*/*.jpg')))
print(image_count_train)
image_count_test = len(list(data_dir_test.glob('*/*.jpg')))
print(image_count_test)


# ### Load using keras.preprocessing
# 
# Let's load these images off disk using the helpful image_dataset_from_directory utility.

# ### Create a dataset
# 
# Define some parameters for the loader:

# In[11]:


batch_size = 32
img_height = 180
img_width = 180
img_size = (img_height, img_width)


# Use 80% of the images for training, and 20% for validation.

# In[13]:


## Write your train dataset here
## Note use seed=123 while creating your dataset using tf.keras.preprocessing.image_dataset_from_directory
## Note, make sure your resize your images to the size img_height*img_width, while writting the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    image_size=img_size,
    batch_size=batch_size,
    subset="training",
    validation_split=0.2,
    seed=123
)


# In[14]:


## Write your validation dataset here
## Note use seed=123 while creating your dataset using tf.keras.preprocessing.image_dataset_from_directory
## Note, make sure your resize your images to the size img_height*img_width, while writting the dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    image_size=img_size,
    batch_size=batch_size,
    subset="validation",
    validation_split=0.2,
    seed=123
)


# In[15]:


# List out all the classes of skin cancer and store them in a list. 
# You can find the class names in the class_names attribute on these datasets. 
# These correspond to the directory names in alphabetical order.
class_names = train_ds.class_names
print(class_names)


# ### Visualize the data
# #### Todo, create a code to visualize one instance of all the nine classes present in the dataset

# In[18]:


import matplotlib.pyplot as plt

### your code goes here, you can use training or validation data to visualize
plt.figure(figsize=(12, 12))
for images, labels in train_ds.take(2):
    for i in range(len(class_names)):
        ax = plt.subplot(3, 3, i + 1)
        class_images = images[labels == i]
        if tf.reduce_sum(tf.cast(labels == i, tf.int32)) > 0:
            plt.imshow(class_images[0].numpy().astype("uint8"))
            plt.title(class_names[i])
            plt.axis("off")
plt.show()


# The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB). The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images.

# `Dataset.cache()` keeps the images in memory after they're loaded off disk during the first epoch.
# 
# `Dataset.prefetch()` overlaps data preprocessing and model execution while training.

# In[21]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ### Create the model
# #### Todo: Create a CNN model, which can accurately detect 9 classes present in the dataset. Use ```layers.experimental.preprocessing.Rescaling``` to normalize pixel values between (0,1). The RGB channel values are in the `[0, 255]` range. This is not ideal for a neural network. Here, it is good to standardize values to be in the `[0, 1]`

# In[23]:


### Your code goes here
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])


# ### Compile the model
# Choose an appropirate optimiser and loss function for model training 

# In[25]:


### Todo, choose an appropirate optimiser and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[26]:


# View the summary of all layers
model.summary()


# ### Train the model

# In[28]:


epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# ### Visualizing training results

# In[30]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# #### Todo: Write your findings after the model fit, see if there is an evidence of model overfit or underfit

# ### Write your findings here

# 1. #### Training vs Validation Accuracy:
# 
# 
# - The training accuracy (blue line) continues to increase steadily, reaching about 65%
# - The validation accuracy (orange line) plateaus around 50-55% and shows fluctuations
# - The growing gap between training and validation accuracy is a classic sign of overfitting
# 
# 
# 2. #### Training vs Validation Loss:
# 
# 
# - The training loss (blue line) continues to decrease steadily
# - The validation loss (orange line) initially decreases but then plateaus and slightly increases
# - The divergence between training and validation loss curves further confirms overfitting
# 
# 
# 3. #### Specific Indicators of Overfitting:
# 
# 
# - The model performs increasingly better on the training data while not improving on validation data
# - Around epoch 10, the curves begin to diverge significantly
# - The model is learning patterns specific to the training data that don't generalize well to new data

# In[35]:


# Todo, after you have analysed the model fit history for presence of underfit or overfit, choose an appropriate data augumentation strategy. 
# Your code goes herea

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])


# In[36]:


augmented_train_dataset = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))


# In[37]:


# Todo, visualize how your augmentation strategy works for one instance of training image.
# Your code goes here

plt.figure(figsize=(12, 12))
for images, labels in augmented_train_dataset.take(2):
    for i in range(len(class_names)):
        ax = plt.subplot(3, 3, i + 1)
        class_images = images[labels == i]
        if tf.reduce_sum(tf.cast(labels == i, tf.int32)) > 0:
            plt.imshow(class_images[0].numpy().astype("uint8"))
            plt.title(class_names[i])
            plt.axis("off")
plt.show()


# ### Todo:
# ### Create the model, compile and train the model
# 

# In[39]:


## You can use Dropout layer if there is an evidence of overfitting in your findings

## Your code goes rehe
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])



# ### Compiling the model

# In[41]:


## Your code goes here

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[42]:


model.summary()


# ### Training the model

# In[44]:


## Your code goes here, note: train your model for 20 epochs
epochs = 20

history = model.fit(
  augmented_train_dataset,
  validation_data=val_ds,
  epochs=epochs
)


# ### Visualizing the results

# In[46]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# #### Todo: Write your findings after the model fit, see if there is an evidence of model overfit or underfit. Do you think there is some improvement now as compared to the previous model run?

# ### Observations for the Current Model Run:
# 
# 1. #### Training vs Validation Accuracy:
# 
# 
# - Both training (blue) and validation (orange) accuracy curves follow similar trajectories
# - They increase gradually and stay close to each other, reaching around 52-55%
# - The curves show healthy fluctuations but maintain similar overall trends
# 
# 
# 2. #### Training vs Validation Loss:
# 
# 
# - Both loss curves decrease consistently
# - The training and validation losses remain relatively close to each other
# - Some natural fluctuations exist but no concerning divergence
# 
# 
# 3. #### Improvements over Previous Model:
# 
# 
# - Much better generalization: Unlike Model 1, there's no significant gap between training and validation metrics
# - More stable learning: The curves are smoother and show less erratic behavior
# - No clear signs of overfitting: The training accuracy isn't climbing far above validation accuracy
# - Both accuracy curves end around 53%, compared to Model 1 where training accuracy reached 65% while validation stayed at 55%
# 
# This model appears to be neither overfitting nor underfitting, showing a balanced learning pattern. However, there might be room for improvement since:
# 
# > The final accuracy (~53%) is relatively low, suggesting the model might benefit from Additional training time
# 
# > Overall, this is a significant improvement over Model 1 in terms of learning stability and generalization, even though the absolute performance might still need enhancement.

# #### **Todo:** Find the distribution of classes in the training dataset.
# #### **Context:** Many times real life datasets can have class imbalance, one class can have proportionately higher number of samples compared to the others. Class imbalance can have a detrimental effect on the final model quality. Hence as a sanity check it becomes important to check what is the distribution of classes in the data.

# In[113]:


## Your code goes here.
# Path to training dataset
train_path = "Skin cancer ISIC The International Skin Imaging Collaboration/Train"

# ImageDataGenerator instance
datagen = ImageDataGenerator()
train_data = datagen.flow_from_directory(train_path)

class_labels = train_data.classes

# Count occurrences of each class
class_distribution = Counter(class_labels)

# Map the counts to the class names
class_names_ref = {v: k for k, v in train_data.class_indices.items()}
class_distribution_named = {class_names_ref[k]: v for k, v in class_distribution.items()}

print("Class Distribution:", class_distribution_named)


# #### **Todo:** Write your findings here: 
# #### - Which class has the least number of samples?
# > #### seborrheic keratosis
# #### - Which classes dominate the data in terms proportionate number of samples?
# > #### pigmented benign keratosis
# 

# #### **Todo:** Rectify the class imbalance
# #### **Context:** You can use a python package known as `Augmentor` (https://augmentor.readthedocs.io/en/master/) to add more samples across all classes so that none of the classes have very few samples.

# In[53]:


pip install Augmentor


# To use `Augmentor`, the following general procedure is followed:
# 
# 1. Instantiate a `Pipeline` object pointing to a directory containing your initial image data set.<br>
# 2. Define a number of operations to perform on this data set using your `Pipeline` object.<br>
# 3. Execute these operations by calling the `Pipeline’s` `sample()` method.
# 

# In[55]:


path_to_training_dataset="Skin cancer ISIC The International Skin Imaging Collaboration/Train"
import Augmentor
for i in class_names:
    p = Augmentor.Pipeline(path_to_training_dataset + "/" + i)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(500) ## We are adding 500 samples per class to make sure that none of the classes are sparse.


# Augmentor has stored the augmented images in the output sub-directory of each of the sub-directories of skin cancer types.. Lets take a look at total count of augmented images.

# In[57]:


image_count_train = len(list(data_dir_train.glob('*/output/*.jpg')))
print(image_count_train)


# ### Lets see the distribution of augmented data after adding new images to the original training data.

# In[59]:


from glob import glob

path_list = [x for x in glob(os.path.join(data_dir_train, '*','output', '*.jpg'))]
path_list


# In[60]:


lesion_list_new = [os.path.basename(os.path.dirname(os.path.dirname(y))) for y in glob(os.path.join(data_dir_train, '*','output', '*.jpg'))]
lesion_list_new


# In[61]:


dataframe_dict_new = dict(zip(path_list, lesion_list_new))


# In[63]:


df2 = pd.DataFrame(list(dataframe_dict_new.items()),columns = ['Path','Label'])
# new_df = original_df.append(df2))


# In[65]:


df2['Label'].value_counts()


# So, now we have added 500 images to all the classes to maintain some class balance. We can add more images as we want to improve training process.

# #### **Todo**: Train the model on the data created using Augmentor

# In[69]:


batch_size = 32
img_height = 180
img_width = 180
img_size = (img_height, img_width)


# #### **Todo:** Create a training dataset

# In[79]:


data_dir_train="Skin cancer ISIC The International Skin Imaging Collaboration/Train"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=123,
  validation_split = 0.2,
  subset = "training", ## Todo choose the correct parameter value, so that only training data is refered to,,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# #### **Todo:** Create a validation dataset

# In[81]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=123,
  validation_split = 0.2,
  subset = "validation", ## Todo choose the correct parameter value, so that only validation data is refered to,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# #### **Todo:** Create your model (make sure to include normalization)

# In[83]:


## your code goes here
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])


# #### **Todo:** Compile your model (Choose optimizer and loss function appropriately)

# In[85]:


### Your code goes here
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# #### **Todo:**  Train your model

# In[89]:


epochs = 30
## Your code goes here, use 50 epochs.
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# #### **Todo:**  Visualize the model results

# In[91]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# #### **Todo:**  Analyze your results here. Did you get rid of underfitting/overfitting? Did class rebalance help?
# 
# 

# ### Observations for the Current Model Run:
# 
# 1. #### Training vs Validation Accuracy:
# 
# 
# - Both curves show strong upward trajectories, reaching much higher accuracies (~80-85%)
# - The curves track each other closely until around epoch 20
# - After epoch 20, there's a slight divergence with training accuracy moving slightly higher than validation accuracy
# - There's one notable validation accuracy drop around epoch 25, but it recovers quickly
# 
# 
# 2. #### Training vs Validation Loss:
# 
# 
# - Both loss curves show consistent decrease initially
# - After epoch 15, training loss (blue) continues to decrease while validation loss (orange) plateaus and shows more fluctuation
# - The growing gap between training and validation loss in later epochs suggests the beginning of overfitting
# 
# 
# 3. #### Improvements over Previous Models:
# 
# 
# - Much higher accuracy achieved (80-85% vs ~53% in Model 2)
# - Better learning stability in the first 20 epochs
# - Stronger overall performance and learning capacity
# - More training epochs allowed (30 vs ~18 in previous models)
# 
# 
# 4. #### Areas of Concern:
# 
# 
# - Mild overfitting appears to begin after epoch 20, as evidenced by:
# 
# > - Diverging loss curves
# > - Training accuracy continuing to climb while validation accuracy plateaus
# > - Increased validation accuracy volatility
# 
# This model represents the best performance of the three, with significantly higher accuracy and better learning characteristics, despite the mild overfitting in later epochs. The improvements in model architecture or hyperparameters have clearly yielded better results.

# In[173]:


test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_test,
    image_size=img_size,
    batch_size=batch_size
)


# In[175]:


# Evaluate the model's performance
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# In[127]:


# Predict labels for the test dataset
predictions = model.predict(test_dataset)


# In[123]:


classes = test_dataset.class_names
classes


# In[125]:


plt.figure(figsize=(12, 12))
for images, labels in test_dataset.take(1):
    for i in range(9):  # Display first 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class = classes[np.argmax(predictions[i])]
        true_class = classes[labels[i]]
        plt.title(f"True: {true_class}\nPred: {predicted_class}")
        plt.axis("off")
plt.show()


# In[215]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Test Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)

def prepare_test_data(test_path):
    test_dataset = test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',  # Generate integer-encoded labels
        shuffle=False  # Keep the order for consistent evaluation
    )
    return test_dataset

def evaluate_model(model, test_dataset):
    # Get the total number of samples
    num_samples = test_dataset.samples
    
    # Calculate steps - convert to integer
    steps = int(np.ceil(num_samples/test_dataset.batch_size))
    
    # Reset the generator to the start
    test_dataset.reset()
    
    # Get predictions for all samples
    predictions = model.predict(test_dataset, steps=steps)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_dataset.classes
    
    # Get class names
    class_names = list(test_dataset.class_indices.keys())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Reset the generator again for evaluation
    test_dataset.reset()
    
    # Calculate and print accuracy
    results = model.evaluate(test_dataset, steps=steps)
    print(f"\nTest Accuracy: {results[1]*100:.2f}%")
    print(f"Test Loss: {results[0]:.4f}")



def predict_single_image(model, image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=img_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence

test_path = "Skin cancer ISIC The International Skin Imaging Collaboration/Test"

# Prepare test data
test_dataset = prepare_test_data(test_path)

# Evaluate model on test data
evaluate_model(model, test_dataset)


# In[185]:


def get_file_names(directory_path):
    try:
        # List all files in the directory
        file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return file_names
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' does not exist.")
        return []


# In[211]:


# Predicting melanoma test image

test_melanoma_file_list = get_file_names("Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma")

success_count = 0
for file in test_melanoma_file_list:
    sample_image_path = f"Skin cancer ISIC The International Skin Imaging Collaboration/Test/melanoma/{file}"
    class_names = list(test_dataset.class_indices.keys())
    predicted_class, confidence = predict_single_image(model, sample_image_path)
    print(f"\nPredicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    if (class_names[predicted_class] == "melanoma" and confidence == 1.0):
        success_count += 1
print(f"Successfully identified melanoma images: {(success_count / len(test_melanoma_file_list))*100:.2f}%")


# In[ ]:




