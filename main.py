import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from tensorflow.keras.models import Sequential, load_model # type :ignore
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout # type: ignore
from tensoreflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from kerastuner.tuners import RandomSearch # type: ignore
from kerastuner.engine.hyperparameters import HyperParameters # type: ignore

#Data Preparation
X_train = []
Y_train = []
image_size = 150
labels = [
    'ALABAMA', 'ALASKA', 'AMERICAN SAMOA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'CNMI', 
    'COLORADO', 'CONNECTICUT', 'DELAWARE','FLORIDA', 'GEORGIA', 'GUAM', 'HAWAI', 'IDAHO',
    'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS','KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND',
    'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA',
    'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 
    'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'OKLAHOMA','OREGON', 'PENNSYLVANIA', 
    'PUERTO RICO', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA', 'TENNESSEE', 
    'TEXAS', 'US VIRGIN ISLANDS','UTAH', 'VERMONT','VIRGINIA', 'WASHINGTON',
    'WASHINGTON DC', 'WEST VIRGINIA', 'WISCONSIN', 'WYOMING'
]

#Loading training images with progress bar
image_counts = {}
for i in labels:
    folderPath = os.path.join('./data',i)
    image_counts[i] = len(os.listdir(folderPath))
    for j in tqdm(os.listdir(folderPath), desc=f'Loading {i} training images'):
    img = cv2.imread(os.path.join(folderPath,j))
    if img is None:
        print(f"Error: Unable to load image from {(os.path.join(folderPath,j))}")
        continue #Skip the current iteration
    img = cv2.resixe(img,(image_size,image_size))
    X_train.append(img)
    Y_train.append(i)

#Print the number of images in each subset before augmentation
print("Number of images is each subset before augmentation:")
for label, count in image_counts.items():
    print(f"{label}:{count}")

# Perform data augmentation to balance the dataset
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Determine the maximum number of images in any subset
max_images = max(image_counts.values())

# Augment images to match the maximum number in each subset
augmented_X_train = []
augmented_Y_train = []

for label in labels:
    current_count = image_counts[label]
    additional_images_needed = max_images - current_count
    if additional_images_needed > 0:
        label_indices = [index for index, y in enumerate(Y_train) if y==label]
        images_to_augment = [X_train[index] for index in label_indices]
        augmented_count = 0 
        for image in images_to_augment:
            image = image.reshape((1,) + image.shape) #in the paranthesis its 1,null 
            #cause we are not making any changes to the label
            for batch in datagen.flow(image, batch_size=1):
                augmented_X_train.append(batch[0])
                augmented_Y_train.append(label)
                augmented_count += 1
                if augmented_count >= additional_images_needed:
                    break
                if augmented_count >= additional_images_needed:
                    break

# Add augmented image to the training set
X_train.extend(augmented_X_train)
Y_train.extend(augmented_Y_train)

# Convert to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Shuffle the dataset
X_train, Y_train = shuffle(X_train, Y_train, random_state=28) #random state could be any number

# Print the number of images in each subset after augmentation
print("\nNumber of images in each subset after augmentation:")
for label in labels:
    label_count = sum ([1 for y in Y_train if y ==  label])
    print(f"{label}: {label_count}")

#Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=28) 

Y_train_new = [labels.index(i) for i in Y_train]
Y_train = tf.keras.utils.to_categorically(Y_train_new, num_classes=15)

# Define a function to build the model for Keras Tuner
def build_model(hp):
model = Sequential()
model.add(Conv2D(hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
(3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(hp. Int('conv_2_filter', min_value=32, max_value=128, step=16),
(3, 3), activation='relu' ) )
model.add(MaxPooling2D(2, 2))
model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.05)))
model.add(Conv2D(hp.Int('conv_3_filter', min_value=64, max_value=256, step=32),
(3, 3), activation='relu' ))
model.add(Conv2D(hp.Int('conv_4_filter', min_value=64, max_value=256, step=32),
(3, 3), activation='relu' ) )
model.add(MaxPooling2D(2, 2))
model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.05)))
model.add(Flatten())
model.add(Dense(hp.Int('dense_1_units', min_value=128, max_value=512, step=64),
activation='relu' ) )
model.add(Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.05)))
model.add(Dense(15, activation='softmax' )) # Updated to 15 output classes

model. compile(optimizer='adam', loss='categorical_crossentropy' , metrics=['accuracy' ])
return model
