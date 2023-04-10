import pandas as pd
import numpy as np
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Step 1: Data Preprocessing
data = pd.read_csv('styles3.csv')
label_map = {}
label_index = 0
for index, row in data.iterrows():
    if row['articleType'] not in label_map:
        label_map[row['articleType']] = label_index
        label_index += 1

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory=None,
    x_col="image_path",
    y_col="articleType",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory=None,
    x_col="image_path",
    y_col="articleType",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Step 2: Model Development
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_map), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Step 3: Model Optimization
# Perform hyperparameter tuning, data augmentation, regularization, etc.

# Step 4: Model Deployment
model.save('model.h5')
