from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=4, activation='softmax'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/Users/sofia/Documents/MyModels/CNNmodel/training_pics', target_size=(64,64), batch_size=32)
test_set = test_datagen.flow_from_directory('/Users/sofia/Documents/MyModels/CNNmodel/training_pics', target_size=(64,64), batch_size=32)

model = classifier.fit_generator(training_set, steps_per_epoch=200, epochs=1, validation_data=test_set, validation_steps=200)

classifier.save("model.h5")
print("Saved model to disk")

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/sofia/Documents/MyModels/CNNmodel/test_pics/Snapseed.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
model1 = classifier.load_weights("model.h5")
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction="Sofia"
    print(prediction)
elif result[0][1]==1:
    prediction='Sam'
    print(prediction)
elif result[0][2]==1:
    prediction='Lavi'
    print(prediction)
elif result[0][3]==1:
    prediction='Mamma'
    print(prediction)
