import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import cProfile
import pstats

def load_and_preprocess_images(image_list, label, image_directory, input_size=64):
    data = []
    labels = []
    for image_name in image_list:
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(os.path.join(image_directory, image_name))
            image = Image.fromarray(image, 'RGB')
            image = image.resize((input_size, input_size))
            data.append(np.array(image))
            labels.append(label)
    return data, labels

def build_model(input_shape=(64, 64, 3), num_classes=2):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_save_model():
    image_directory = './dataset/BoneImages/train'

    fractured_directory = os.path.join(image_directory, 'fractured')
    notfractured_directory = os.path.join(image_directory, 'notfractured')

    fractured_images = os.listdir(fractured_directory)
    notfractured_images = os.listdir(notfractured_directory)

    fractured_data, fractured_labels = load_and_preprocess_images(fractured_images, 0, fractured_directory)
    notfractured_data, notfractured_labels = load_and_preprocess_images(notfractured_images, 1, notfractured_directory)

    data = np.concatenate([fractured_data, notfractured_data])
    labels = np.concatenate([fractured_labels, notfractured_labels])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    model = build_model()
    profiler = cProfile.Profile()
    profiler.enable()

    history = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test))

    profiler.disable()

    profiler.dump_stats('profile_data.prof')

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Accuracy: {accuracy}')

  
    profiler_stats = pstats.Stats('profile_data.prof')
    profiler_stats.strip_dirs().sort_stats('cumulative').print_stats()


train_and_save_model()
