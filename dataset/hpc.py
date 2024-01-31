import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
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
def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
def train_and_save_model():
    image_directory = 'BoneImages/train'

    fractured_images = os.listdir(os.path.join(image_directory, 'fractured/'))
    notfractured_images = os.listdir(os.path.join(image_directory, 'notfractured/'))

    fractured_data, fractured_labels = load_and_preprocess_images(fractured_images, 0, os.path.join(image_directory, 'fractured/'))
    notfractured_data, notfractured_labels = load_and_preprocess_images(notfractured_images, 1, os.path.join(image_directory, 'notfractured/'))

    data = np.concatenate([fractured_data, notfractured_data])
    labels = np.concatenate([fractured_labels, notfractured_labels])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    model = build_model()
    history = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Accuracy: {accuracy}')

    plot_training_history(history)

    model.save('Model.h5')
    model.summary()

train_and_save_model()
def load_and_preprocess_test_image(image_path, input_size=64):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = np.array(image) / 255.0
    return image

def predict_class(model, image_path):
    test_image = load_and_preprocess_test_image(image_path)
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
    prediction = model.predict(test_image)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Load the trained model
loaded_model = load_model('Model.h5')

# Test the model on a new image
test_image_path = 'BoneImages/train/fractured/106-rotated1.jpg'  # Replace with the actual path
predicted_class = predict_class(loaded_model, test_image_path)

# Display the test image and predicted class
test_image = cv2.imread(test_image_path)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

plt.imshow(test_image)
plt.title(f'Predicted class: {"Fractured" if predicted_class == 0 else "Not Fractured"}')

plt.axis('off')
plt.show()