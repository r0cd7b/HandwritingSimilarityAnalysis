import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import os


# CNN Model을 구축하고 관련 작업을 수행하는 Class.
class CNN:
    def __init__(self, browser):
        self.__browser = browser  # 수행 과정에서 발생한 Text를 출력하기 위해 Flame 객체를 browser로 받는다.
        
        self.__height = 180
        self.__width = 180
        self.__train_ds = None
        self.__class_names = None
        
        try:
            self.__model = tf.keras.models.load_model('model.h5')  # 이미 학습된 Model이 있다면 불러온다.
        except Exception as e:
            self.__model = None
            self.__browser.append(str(e))

    def train(self):
        data_dir = "train"
        batch_size = 32  # 전체 Data set을 나눠서 학습하기 위한 Batch size.
        
        self.__train_ds = tf.keras.preprocessing.image_dataset_from_directory(  # Directory에서 학습할 image를 불러온다.
            data_dir,
            color_mode="grayscale",  # 필기체 분석에서는 색이 중요하지 않으므로 grayscale로 변환한다.
            validation_split=0.2,  # 전체 Train data 중 검증 Data는 20%로 한다.
            subset="training",
            seed=123,  # Random seed를 고정하고 이후 검증 Data를 나눌 때 동일한 Seed를 사용한다.
            image_size=(self.__height, self.__width),
            batch_size=batch_size
        )
        self.__class_names = self.__train_ds.class_names  # 학습할 필기체 주인의 이름을 저장한다.
        self.__browser.append(f"Class names: {self.__class_names}")
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="validation",
            seed=123,  # Train data를 제외한 나머지 검증 Data를 뽑아내기 위해 앞서 사용한 seed와 동일하게 맞춘다.
            image_size=(self.__height, self.__width),
            batch_size=batch_size
        )

        # Batch data의 shape을 확인한다.
        for image_batch, labels_batch in self.__train_ds:
            self.__browser.append(f"Image batch shape: {image_batch.shape}")
            self.__browser.append(f"Labels batch shape: {labels_batch.shape}")
            break

        # 더 빠른 기억장치 사용을 위해 보조기억장치의 cache를 활용한다.
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.__train_ds = self.__train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Train에 사용될 Image 무작위로 변형시켜 양을 증강한다.
        data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(self.__height, self.__width, 1)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1)
        ])
        # 변형된 9개의 Image를 확인한다.
        plt.figure(figsize=(10, 10))
        for images, _ in self.__train_ds.take(1):
            for i in range(9):
                augmented_images = data_augmentation(images)
                plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_images[0].numpy().astype("uint8"), cmap="gray")
                plt.axis("off")

        # CNN Model을 구축한다.
        num_classes = len(self.__class_names)
        self.__model = Sequential([
            data_augmentation,
            layers.experimental.preprocessing.Rescaling(1. / 255),  # 학습을 위해 0 ~ 255 범위의 Pixel 값을 0 ~ 1 범위로 Rescale한다.
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)  # 최종 Output 수는 훈련할 필기체 주인의 수이다.
        ])
        self.__model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        # 최종 구성된 Layer들을 확인한다.
        self.__model.summary()

        # 구성된 Model을 이용하여 훈련 Data를 학습한다.
        epochs = 100  # 총 학습 반복 횟수를 정한다.
        history = self.__model.fit(  # 학습을 수행하고 과정은 history에 기록한다.
            self.__train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        self.__model.save('model.h5')  # 완성된 Model을 저장한다.

        # 기록된 학습 과정을 Graph로 그려낸다.
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

    def test(self):
        if self.__model:  # 학습된 Model이 없을 때 수행되지 않게 검사한다.
            predict_dir = "predict"
            for predict in os.listdir(predict_dir):
                img = keras.preprocessing.image.load_img(  # 분류할 Image를 불러온다.
                    f"{predict_dir}/{predict}",
                    color_mode="grayscale",
                    target_size=(self.__height, self.__width)
                )
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                predictions = self.__model.predict(img_array)  # 학습된 Model을 활용하여 필기체를 분석한다.
                score = tf.nn.softmax(predictions[0])
                self.__browser.append(  # 분류 결과와 확신 정도를 출력한다.
                    f"\"{predict}"
                    f"\" image most likely belongs to {self.__class_names[np.argmax(score)]}"
                    f" with a {100 * np.max(score):.2f}"
                    f" percent confidence."
                )
        else:
            self.__browser.append("Model does not exist. Please train first.")
