import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import os


class CNN:
    def __init__(self, data_dir):
        batch_size = 32
        self.height = 180
        self.width = 180

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            color_mode="grayscale",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.height, self.width),
            batch_size=batch_size
        )

        self.class_names = train_ds.class_names
        print(self.class_names)

        try:
            self.model = tf.keras.models.load_model('model.h5')
        except Exception as e:
            print(e)

            val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                color_mode="grayscale",
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(self.height, self.width),
                batch_size=batch_size
            )

            # 생략 가능
            for image_batch, labels_batch in train_ds:
                print(image_batch.shape)
                print(labels_batch.shape)
                break

            # 생략 가능
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

            # 생략 가능
            normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
            normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
            image_batch, labels_batch = next(iter(normalized_ds))
            first_image = image_batch[0]
            print(np.min(first_image), np.max(first_image))

            data_augmentation = keras.Sequential([
                layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(self.height, self.width, 1)),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1)
            ])

            # 생략 가능
            # plt.figure(figsize=(10, 10))
            # for images, _ in train_ds.take(1):
            #     for i in range(9):
            #         augmented_images = data_augmentation(images)
            #         plt.subplot(3, 3, i + 1)
            #         plt.imshow(augmented_images[0].numpy().astype("uint8"), cmap="Greys")
            #         plt.axis("off")
            # plt.show()

            num_classes = len(self.class_names)
            self.model = Sequential([
                data_augmentation,
                layers.experimental.preprocessing.Rescaling(1. / 255),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes)
            ])
            self.model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

            # 생략 가능
            self.model.summary()

            epochs = 200
            history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)
            self.model.save('model.h5')

            # 생략 가능
            # acc = history.history['accuracy']
            # val_acc = history.history['val_accuracy']
            # loss = history.history['loss']
            # val_loss = history.history['val_loss']
            # epochs_range = range(epochs)
            # plt.figure(figsize=(8, 8))
            # plt.subplot(1, 2, 1)
            # plt.plot(epochs_range, acc, label='Training Accuracy')
            # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            # plt.legend(loc='lower right')
            # plt.title('Training and Validation Accuracy')
            # plt.subplot(1, 2, 2)
            # plt.plot(epochs_range, loss, label='Training Loss')
            # plt.plot(epochs_range, val_loss, label='Validation Loss')
            # plt.legend(loc='upper right')
            # plt.title('Training and Validation Loss')
            # plt.show()

    def predict(self, predict_dir):
        for predict in os.listdir(predict_dir):
            img = keras.preprocessing.image.load_img(
                f"{predict_dir}/{predict}",
                color_mode="grayscale",
                target_size=(self.height, self.width))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            predictions = self.model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            print(
                f"\n\"{predict}\" image most likely belongs to "
                f"{self.class_names[np.argmax(score)]} with a "
                f"{100 * np.max(score):.2f} percent confidence."
            )
