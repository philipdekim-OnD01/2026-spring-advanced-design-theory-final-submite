import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomTranslation(0.12, 0.12),
    tf.keras.layers.RandomZoom(0.10),
], name="data_augmentation")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    data_augmentation,

    tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "digits_model_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
]

history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Final test accuracy:", acc)

model.save("digits_model_cnn.keras")

print("Saved: digits_model_cnn.keras")
print("Best model saved: digits_model_best.keras")