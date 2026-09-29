import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training images:", X_train.shape)
print("Testing images :", X_test.shape)


X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(10, activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)


test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", round(test_accuracy, 4))


index = 0

image = X_test[index]
prediction = model.predict(image.reshape(1, 28, 28, 1))

predicted_digit = np.argmax(prediction)

print("Actual digit   :", y_test[index])
print("Predicted digit:", predicted_digit)


plt.imshow(X_test[index].reshape(28, 28), cmap="gray")
plt.title(
    f"Actual: {y_test[index]} | Predicted: {predicted_digit}"
)
plt.axis("off")
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training Accuracy")
plt.legend()

plt.show()