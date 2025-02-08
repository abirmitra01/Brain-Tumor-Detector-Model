import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import numpy as np
import matplotlib.pyplot as plt

# Function to build the 2D U-Net architecture
def build_unet(input_size=(128, 128, 1), initial_filters=16):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(initial_filters, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(initial_filters, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(initial_filters * 2, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(initial_filters * 2, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(initial_filters * 4, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(initial_filters * 4, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(initial_filters * 8, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(initial_filters * 8, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(initial_filters * 16, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(initial_filters * 16, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(initial_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(initial_filters * 8, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(initial_filters * 8, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(initial_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(initial_filters * 4, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(initial_filters * 4, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(initial_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(initial_filters * 2, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(initial_filters * 2, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(initial_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(initial_filters, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(initial_filters, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Prepare the U-Net model
input_shape = (128, 128, 1)
unet_model = build_unet(input_size=input_shape)
unet_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Callbacks for early stopping and model checkpointing
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1),
    callbacks.ModelCheckpoint('unet_model.keras', monitor='val_loss', save_best_only=True),
    callbacks.CSVLogger('training_log.csv')
]

# Summary of the U-Net model
unet_model.summary()

# Simulating training data (replace with actual MRI data)
x_train = np.random.rand(100, 128, 128, 1)  # 100 samples of random 128x128 grayscale images
y_train = np.random.randint(0, 2, (100, 128, 128, 1))  # Binary segmentation masks

x_val = np.random.rand(20, 128, 128, 1)
y_val = np.random.randint(0, 2, (20, 128, 128, 1))

# Train the model
history = unet_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=callbacks_list
)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy = unet_model.evaluate(x_val, y_val)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
