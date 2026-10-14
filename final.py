import os
import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint, ProgbarLogger
from keras.utils import Sequence

# Directory containing the binary files
dataset_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)

# Image dimensions
IMAGE_SHAPE = (256, 256)

# --- Dataset Loading ---
def load_binary_file(file_path):
    """
    Load a single binary file and extract time and image data.
    """
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        time = data[0]  # First float is the time
        image = data[1:].reshape(IMAGE_SHAPE)  # Remaining floats are the image pixels
    return time, image

def load_dataset(dataset_directory):
    """
    Load all binary files in the dataset directory.
    """
    files = sorted([os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory) if f.endswith('.data')])
    times = []
    images = []
    for file in files:
        time, image = load_binary_file(file)
        times.append(time)
        images.append(image)
    return np.array(times, dtype=np.float32), np.array(images, dtype=np.float32)

# Load the entire dataset (images and times are assumed to be normalized)
times, images = load_dataset(dataset_directory)

# Reshape images for Keras
images = images.reshape(-1, 256, 256, 1)

# Create all pairs of indices
pairs = []
for i in range(len(times) - 10):  # Ensure at least 4 following images exist
    for j in range(1, 6):  # Pair with the next 4 images
        pairs.append((i, i + j*2))

# Shuffle the pairs
random.shuffle(pairs)

# Split the pairs into training and evaluation sets
split_index = int(0.8 * len(pairs))  # 80% for training, 20% for evaluation
train_pairs = pairs[:split_index]
eval_pairs = pairs[split_index:]

def augment_image(image):
    """
    Augment an image with rotations and symmetries.
    """
    augmented_images = [image]  # Start with the original image
    augmented_images.append(np.rot90(image, k=1, axes=(0, 1)))  # 90 degrees
    augmented_images.append(np.rot90(image, k=2, axes=(0, 1)))  # 180 degrees
    augmented_images.append(np.rot90(image, k=3, axes=(0, 1)))  # 270 degrees
    augmented_images.append(np.flipud(image))  # Vertical flip
    augmented_images.append(np.fliplr(image))  # Horizontal flip
    return augmented_images

class PairedImageGenerator(Sequence):
    def __init__(self, times, images, pairs, batch_size=32):
        self.times = times
        self.images = images
        self.pairs = pairs
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        batch_pairs = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_input_images = []
        batch_input_times = []
        batch_delta_ts = []
        batch_output_images = []

        for input_idx, output_idx in batch_pairs:
            # Input image and time
            input_image = self.images[input_idx]
            input_time = self.times[input_idx]

            # Output image and Δt
            output_image = self.images[output_idx]
            delta_t = self.times[output_idx] - input_time

            # Augment both input and output images
            input_augmented = augment_image(input_image)
            output_augmented = augment_image(output_image)

            # Append all augmented pairs to the batch
            for in_img, out_img in zip(input_augmented, output_augmented):
                batch_input_images.append(in_img)
                batch_input_times.append(input_time)  # Time remains the same
                batch_delta_ts.append(delta_t)       # Δt remains the same
                batch_output_images.append(out_img)

        return [np.array(batch_input_images),
                np.array(batch_input_times).reshape(-1, 1),
                np.array(batch_delta_ts).reshape(-1, 1)], np.array(batch_output_images)

# --- Model Creation ---
# Define the model
def create_model():
    image_input = Input(shape=(256, 256, 1), name="image_input")  # Grayscale image input
    scalar_input = Input(shape=(1,), name="time_input")           # Scalar input for time
    delta_input = Input(shape=(1,), name="delta_t_input")         # Scalar input for delta_t

    # Process scalar inputs to expand dimensions
    scalar_features = Concatenate()([scalar_input, delta_input])
    scalar_features = UpSampling2D(size=(256, 256))(Reshape((1, 1, 2))(scalar_features))

    # Concatenate image with scalar features
    x = Concatenate()([image_input, scalar_features])

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder: Reconstruct the difference image
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    difference_image = Conv2D(1, (3, 3), activation='tanh', padding='same', name="difference_image")(x)

    # Add difference image to input image to get final output
    output_image = Add(name="output_image")([image_input, difference_image])

    # Create the model
    model = Model(inputs=[image_input, scalar_input, delta_input], outputs=output_image)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

model = create_model()

model.summary()

# --- Training ---
# Callbacks for EarlyStopping and ModelCheckpoint
best_model_path = "best_model.h5"
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, verbose=1)
callbacks = [early_stopping, model_checkpoint]
# Initialize training and evaluation generators
batch_size = 4
train_generator = PairedImageGenerator(times, images, train_pairs, batch_size=batch_size)
eval_generator = PairedImageGenerator(times, images, eval_pairs, batch_size=batch_size)

model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=eval_generator,
    validation_steps=len(eval_generator),
    callbacks=callbacks
)
