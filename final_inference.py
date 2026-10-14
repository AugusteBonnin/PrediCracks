import numpy as np
from keras.models import load_model
from PIL import Image
import os

# --- Load the model ---
model_path = "./best_model.h5"
model = load_model(model_path)

# --- Parameters ---
image_shape = (256, 256, 1)  # Shape of the image
delta_t = 0.01  # Step size for time increments
output_directory = "."  # Directory to save outputs
num_random_black_points = 3000  # Number of random points to set to black

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Create the initial image
input_image = np.full(image_shape, 0.95, dtype=np.float32)  # Gray level 0.5

# Set a few random points to black (0.0)
random_points = np.random.randint(0, 256, size=(num_random_black_points, 2))  # Random points
for point in random_points:
    input_image[point[0], point[1], 0] = 0.0  # Set to black

# Initialize time
current_time = 0.0

# Perform inference
step = 0
while current_time <= 1.0:
    # Prepare inputs for the model
    input_time = np.array([[current_time]], dtype=np.float32)  # Current time as scalar input
    delta_input = np.array([[delta_t]], dtype=np.float32)      # Step size as scalar input
    input_image_batch = input_image[np.newaxis, ...]           # Add batch dimension

    # Infer the output image
    output_image = model.predict([input_image_batch, input_time, delta_input])[0]  # Remove batch dimension

    # Save the generated output image to a common image format
    output_image_clipped = np.clip(output_image * 255, 0, 255).astype(np.uint8)  # Scale to 0-255
    output_image_pil = Image.fromarray(output_image_clipped.squeeze(), mode='L')  # Convert to grayscale image
    output_image_pil.save(f"{output_directory}/generated_image_step_{step:03d}.png")

    # Update the input image for the next step
    input_image = output_image

    # Increment the time
    current_time += delta_t
    step += 1

print(f"Inference completed. Images saved to {output_directory}.")
