import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: PREPARE THE DATA 
print("Loading MNIST Database (Handwritten digits)...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images to be between 0.0 and 1.0
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel dimension (Make it 28x28x1 for the AI)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


# STEP 2: BUILD & TRAIN THE AI MODEL
def create_model():
    model = tf.keras.models.Sequential([
        # The "Eyes": Look for features (lines, curves)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # The "Brain": Decide what the features mean
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        # The "Output": Score for digits 0-9
        tf.keras.layers.Dense(10) 
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

print("Training the model... (This might take 1-2 minutes)")
model = create_model()
model.fit(x_train, y_train, epochs=3) 


# STEP 3: THE ATTACK FUNCTION (FGSM)
def fast_gradient_sign_method(input_image, input_label, model, epsilon):
    # Convert data to TensorFlow tensors
    input_image = tf.convert_to_tensor(input_image)
    input_label = tf.convert_to_tensor(input_label)
    
    with tf.GradientTape() as tape:
        tape.watch(input_image) # IMPORTANT: Watch the image, not the weights
        prediction = model(input_image)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(input_label, prediction)

    # Get the gradient (direction of the error)
    gradient = tape.gradient(loss, input_image)
    
    # Get the sign of the gradient (+1 or -1)
    signed_grad = tf.sign(gradient)
    
    # Create the noise (epsilon * direction)
    noise = epsilon * signed_grad
    
    # Add noise to image
    adversarial_image = input_image + noise
    
    # Ensure pixel values stay valid (0 to 1)
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image, noise

# STEP 4: RUN THE ATTACK
# Pick a random image (Index 12 is usually a '9')
image_index =  12
image = x_test[image_index]
label = y_test[image_index]

# Reshape for the model (Batch size of 1)
image_tensor = tf.convert_to_tensor(image.reshape(1, 28, 28, 1))
label_tensor = tf.convert_to_tensor(label.reshape(1))

# Get Clean Prediction
pred_logits = model(image_tensor)
pred_label = tf.argmax(pred_logits[0])
print(f"\nTRUE LABEL: {label}")
print(f"AI PREDICTION (CLEAN): {pred_label.numpy()}")

# ATTACK!
epsilon = 0.1 # Strength of attack (0.1 is subtle, 0.5 is obvious)
adv_image, noise = fast_gradient_sign_method(image_tensor, label_tensor, model, epsilon)

# Get Attacked Prediction
adv_logits = model(adv_image)
adv_label = tf.argmax(adv_logits[0])
print(f"AI PREDICTION (ATTACKED): {adv_label.numpy()}")


# STEP 5: VISUALIZE RESULTS
plt.figure(figsize=(10, 4))

# Original
plt.subplot(1, 3, 1)
plt.title(f"Original\nPred: {pred_label.numpy()}")
plt.imshow(image.squeeze(), cmap='gray')
plt.axis('off')

# Noise
plt.subplot(1, 3, 2)
plt.title("The Poison (Noise)")
plt.imshow(noise[0].numpy().squeeze(), cmap='gray')
plt.axis('off')

# Attacked
plt.subplot(1, 3, 3)
plt.title(f"Adversarial\nPred: {adv_label.numpy()}")
plt.imshow(adv_image[0].numpy().squeeze(), cmap='gray')
plt.axis('off')

print("Displaying image... Check the popup window!")

plt.show()
