import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
from PIL import Image
import matplotlib.pyplot as plt

# Dimensions
image_shape = (64, 64, 3)
message_dim = 100

def load_data(image_folder):
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        try:
            image = Image.open(image_path).convert('RGB').resize((64, 64))
            image = np.array(image) / 127.5 - 1.0  # Normaliser l'image entre -1 et 1
            images.append(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    return np.array(images)

# Générateur
def build_generator():
    noise_input = layers.Input(shape=(message_dim,))
    image_input = layers.Input(shape=image_shape)
    
    x = layers.Dense(8 * 8 * 256)(noise_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((8, 8, 256))(x)
    
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    hidden_image = layers.Conv2D(3, kernel_size=3, padding="same", activation='tanh')(x)
    combined_image = layers.Add()([image_input, hidden_image])
    
    model = models.Model([noise_input, image_input], combined_image)
    return model

# Discriminateur
def build_discriminator():
    image_input = layers.Input(shape=image_shape)
    
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(image_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(image_input, x)
    return model


generator = build_generator()
discriminator = build_discriminator()


def build_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False
    
    noise_input = layers.Input(shape=(message_dim,))
    image_input = layers.Input(shape=image_shape)
    
    generated_image = generator([noise_input, image_input])
    validity = discriminator(generated_image)
    
    model = models.Model([noise_input, image_input], validity)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


gan = build_gan(generator, discriminator)


def train_gan(gan, generator, discriminator, image_folder, epochs, batch_size=128):
    X_train = load_data(image_folder)
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    

    d_losses = []
    g_losses = []
    
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, message_dim))
        gen_imgs = generator.predict([noise, imgs])
        
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, message_dim))
        g_loss = gan.train_on_batch([noise, imgs], valid)
        
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        
        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]*100:.2f}%] [G loss: {g_loss}]")
    
    # Afficher les courbes de métriques à la fin de l'entraînement
    plot_metrics(d_losses, g_losses)

# Fonction pour tracer les métriques
def plot_metrics(d_losses, g_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator loss', alpha=0.8)
    plt.plot(g_losses, label='Generator loss', alpha=0.8)
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Entraîner le GAN
epochs = 100
batch_size = 70
train_gan(gan, generator, discriminator, 'image_test', epochs, batch_size)




def hide_message(image, message, generator_model):
   
    message_bits = np.array([float(bit) for bit in message])
    

    noise = np.random.normal(0, 1, (1, message_dim))
    
    image_preprocessed = (image.astype(np.float32) / 127.5) - 1.0
    message_preprocessed = message_bits * 2 - 1  
    

    generated_image = generator_model.predict([noise, image_preprocessed.reshape((1, 64, 64, 3))])
    

    generated_image = ((generated_image[0] + 1) * 127.5).astype(np.uint8)
    
    return generated_image

def extract_message(image_with_message, discriminator_model):

    image_preprocessed = (image_with_message.astype(np.float32) / 127.5) - 1.0
    
    # Prédire la validité de l'image 
    validity = discriminator_model.predict(image_preprocessed.reshape((1, 64, 64, 3)))
    

    extracted_message = ''.join(['1' if v > 0.5 else '0' for v in validity[0]])
    
    return extracted_message


### test
message = "1010101010"


image_path = 'test.jpg'
image = np.array(Image.open(image_path).convert('RGB').resize((64, 64)))

image_with_message = hide_message(image, message, generator)


extracted_message = extract_message(image_with_message, discriminator)

print(f"Message caché : {message}")
print(f"Message extrait : {extracted_message}")
