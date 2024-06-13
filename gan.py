import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2

# Fonction pour générer le GAN
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def hide_message(image, message):
    # Convertir le message binaire en une séquence de bits
    message_bits = [int(bit) for bit in message]
    
    # Convertir la longueur du message en une séquence de bits
    message_length_bits = [int(bit) for bit in format(len(message_bits), '032b')]

    # Combiner les bits de longueur et les bits du message
    combined_bits = message_length_bits + message_bits

    # Convertir l'image en une matrice numpy modifiable
    stego_image = np.copy(image)

    # Vérifier si l'image a suffisamment de pixels pour cacher le message
    num_pixels_required = len(combined_bits)
    if num_pixels_required > (stego_image.shape[0] * stego_image.shape[1] * stego_image.shape[2]):
        raise ValueError("L'image est trop petite pour cacher ce message.")

    # Appliquer la substitution de pixels pour cacher le message
    pixel_index = 0
    for bit in combined_bits:
        # Obtenir les coordonnées du pixel dans l'image
        row = pixel_index // (image.shape[1] * 3)
        col = (pixel_index % (image.shape[1] * 3)) // 3
        channel = pixel_index % 3

        # Modifier la valeur du pixel pour encoder le bit du message
        stego_image[row, col, channel] = (stego_image[row, col, channel] & 0xFE) | bit  # Réinitialiser le bit de poids faible et le remplacer par le bit du message

        pixel_index += 1

    return stego_image


def reveal_message(image):
    # Extraire les 32 premiers bits pour obtenir la longueur du message
    length_bits = []
    pixel_index = 0

    while pixel_index < 32:
        row = pixel_index // (image.shape[1] * 3)
        col = (pixel_index % (image.shape[1] * 3)) // 3
        channel = pixel_index % 3

        # Extraire le bit de poids faible du pixel
        bit = image[row, col, channel] & 1
        length_bits.append(bit)

        pixel_index += 1

    # Convertir les bits de longueur en un entier
    message_length = int(''.join(map(str, length_bits)), 2)

    # Extraire les bits du message
    message_bits = []
    total_pixels = image.shape[0] * image.shape[1] * 3

    while pixel_index < total_pixels and len(message_bits) < message_length:
        row = pixel_index // (image.shape[1] * 3)
        col = (pixel_index % (image.shape[1] * 3)) // 3
        channel = pixel_index % 3

        # Extraire le bit de poids faible du pixel
        bit = image[row, col, channel] & 1
        message_bits.append(bit)

        pixel_index += 1

    # Convertir la séquence de bits en une chaîne de caractères binaire
    message = ''.join(map(str, message_bits))

    return message

# Génération du GAN
generator = make_generator_model()

# Charger le message à cacher

binary_message = "0101010101010101"  # Votre message binaire ici

# Charger l'image à traiter
image_path = "test.jpg"  # Chemin de votre image ici
original_image = cv2.imread(image_path)

# Cacher le message dans l'image
stego_image = hide_message(original_image, binary_message)

# Enregistrer l'image avec le message caché
cv2.imwrite("stego_image.png", stego_image)

# Récupérer le message de l'image
retrieved_message = reveal_message(stego_image)
print("Message retrieved:", retrieved_message)
