import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from PIL import Image  

def string_to_binary(string):
    binary_representation = ""
    for char in string:
        binary_representation += format(ord(char), '08b')  # Convertir chaque caractère en 8 bits binaire
    return binary_representation

def load_and_resize_image(image_path, target_size=(256, 256)):
    try:
        image = Image.open(image_path)
        image = image.resize(target_size, Image.ANTIALIAS)
        image = np.array(image) / 255.0  
        return image
    except Exception as e:
        print(f"Error loading or resizing image: {e}")
        return None

def binary_to_string(binary_representation):
    string = ""
    for i in range(0, len(binary_representation), 8):
        byte = binary_representation[i:i+8]
        char = chr(int(byte, 2))  # Convertir chaque octet binaire en caractère ASCII
        string += char
    return string


def create_autoencoder(input_shape=(256, 256, 3)):
    input_img = Input(shape=input_shape)

    # Encodeur
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Décodeur
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) 

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')  # Utilisation de la Mean Squared Error pour la reconstruction

    return autoencoder



def stegano_bits(image_path, bit_string):
    bit_string=string_to_binary(bit_string)
    im = Image.open(image_path)
    image=np.copy(im)
    w, h = im.size


    r, g, b = im.split()

    r = list(r.getdata())


    msg_length = len(bit_string)
    msg_length_bin = bin(msg_length)[2:].rjust(8, "0")


    for j in range(8):
        r[j] = 2 * (r[j] // 2) + int(msg_length_bin[j])


    for i in range(msg_length):
        r[i + 8] = 2 * (r[i + 8] // 2) + int(bit_string[i])


    nr = Image.new("L", (w, h))
    nr.putdata(r)


    img_new = Image.merge('RGB', (nr, g, b))
    #autoencoder = create_autoencoder(input_shape=image.shape)
    #autoencoder.fit(image[np.newaxis, ...], image[np.newaxis, ...], epochs=10, batch_size=1)

    return img_new
    
    
def get_bits(image_path):
    im = Image.open(image_path)
    r, g, b = im.split()


    r = list(r.getdata())


    msg_length_bin = ''.join([str(x % 2) for x in r[:8]])
    msg_length = int(msg_length_bin, 2)

    bit_string = ''.join([str(r[i + 8] % 2) for i in range(msg_length)])

    hidden_message = binary_to_string(bit_string)

    return hidden_message

image_path = "test2.png"
bit_string_to_hide = "101100101011001011101001"  # Exemple de chaîne de bits à cacher
stegano_bits(image_path, bit_string_to_hide)

# Récupérer la chaîne de bits cachée dans l'image modifiée
stegano_image_path = "stegano_test2.png"
retrieved_bit_string = get_bits(stegano_image_path)

print("Chaîne de bits récupérée :", retrieved_bit_string) 