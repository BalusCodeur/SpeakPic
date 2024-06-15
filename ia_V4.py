import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from PIL import Image  



def load_and_resize_image(image_path, target_size=(256, 256)):
    try:
        image = Image.open(image_path)
        image = image.resize(target_size, Image.ANTIALIAS)
        image = np.array(image) / 255.0  
        return image
    except Exception as e:
        print(f"Error loading or resizing image: {e}")
        return None




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



# Fonction pour formater le message binaire
def format_message(message, target_shape):
    message_list = [int(bit) for bit in message]
    target_size = np.prod(target_shape)
    if len(message_list) < target_size:
        message_list += [0] * (target_size - len(message_list))
    formatted_message = np.array(message_list).reshape(target_shape)
    return formatted_message.astype('float32')

def hide_message(image, message):
    
    image = load_and_resize_image(image)
    if image is None:
        print("Erreur lors du chargement de l'image.")
        exit()
   
    message = np.array([int(bit) for bit in message])
    stego_image = np.copy(image)
    
    message_length = len(message)
    max_length = stego_image.shape[0] * stego_image.shape[1]  # Calculer la longueur maximale du message possible

    if message_length > max_length:
        raise ValueError("Le message est trop long pour être caché dans cette image.")
    flat_image = stego_image.reshape(-1, 3)  
    flat_image[-message_length:, -1] = message  
    stego_image = flat_image.reshape(stego_image.shape)
    autoencoder = create_autoencoder(input_shape=image.shape)
    autoencoder.fit(image[np.newaxis, ...], image[np.newaxis, ...], epochs=10, batch_size=1)    
    return stego_image





def retrieve_message(stego_image):
    flat_image = stego_image.reshape(-1, 3)
    retrieved_message = flat_image[:, -1].astype(int)
    return clean(retrieved_message)


def clean(retrieved_message):
    end_index = None
    for i in range(len(retrieved_message) - 1, -1, -1):
        if retrieved_message[i] == 1:
            end_index = i
            break
    
    if end_index is None:
        raise ValueError("Message non trouvé dans le message récupéré.")
    
    hidden_message = retrieved_message[end_index - 15:end_index + 1]  
    return hidden_message


####TEST
image_path = 'test2.jpg'  # Chemin de l'image à charger et à traiter
original_message = "0001010101010110"  # Message binaire à cacher dans l'image

img=hide_message(image_path, original_message)
print(retrieve_message(img))
