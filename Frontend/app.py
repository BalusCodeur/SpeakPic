from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import os
from PIL import Image
import encodedecode as codeur
import ia_V4 as ia
import numpy as np
from io import BytesIO
from zipfile import ZipFile

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def encrypt_message(message, algorithm, image_path):
    encrypt_message, key= codeur.encrypt_message(message, algorithm)
    
    image= ia.stegano_bits(image_path, message)

    return image, key



def decode_message(image_path, algorithm, keys):
    message= ia.get_bits(image_path)
   
    #message= codeur.decrypt_message(message_encrypte, algorithm, keys)
    return message

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        message = request.form['message']
        algorithm = request.form['algorithm']


        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        encoded_image, key = encrypt_message(message, algorithm, file_path)

        # Sauvegarde de l'image encodée
        encoded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encoded_' + filename)
        encoded_image.save(encoded_image_path)  # Sauvegarde directe sans conversion
        key_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'key.txt')

        with open(key_file_path, 'w') as key_file:
            if algorithm.lower() == 'aes':
                key_file.write(f"AES Key: {key}\n")
            elif algorithm.lower() == 'rsa':
                print("KEY")
                print(key)
                private_key_pem, public_key_pem = key
                key_file.write(f"Public Key (PEM):\n{public_key_pem}\n")
                key_file.write(f"Private Key (PEM):\n{private_key_pem}\n")
            else:
                raise ValueError("Le type de chiffrement doit être 'aes' ou 'rsa'.")
        
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, 'w') as zip_file:
            # Ajouter l'image encodée au fichier ZIP
            zip_file.write(encoded_image_path, arcname='encoded_image.jpg')
            # Ajouter la clé au fichier ZIP
            zip_file.write(key_file_path, arcname='key.txt')

        # Retourner le fichier ZIP en tant que réponse
        zip_buffer.seek(0)

        return send_file(zip_buffer, as_attachment=True,download_name="Encoded.zip")
    return render_template('index.html')




@app.route('/decode', methods=['POST'])
def decode_file():
    if request.method == 'POST':
        file = request.files['encoded_file']
        algorithm = request.form['decode_algorithm']
        keys = {
            'public_key': request.form.get('decode_public_key'),
            'private_key': request.form.get('decode_private_key'),
            'aes_key': request.form.get('decode_aes_key')
        }

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        decoded_message = decode_message(file_path, algorithm, keys)
        return render_template('index.html', decoded_message=decoded_message)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)