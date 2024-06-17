from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import os
from PIL import Image
import encodedecode as codeur
import ia_V4 as ia
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def encrypt_message(message, algorithm, image_path):
    key, encrypt_message= codeur.encrypt_message(message, algorithm)
    
    image= ia.stegano_bits(image_path, message)

    return image, key



def decode_message(image_path, algorithm, keys):
    message= ia.get_bits(image_path)
   
    #message= codeur.decrypt_message(message_encrypte, algorithm, keys)
    return message



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        message = request.form['message']
        algorithm = request.form['algorithm']
        keys = {
            'public_key': request.form.get('public_key'),
            'private_key': request.form.get('private_key'),
            'aes_key': request.form.get('aes_key')
        }

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        encoded_image, key = encrypt_message(message, algorithm, file_path)

        # Sauvegarde de l'image encodée
        encoded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encoded_' + filename)
        encoded_image.save(encoded_image_path)  # Sauvegarde directe sans conversion
        
        print(key)
        return send_file(encoded_image_path, as_attachment=True)

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