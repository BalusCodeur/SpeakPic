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
    message_encrypte, key= codeur.encrypt_message(message, algorithm)
    image= ia.hide_message(image_path, message_encrypte)

    return image, key



def decode_message(image_path, algorithm, keys):
    message_encrypte= ia.retrieve_message(image_path)
    print(type(keys))
    message= codeur.decrypt_message(message_encrypte, algorithm, keys)
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

        encoded_image, key = encrypt_message(message, algorithm, file_path)         #clef à retourner
        encoded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encoded_' + filename)
        
        stego_image_uint8 = (encoded_image * 255).astype(np.uint8)
        Image.fromarray(stego_image_uint8).save(encoded_image_path)
        print(key)
        return send_file(encoded_image_path, as_attachment=True)
    return render_template('index.html')



@app.route('/decode', methods=['POST'])
def decode_file():
    if request.method == 'POST':
        file = request.files['encoded_file']
        algorithm = request.form['decode_algorithm']
        keys = {
            'public_key': request.form.get('public_key'),
            'private_key': request.form.get('private_key'),
            'aes_key': request.form.get('aes_key')
        }

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        decoded_message = decode_message(file_path, algorithm, keys)
        return render_template('index.html', decoded_message=decoded_message)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)