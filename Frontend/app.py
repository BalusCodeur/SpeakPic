from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import os
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def encrypt_message(message, algorithm, keys):
    return "encryptmessage"

def encode_message(image_path, encryptmessage):
    image = Image.open(image_path)
    return image


def decode_message(image_path, algorithm, keys):
    return "Encrypted message"

def decrypt_message(message, algorithm, keys):
    return "Decrypted message"


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

        encoded_image = encode_message(file_path, message, algorithm, keys)
        encoded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encoded_' + filename)
        encoded_image.save(encoded_image_path)
        return send_file(encoded_image_path, as_attachment=True)
    return render_template('index.html')


@app.route('/decode', methods=['POST'])
def decode_file():
    if request.method == 'POST':
        file = request.files['encoded_file']
        algorithm = request.form['algorithm']
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