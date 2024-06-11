from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import os
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def encode_message(image_path, message):
    img = Image.open(image_path)
    binary_message = ''.join(format(ord(i), '08b') for i in message)
    print("Message à encoder : " + message)
    print("Message binaire : " + binary_message)
    binary_message += '11111111'  # Marqueur de fin de message
    print("Message binaire avec marqueur : " + binary_message)

    pixels = list(img.getdata())
    print("Nombre de pixels dans l'image : " + str(len(pixels)))
    encoded_pixels = []

    if len(binary_message) > len(pixels) * 3:
        raise ValueError("Message too long to encode")

    message_index = 0
    for pixel in pixels:
        if message_index < len(binary_message):
            new_pixel = list(pixel)
            print("Pixel avant encodage :", pixel)  # Imprimer le pixel avant l'encodage
            for i in range(3):  # R, G, B channels
                if message_index < len(binary_message):
                    new_pixel[i] = new_pixel[i] & 0xFE | int(binary_message[message_index])
                    message_index += 1
            print("Pixel après encodage :", new_pixel)  # Imprimer le pixel après l'encodage
            encoded_pixels.append(tuple(new_pixel))
        else:
            encoded_pixels.append(pixel)

    print("Nombre total de pixels encodés :", len(encoded_pixels))

    encoded_img = Image.new(img.mode, img.size)
    encoded_img.putdata(encoded_pixels)
    return encoded_img

def decode_message(encoded_image_path):
    encoded_img = Image.open(encoded_image_path)
    pixels = list(encoded_img.getdata())
    binary_message = ''

    for pixel in pixels:
        for i in range(3):  # R, G, B channels
            binary_message += bin(pixel[i] & 1)[-1]

    message = ''
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i+8]
        #if byte == '11111111':  # Marqueur de fin de message
        #    break
        message += chr(int(byte, 2))

    return message




@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template('index.html')
    if request.method == 'POST':
        file = request.files['file']
        message = request.form['message']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        encoded_image = encode_message(file_path, message)
        encoded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encoded_' + filename)
        encoded_image.save(encoded_image_path)
        return send_file(encoded_image_path, as_attachment=True)
    return render_template('index.html')


@app.route('/decode', methods=['POST'])
def decode_file():
    return render_template('index.html')
    if request.method == 'POST':
        file = request.files['encoded_file']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        decoded_message = decode_message(file_path)
        return render_template('index.html', decoded_message=decoded_message)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
