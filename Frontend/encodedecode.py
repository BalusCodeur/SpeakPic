from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
import base64

# Générer une paire de clés RSA
def generate_rsa_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = private_key.public_key()
    return private_key, public_key

# Convertir une chaîne base64 en trame de bits
def base64_to_bits(b64_string):
    # Ajouter le remplissage nécessaire pour que la longueur soit un multiple de 4
    missing_padding = len(b64_string) % 4
    if missing_padding:
        b64_string += '=' * (4 - missing_padding)
    binary_data = base64.b64decode(b64_string)
    return ''.join(format(byte, '08b') for byte in binary_data)

# Convertir une trame de bits en chaîne de base64
def bits_to_base64(bit_string):
    if not isinstance(bit_string, str):
        raise ValueError("bit_string doit être une chaîne de caractères")

    byte_array = bytearray(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))
    return base64.b64encode(byte_array).decode('utf-8')

# Fonction pour chiffrer un message
def encrypt_message(message, encryption_type):
    if encryption_type.lower() == 'aes':
        key = get_random_bytes(16)  # Clé AES de 128 bits
        cipher = AES.new(key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(message.encode(), AES.block_size))
        encrypted_message = cipher.iv + ct_bytes
        return base64_to_bits(base64.b64encode(encrypted_message).decode('utf-8')), base64_to_bits(base64.b64encode(key).decode('utf-8'))
    
    elif encryption_type.lower() == 'rsa':
        private_key, public_key = generate_rsa_keys()
        encrypted_message = public_key.encrypt(
            message.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        encrypted_message = base64.b64encode(encrypted_message).decode('utf-8')
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        return base64_to_bits(encrypted_message), base64_to_bits(base64.b64encode(private_key_pem.encode('utf-8')).decode('utf-8'))
    else:
        raise ValueError("Le type de chiffrement doit être 'aes' ou 'rsa'.")

# Fonction pour déchiffrer un message
def decrypt_message(encrypted_message_bits, encryption_type, key_bits):
    encrypted_message = base64.b64decode(bits_to_base64(encrypted_message_bits))
    
    if encryption_type.lower() == 'aes':
        #key_bits= key_bits.get('aes_key')
        key = base64.b64decode(bits_to_base64(key_bits))
        if len(key) not in {16, 24, 32}:
            raise ValueError("AES key must be either 16, 24, or 32 bytes long.")
        iv = encrypted_message[:AES.block_size]
        ct = encrypted_message[AES.block_size:]
        cipher = AES.new(key, AES.MODE_CBC, iv)

        #print(encrypted_message_bits)
        pt = unpad(cipher.decrypt(ct), AES.block_size)
        return pt.decode('utf-8')
    
    elif encryption_type.lower() == 'rsa':
        #private_key_b64 = key_bits.get('private_key')
        key_pem = base64.b64decode(bits_to_base64(key_bits)).decode('utf-8')
        private_key = serialization.load_pem_private_key(
            key_pem.encode('utf-8'),
            password=None,
        )
        decrypted_message = private_key.decrypt(
            encrypted_message,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_message.decode('utf-8')
    else:
        raise ValueError("Le type de chiffrement doit être 'aes' ou 'rsa'.")

# Tests
if __name__ == "__main__":
    original_message = "Balus est vraiment un petit ostie en ostie d'ostie"
    aes = "aes"
    rsaa = "rsa"

    # Chiffrement AES
    encrypted_message_aes_bits, aes_key_bits = encrypt_message(original_message, aes)
    decrypted_message_aes = decrypt_message(encrypted_message_aes_bits, aes, aes_key_bits)
    print("Message original AES:", original_message)

    print("Message chiffré AES (bits):", encrypted_message_aes_bits)
    print("Clé AES (bits):", aes_key_bits)
    print("Message déchiffré AES:", decrypted_message_aes)

    # Chiffrement RSA
    encrypted_message_rsa_bits, rsa_private_key_bits = encrypt_message(original_message, rsaa)
    decrypted_message_rsa = decrypt_message(encrypted_message_rsa_bits, rsaa, rsa_private_key_bits)
    print("Message original RSA:", original_message)
    print("Message chiffré RSA (bits):", encrypted_message_rsa_bits)
    print("Clé privée RSA (bits):", rsa_private_key_bits)
    print("Message déchiffré RSA:", decrypted_message_rsa)
