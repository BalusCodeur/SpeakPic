# Utiliser l'image TensorFlow officielle comme image de base
FROM tensorflow/tensorflow:latest

# Installation de TensorFlow Datasets
RUN pip install tensorflow-datasets

# Copie des fichiers sources dans le conteneur
COPY gan.py /app/gan.py

# Commande par défaut à exécuter lorsque le conteneur démarre
CMD ["python", "/app/gan.py"]
