# SpeakPic
SpeakPic, appli web tres cool de steganographie et chiffrement.

Pour commencer veuiller installer python en utilisant cette documentation: https://www.python.org/
Pour lancer l'application commencer par télécharger les bibliothèques nécessaires présentes dans requirements.txt
```bash
pip install -r requirements.txt

```

Enfin vous pouvez lancer l'application en allant dans le dossier frontend et en lançant la commande:
```bash
python .\app.py
```

## Utilisation

### Encodage
Pour cacher un message dans une image, choisir une image, inscrire son message et choisir le type d'encryption dans la section Encode.

Cliquer sur le bouton, une archive contenant l'image avec le message ainsi que la ou les clés d'encryption est téléchargée.

### Décodage
Pour décoder, sélectionner l'image et le fichier `key.txt` dans les champs correspondants et préciser l'algorithme. Le message devrait s'afficher en bas de la fenêtre.