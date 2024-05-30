from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.conf import settings

import keras
import numpy as np
import tensorflow as tf
import cv2

from keras.preprocessing.image import load_img
from tensorflow.python.keras.backend import set_session

def index(request):
    if request.method == 'POST':
        file = request.FILES['imageFile']
        file_name = default_storage.save(file.name, file)
        file_path = default_storage.path(file_name)

        image = load_img(file_path)

        faces = settings.FACE_CASCADER.detectMultiScale(cv2.imread(file_path), 1.3, 5)
        print("DEBUG: ", image.size)

        for (x, y, w, h) in faces: 
            image = image.crop((x, y, x + w, y + h))
            image = image.resize((224, 224))
            break
        
        if len(faces) == 0:
            image = image.resize((224, 224))

        # Save the image to see if it is cropped correctly
        #image.save('test.jpg', 'JPEG')

        numpy_image = np.array(image)
        numpy_image = np.array(numpy_image).astype('float32') / 255
        image_batch = np.expand_dims(numpy_image, axis=0)

        predictions = settings.IMAGE_MODEL.predict(image_batch)
        predictedMood = decode_emotions(predictions)

        print("DEBUG: ", predictedMood)

        return render(request, 'index.html', {'predictions': predictedMood})
    else:
        return render(request, 'index.html')
        
    return render(request, 'index.html')

def decode_emotions(predictions):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    return emotions[np.argmax(predictions)]