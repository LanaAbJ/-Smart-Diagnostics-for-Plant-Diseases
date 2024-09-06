from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def custom_SCC(*args, **kwargs):
    kwargs.pop('fn', None)  # Remove the unexpected 'fn' argument if present
    return SparseCategoricalCrossentropy(*args, **kwargs)

cnn_model = load_model('C:/Users/User/Desktop/SmartDiagnosis/model/CNNplantDiseaseModelv3.h5', custom_objects={'SparseCategoricalCrossentropy': custom_SCC})


# Load models
#cnn_model = load_model('C:/Users/User/Desktop/SmartDiagnosis/model/CNNplantDiseaseModelv3.h5')
densenet_model = load_model('C:/Users/User/Desktop/SmartDiagnosis/model/densenet_modelv2.h5', custom_objects={'SparseCategoricalCrossentropy': custom_SCC})

class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Disease information dictionary
disease_info = {
    'Pepper__bell___Bacterial_spot': {
        'info': 'Caused by bacteria Xanthomonas. Common in warm, wet conditions.',
        'treatment': 'Use copper-based fungicides and avoid overhead watering.'
    },
    'Pepper__bell___healthy': {
        'info': 'No disease detected. Pepper is healthy.',
        'treatment': 'Continue regular care and monitor for any changes.'
    },
    'Potato___Early_blight': {
        'info': 'Fungal disease causing dark spots on leaf tips. Can spread quickly.',
        'treatment': 'Apply fungicides and improve air circulation around the plants.'
    },
    'Potato___Late_blight': {
        'info': 'Serious disease caused by a fungus-like organism. Can destroy crops rapidly.',
        'treatment': 'Use fungicidal spray and destroy infected plants.'
    },
    'Potato___healthy': {
        'info': 'No disease detected. Potato is healthy.',
        'treatment': 'Continue regular care and monitor for any changes.'
    }
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    if image_file:
        # Save the image to a temporary file
        temp_path = "temp_image.jpg"
        image_file.save(temp_path)
        
        # Read image using cv2
        #img = cv2.imread(temp_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        # Resize image using TensorFlow/Keras preprocessing
        image = tf.keras.preprocessing.image.load_img(temp_path,target_size=(256,256))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        # Make predictions
        cnn_predictions = cnn_model.predict(input_arr)
        densenet_predictions = densenet_model.predict(input_arr)

        # Get the results
        cnn_result_index = np.argmax(cnn_predictions)
        dn_result_index = np.argmax(densenet_predictions)

        cnn_prediction = class_names[cnn_result_index]
        dn_prediction = class_names[dn_result_index]

        cnn_info = disease_info.get(cnn_prediction, {'info': 'No information available', 'treatment': 'No treatment suggested'})

        return render_template('results.html',
                                cnn_prediction=cnn_prediction,
                                dn_prediction=dn_prediction,
                                cnn_info=cnn_info['info'], 
                                cnn_treatment=cnn_info['treatment'])

    return 'No image provided', 400

if __name__ == '__main__':
    app.run(debug=True)
