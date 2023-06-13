from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('number.model')

@app.route('/predict', methods=['POST'])
def predict():
    # Convert the data URL to an image
    data_url = request.json['image']
    data = data_url.split(',', 1)[1]
    image_data = io.BytesIO(base64.b64decode(data))
    image = Image.open(image_data)

    # Convert the image to grayscale, resize it, and invert the colors for MNIST
    image = image.convert('L')
    image = image.resize((28, 28))
    image = ImageOps.invert(image)

    # Convert the image to a numpy array and reshape it for the model
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)

    # Normalize the data to 0-1
    image = image / 255.0

    # Make a prediction
    prediction = model.predict(image).argmax()  # Assumes your model returns one-hot encoded output

    # Return the prediction
    return jsonify({'prediction': int(prediction)})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
