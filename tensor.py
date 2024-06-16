from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf  # Ensure TensorFlow is installed

# Load your trained TensorFlow model (replace with your actual loading logic)
model = tf.keras.models.load_model('modelvirus.keras')  # Assuming your model is saved as 'modelvirus.keras'
max_length = 1459  # Replace with the actual maximum sequence length from your training process

def predict_virus_class(sequence_string):
  """
  Predicts the class of a new virus sequence using the trained model.

  Args:
      sequence_string: A comma-separated string representing the virus sequence.

  Returns:
      predicted_class: The predicted class label for the sequence.
  """

  # Preprocess the sequence string
  sequence_array = np.array([int(num) for num in sequence_string.split(',') if num.strip()])
  sequence_padded = np.pad(sequence_array, (0, max_length - len(sequence_array)), padding='post')
  sequence_padded = sequence_padded[np.newaxis, :]  # Reshape to add batch dimension

  # Make prediction
  predicted_probabilities = model.predict(sequence_padded)
  predicted_class = np.argmax(predicted_probabilities, axis=1)[0]

  return predicted_class

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  # Get the sequence data from the request
  data = request.get_json()
  if not data or 'sequence' not in data:
    return jsonify({'error': 'Missing sequence data'}), 400

  sequence_string = data['sequence']

  # Make prediction and return the result
  predicted_class = predict_virus_class(sequence_string)
  return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
  # Run the Flask app (use port appropriate for Heroku deployment)
  app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

