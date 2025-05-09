import tensorflow as tf

# Load the model architecture from JSON file
with open('my_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)


# Output confirmation message
# print(&quot;Model architecture loaded successfully.&quot;)