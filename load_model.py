from tensorflow.keras.models import load_model

# Define the custom loss function again
def weighted_loss(y_true, y_pred):
    # Example loss (replace with your actual implementation)
    import tensorflow as tf
    weights = tf.constant([1.0, 2.0, 1.5])  # Example weights for classes
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    weighted = loss * tf.reduce_sum(weights * y_true, axis=-1)
    return tf.reduce_mean(weighted)

# Now load the model with custom_objects
model = load_model('best_plant_disease_model.keras', custom_objects={'weighted_loss': weighted_loss})