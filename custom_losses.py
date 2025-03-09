import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def weighted_loss(y_true, y_pred):
 # Example: Custom weighted loss function (modify as per your needs)
    weights = tf.constant([1.0, 2.0])  # Example weights for two classes
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    weighted_loss = loss * tf.reduce_sum(weights)
    return tf.reduce_mean(weighted_loss)