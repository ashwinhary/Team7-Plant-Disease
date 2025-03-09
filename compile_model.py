from tensorflow.keras.optimizers import Adam
from build_model import build_model

model = build_model()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)