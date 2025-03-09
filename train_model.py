import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# ========================== #
#      DATASET PATHS         #
# ========================== #
dataset_path = '/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation'
train_dir = os.path.join(dataset_path, 'train')
val_dir = os.path.join(dataset_path, 'validation')
test_dir = os.path.join(dataset_path, 'test')

# ========================== #
#    DATA AUGMENTATION       #
# ========================== #

# Training Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,        # Reduced rotation range for faster training
    width_shift_range=0.1,    # Reduced horizontal shift
    height_shift_range=0.1,   # Reduced vertical shift
    zoom_range=0.1,           # Reduced zoom range
    horizontal_flip=True      # Horizontal flipping
)

# Validation Data (minimal augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255
)

# ========================== #
#      DATA LOADERS          #
# ========================== #

# Training Generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Validation Generator
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ========================== #
#       MODEL SETUP          #
# ========================== #

# Base Model: EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers initially
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Added dropout to prevent overfitting
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Final Model
model = Model(inputs=base_model.input, outputs=predictions)

# ========================== #
#       COMPILE MODEL        #
# ========================== #
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# ========================== #
#        CALLBACKS           #
# ========================== #
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_plant_disease_model.keras', monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# ========================== #
#      MODEL TRAINING        #
# ========================== #
epochs = 5  # Reduced epochs for faster iteration

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# ========================== #
#       FINE-TUNE BASE MODEL  #
# ========================== #
# Unfreeze the last few layers after some training
base_model.trainable = True
for layer in base_model.layers[:-5]:  # Unfreeze the last 5 layers
    layer.trainable = False

# Recompile with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training for the last few epochs if needed
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,  # Can set to continue or adjust to a higher value
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# ========================== #
#       SAVE FINAL MODEL      #
# ========================== #
model.save('final_plant_disease_model.keras')