import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

# ========================== #
#      LOAD SAVED MODEL      #
# ========================== #
model = load_model('final_plant_disease_model.keras')

# ========================== #
#      LOAD TEST DATA        #
# ========================== #
test_dir = '/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/test'

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ========================== #
#        MODEL EVALUATE      #
# ========================== #
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ========================== #
#     CLASSIFICATION REPORT  #
# ========================== #
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
print(report)

# Save report to file
with open('classification_report.txt', 'w') as f:
    f.write(report)

# ========================== #
#     CONFUSION MATRIX       #
# ========================== #
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# ========================== #
#     BAR CHART OF ACCURACY  #
# ========================== #
class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 8))
sns.barplot(x=list(test_generator.class_indices.keys()), y=class_accuracy)
plt.xticks(rotation=90)
plt.title('Class-wise Accuracy')
plt.ylabel('Accuracy')
plt.savefig('class_accuracy.png')
plt.close()

print("? Evaluation completed. Plots and reports saved.")