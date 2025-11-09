"""
Retrain CNN model using actual board crop data
Training set: 01-016
Test set: 017, 018
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nn_models import CNN
import tensorflow as tf
from tensorflow import keras
from major_2 import resize_and_pad

# Define class mapping
piece_names = ["bing", "che", "ma", "pao", "xiang", "shi", "jiang"]
colors = ["r", "g"]
valid_classes = [f"{piece}_{color}" for piece in piece_names for color in colors]
valid_classes.append("empty")  # Note: this is "empty", not "empty_0"

# Create class name to ID mapping
class_to_id = {class_name: idx for idx, class_name in enumerate(valid_classes)}
id_to_class = {idx: class_name for idx, class_name in enumerate(valid_classes)}

print("="*80)
print("Class Mapping:")
print("="*80)
for idx, cls in id_to_class.items():
    print(f"  {idx:2d}: {cls}")
print()

def load_crops_dataset(board_ids, crops_dir='./data/train_cnn2_image/crops', target_size=(256, 256), csv_mapping=None):
    """
    Load dataset from cropped images and CSV label files
    
    Args:
        board_ids: List of board IDs, e.g. ['01', '02', ...]
        crops_dir: Root directory of cropped images
        target_size: Target image size
        csv_mapping: Dictionary specifying CSV filename for each board_id, uses default naming if None
    
    Returns:
        images: numpy array (N, 256, 256, 3)
        labels: numpy array (N,) class IDs
        filenames: List of filenames
    """
    images = []
    labels = []
    filenames = []
    
    for board_id in board_ids:
        print(f"Loading board {board_id}...")
        
        # Determine which CSV file to use
        if csv_mapping and board_id in csv_mapping:
            csv_file = csv_mapping[board_id]
        else:
            csv_file = f'./data/train_cnn2_image/{board_id}_recognition.csv'
        
        if not os.path.exists(csv_file):
            print(f"  Warning: CSV file {csv_file} not found, skipping...")
            continue
        
        df = pd.read_csv(csv_file)
        
        # Read image from each position
        for _, row in df.iterrows():
            r, c, label = row['row'], row['col'], row['label']
            filename = f"{board_id}-{r}-{c}.jpg"
            filepath = os.path.join(crops_dir, board_id, filename)
            
            if not os.path.exists(filepath):
                print(f"  Warning: {filepath} not found")
                continue
            
            # Read and preprocess image
            img = cv2.imread(filepath)
            if img is None:
                print(f"  Warning: Unable to read {filepath}")
                continue
            
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize and pad
            img = resize_and_pad(img, target_size)
            
            # Get class ID
            if label not in class_to_id:
                print(f"  Warning: Unknown label '{label}' in {filename}")
                continue
            
            images.append(img)
            labels.append(class_to_id[label])
            filenames.append(filename)
        
        print(f"  Loaded {len([f for f in filenames if f.startswith(board_id)])} images")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    return images, labels, filenames

# Load training data (01-016)
# 01-09: use Correct_Xiangqi_board_layout_01.csv
# 010-016: use Correct_Xiangqi_board_layout_012.csv
print("="*80)
print("Loading Training Data (boards 01-016)")
print("="*80)

# 01-09 use layout 1
train_board_ids_layout1 = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
# 010-016 use layout 2
train_board_ids_layout2 = ['010', '011', '012', '013', '014', '015', '016']

# Merge all training board IDs
train_board_ids = train_board_ids_layout1 + train_board_ids_layout2

# Create CSV mapping
train_csv_mapping = {}
for board_id in train_board_ids_layout1:
    train_csv_mapping[board_id] = './correct_layout_csv/Correct_Xiangqi_board_layout_01.csv'
for board_id in train_board_ids_layout2:
    train_csv_mapping[board_id] = './correct_layout_csv/Correct_Xiangqi_board_layout_012.csv'

print(f"  Layout 1 (01-09): {len(train_board_ids_layout1)} boards")
print(f"  Layout 2 (010-016): {len(train_board_ids_layout2)} boards")
print(f"  Total training boards: {len(train_board_ids)}")

X_train, y_train, train_files = load_crops_dataset(train_board_ids, csv_mapping=train_csv_mapping)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# Count training data class distribution
print("\nTraining data class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for cls_id, count in zip(unique, counts):
    print(f"  {id_to_class[cls_id]:12s} (ID={cls_id:2d}): {count:3d} samples")

# Load test data (017, 018)
# Note: 017, 018 use layout 2 (Correct_Xiangqi_board_layout_012.csv)
print("\n" + "="*80)
print("Loading Test Data (boards 017, 018)")
print("="*80)
test_board_ids = ['017', '018']

# Create CSV mapping: 017, 018 both use layout 2 annotations
test_csv_mapping = {board_id: './correct_layout_csv/Correct_Xiangqi_board_layout_012.csv' for board_id in test_board_ids}

print(f"  Test boards: {len(test_board_ids)}")

X_test, y_test, test_files = load_crops_dataset(test_board_ids, csv_mapping=test_csv_mapping)

print(f"\nTest data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Count test data class distribution
print("\nTest data class distribution:")
unique, counts = np.unique(y_test, return_counts=True)
for cls_id, count in zip(unique, counts):
    print(f"  {id_to_class[cls_id]:12s} (ID={cls_id:2d}): {count:3d} samples")

# Convert labels to one-hot encoding
num_classes = len(valid_classes)
y_train_onehot = tf.one_hot(y_train, num_classes).numpy()
y_test_onehot = tf.one_hot(y_test, num_classes).numpy()

# Build CNN model (using CNN 2 architecture)
print("\n" + "="*80)
print("Building CNN Model")
print("="*80)

model_params = {
    'params': {
        'cnn': [
            {'filters': 8, 'kernel_size': (7, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
            {'filters': 32, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
            {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
        ],
        'pool': [
            {'pool_size': (4, 4)},  # (64, 64, 8)
            {'pool_size': (4, 4)},  # (16, 16, 32)
            {'pool_size': (4, 4)},  # (4, 4, 64)
        ],
        'output_dense': {'units': num_classes},
        'output_act': {'activation': 'softmax'},
    }
}

model = CNN(**model_params)
model.build(input_shape=[(None,) + X_train.shape[1:]])
model.call(np.zeros((1,) + X_train.shape[1:]))

print("\nModel architecture:")
model.summary()

# Compile model
print("\nCompiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train model
print("\n" + "="*80)
print("Training Model")
print("="*80)

history = model.fit(
    X_train, y_train_onehot,
    epochs=50,  # Increased to 50 epochs due to increased data size
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate model
print("\n" + "="*80)
print("Evaluating Model on Test Set")
print("="*80)

test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Detailed prediction analysis
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

print("\n" + "="*80)
print("Classification Report")
print("="*80)
print(classification_report(y_test, y_pred, target_names=valid_classes, zero_division=0))

print("\n" + "="*80)
print("Confusion Matrix")
print("="*80)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Calculate per-class accuracy
print("\n" + "="*80)
print("Per-Class Accuracy")
print("="*80)
for cls_id in range(num_classes):
    cls_mask = (y_test == cls_id)
    if cls_mask.sum() > 0:
        cls_acc = (y_pred[cls_mask] == cls_id).sum() / cls_mask.sum()
        print(f"{id_to_class[cls_id]:12s} (ID={cls_id:2d}): {cls_acc*100:6.2f}% ({(y_pred[cls_mask] == cls_id).sum()}/{cls_mask.sum()})")

# Display some incorrect predictions
print("\n" + "="*80)
print("Sample Incorrect Predictions")
print("="*80)
errors = np.where(y_pred != y_test)[0]
print(f"Total errors: {len(errors)} / {len(y_test)}")

if len(errors) > 0:
    print("\nFirst 20 errors:")
    for i, idx in enumerate(errors[:20]):
        true_label = id_to_class[y_test[idx]]
        pred_label = id_to_class[y_pred[idx]]
        confidence = y_pred_prob[idx][y_pred[idx]]
        print(f"  {test_files[idx]:20s}: True={true_label:12s}, Pred={pred_label:12s}, Conf={confidence:.3f}")

# 保存模型
print("\n" + "="*80)
print("Saving Model")
print("="*80)

output_dir = './model/CNN_01_018_retrained'
os.makedirs(output_dir, exist_ok=True)

# Save model weights (not the entire model)
weights_path = os.path.join(output_dir, 'CNN_crops_retrained.weights.h5')
model.save_weights(weights_path)
print(f"Model weights saved to: {weights_path}")

# Save model architecture
import pickle
model_arch_path = os.path.join(output_dir, 'model_architecture.pkl')
with open(model_arch_path, 'wb') as f:
    pickle.dump(model_params, f)
print(f"Model architecture saved to: {model_arch_path}")

# Save training history
history_path = os.path.join(output_dir, 'training_history.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)
print(f"Training history saved to: {history_path}")

# Save class mapping
mapping_path = os.path.join(output_dir, 'class_mapping.pkl')
with open(mapping_path, 'wb') as f:
    pickle.dump({'class_to_id': class_to_id, 'id_to_class': id_to_class}, f)
print(f"Class mapping saved to: {mapping_path}")

print("\n" + "="*80)
print("Training Complete!")
print("="*80)
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Model saved to: {output_dir}")

