"""
Train using crops from processed_boards
Training set: indoor_1, indoor_2, indoor_3, outdoor_1, outdoor_2, outdoor_3
Test set: indoor_4, indoor_5, outdoor_4, outdoor_5
"""

import os
import sys
import io
import cv2
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from nn_models import CNN
from major_2 import resize_and_pad

# Set output encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# Path configuration
PROCESSED_BOARDS_DIR = Path("./processed_boards")
CORRECT_CSV_DIR = Path("./correct_layout_csv")
MODEL_OUTPUT_DIR = Path("./model/CNN_from_board_crops")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training and test scenes
TRAIN_SCENES = ['indoor_1', 'indoor_2', 'indoor_3', 'outdoor_1', 'outdoor_2','outdoor_3',]
TEST_SCENES = [ 'indoor_4', 'indoor_5', 'outdoor_4', 'outdoor_5']

# Define class mapping
piece_names = ["bing", "che", "ma", "pao", "xiang", "shi", "jiang"]
colors = ["r", "g"]
valid_classes = [f"{piece}_{color}" for piece in piece_names for color in colors]
valid_classes.append("empty")

class_to_id = {class_name: idx for idx, class_name in enumerate(valid_classes)}
id_to_class = {idx: class_name for idx, class_name in enumerate(valid_classes)}

print("="*80)
print("Class Mapping:")
print("="*80)
for idx, class_name in id_to_class.items():
    print(f"  {idx:2d}: {class_name}")

def load_crops_from_scene(scene_name, target_size=(250, 250)):
    """
    Load all crops from a scene
    
    Args:
        scene_name: Scene name (e.g. 'indoor_1')
        target_size: Target image size
    
    Returns:
        images: Image array
        labels: Label array
        count: Number of images loaded
    """
    scene_dir = PROCESSED_BOARDS_DIR / scene_name
    crops_dir = scene_dir / "crops"
    
    if not crops_dir.exists():
        print(f"Skip {scene_name} (crops folder does not exist)")
        return [], [], 0
    
    # Find corresponding CSV file
    csv_path = CORRECT_CSV_DIR / f"Correct_Xiangqi_board_layout_{scene_name}.csv"
    if not csv_path.exists():
        print(f"Skip {scene_name} (CSV file does not exist)")
        return [], [], 0
    
    # Read labels
    df_layout = pd.read_csv(csv_path)
    
    images = []
    labels = []
    
    # Iterate through all image subfolders in this scene
    for board_dir in sorted(crops_dir.iterdir()):
        if not board_dir.is_dir():
            continue
        
        board_name = board_dir.name
        
        # Read crops from 90 positions
        for _, row in df_layout.iterrows():
            r = row['row']
            c = row['col']
            label = row['label']
            
            crop_filename = f"r{r}_c{c}.jpg"
            crop_path = board_dir / crop_filename
            
            if not crop_path.exists():
                continue
            
            # Read image
            img = cv2.imread(str(crop_path))
            if img is None:
                continue
            
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize
            img = resize_and_pad(img, target_size)
            
            images.append(img)
            labels.append(class_to_id[label])
    
    return images, labels, len(images)

def load_all_data(scenes, dataset_name):
    """Load data from multiple scenes"""
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} Data")
    print("="*80)
    
    all_images = []
    all_labels = []
    
    for scene_name in scenes:
        print(f"\n  Scene: {scene_name}")
        images, labels, count = load_crops_from_scene(scene_name)
        
        if count > 0:
            all_images.extend(images)
            all_labels.extend(labels)
            print(f"Loaded {count} images")
            
            # Count class distribution
            unique, counts = np.unique(labels, return_counts=True)
            label_dist = {id_to_class[label_id]: count for label_id, count in zip(unique, counts)}
            for label_name in sorted(label_dist.keys()):
                print(f"      - {label_name:15s}: {label_dist[label_name]:4d} samples")
    
    if len(all_images) == 0:
        print(f"\n No images loaded!")
        return None, None
    
    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    print(f"\n  Total samples: {len(X)}")
    print(f"  Data shape: {X.shape}")
    
    # Overall class distribution
    print(f"\n  Overall class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"    {id_to_class[class_id]:15s}: {count:5d} samples")
    
    return X, y

def build_and_train_model(X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    """Build and train CNN model"""
    print("\n" + "="*80)
    print("Building and Training CNN Model")
    print("="*80)
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Convert to one-hot encoding
    num_classes = len(class_to_id)
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Build model
    model_params = {
        'params': {
            'cnn': [
                {'filters': 8, 'kernel_size': (7, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
                {'filters': 32, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
                {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
            ],
            'pool': [
                {'pool_size': (4, 4)},
                {'pool_size': (4, 4)},
                {'pool_size': (4, 4)},
            ],
            'output_dense': {'units': num_classes},
            'output_act': {'activation': 'softmax'},
        }
    }
    
    model = CNN(**model_params)
    model.build(input_shape=[(None, 250, 250, 3)])
    model.call(np.zeros((1, 250, 250, 3), dtype=np.float32))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Train model
    print(f"\nStarting training ({epochs} epochs)...")
    
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_test, y_test_onehot),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Save model
    weights_path = MODEL_OUTPUT_DIR / 'CNN_board_crops.weights.h5'
    model.save_weights(str(weights_path))
    print(f"\nModel weights saved: {weights_path}")
    
    # Save model architecture
    arch_path = MODEL_OUTPUT_DIR / 'model_architecture.pkl'
    with open(arch_path, 'wb') as f:
        pickle.dump(model_params, f)
    print(f"Model architecture saved: {arch_path}")
    
    # Save class mapping
    mapping_path = MODEL_OUTPUT_DIR / 'class_mapping.pkl'
    with open(mapping_path, 'wb') as f:
        pickle.dump({'class_to_id': class_to_id, 'id_to_class': id_to_class}, f)
    print(f"Class mapping saved: {mapping_path}")
    
    # Save training history
    history_path = MODEL_OUTPUT_DIR / 'training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved: {history_path}")
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate model"""
    print("\n" + "="*80)
    print("Model Evaluation")
    print("="*80)
    
    # Predict
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    # Get classes that actually exist in test set
    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names_actual = [id_to_class[i] for i in unique_labels]
    
    print(classification_report(
        y_test, y_pred,
        labels=unique_labels,
        target_names=target_names_actual,
        digits=4,
        zero_division=0
    ))
    
    # Confusion matrix analysis
    print("\nConfusion Matrix (showing main errors):")
    cm = confusion_matrix(y_test, y_pred)
    
    # Find most confused class pairs
    confusion_pairs = []
    for i in range(len(id_to_class)):
        for j in range(len(id_to_class)):
            if i != j and cm[i][j] > 0:
                confusion_pairs.append((id_to_class[i], id_to_class[j], cm[i][j]))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nMost confused class pairs (Top 10):")
    print(f"{'True Label':<15} {'Pred Label':<15} {'Error Count'}")
    print("-"*50)
    for true_label, pred_label, count in confusion_pairs[:10]:
        print(f"{true_label:<15} {pred_label:<15} {count:>5}")
    
    return accuracy

def main():
    print("\n" + "="*80)
    print("Training CNN Model Using Real Board Crops")
    print("="*80)
    
    print(f"\nTraining scenes: {', '.join(TRAIN_SCENES)}")
    print(f"Test scenes: {', '.join(TEST_SCENES)}")
    
    # Load training data
    X_train, y_train = load_all_data(TRAIN_SCENES, "Training")
    if X_train is None:
        print("\n Failed to load training data, exiting")
        return
    
    # Load test data
    X_test, y_test = load_all_data(TEST_SCENES, "Test")
    if X_test is None:
        print("\n Failed to load test data, exiting")
        return
    
    # Build and train model
    model, history = build_and_train_model(X_train, y_train, X_test, y_test, epochs=30, batch_size=32)
    
    # Evaluate model
    test_accuracy = evaluate_model(model, X_test, y_test)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    print(f"\nTraining scenes: {', '.join(TRAIN_SCENES)}")
    print(f"Test scenes: {', '.join(TEST_SCENES)}")

if __name__ == "__main__":
    main()

