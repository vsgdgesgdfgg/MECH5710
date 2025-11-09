"""
Use trained CNN2 model to recognize all boards in processed_boards
Generate recognition results and visual comparison for each image
"""

import os
import sys
import io
import cv2
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from nn_models import CNN
from major_2 import resize_and_pad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set output encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Path configuration
PROCESSED_BOARDS = Path("./processed_boards")
MODEL_DIR = Path("./model/CNN_from_board_crops")
OUTPUT_DIR = Path("./recognition_results_board_crops")
CORRECT_CSV_DIR = Path("./correct_layout_csv")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Define color and abbreviation mappings
piece_colors = {
    'empty': 'white',
    'che_g': 'green', 'che_r': 'red',
    'ma_g': 'lightgreen', 'ma_r': 'lightcoral',
    'pao_g': 'darkgreen', 'pao_r': 'darkred',
    'xiang_g': 'lime', 'xiang_r': 'orange',
    'shi_g': 'forestgreen', 'shi_r': 'tomato',
    'jiang_g': 'darkslategray', 'jiang_r': 'crimson',
    'bing_g': 'yellowgreen', 'bing_r': 'salmon',
}

piece_abbr = {
    'empty': '□',
    'che_g': '车', 'che_r': '車',
    'ma_g': '马', 'ma_r': '馬',
    'pao_g': '炮', 'pao_r': '砲',
    'xiang_g': '象', 'xiang_r': '相',
    'shi_g': '士', 'shi_r': '仕',
    'jiang_g': '将', 'jiang_r': '帅',
    'bing_g': '卒', 'bing_r': '兵',
}

def load_model():
    """Load trained CNN model"""
    print("="*80)
    print("Loading model...")
    print("="*80)
    
    # Load class mapping
    mapping_path = MODEL_DIR / 'class_mapping.pkl'
    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)
    class_to_id = mappings['class_to_id']
    id_to_class = mappings['id_to_class']
    
    # Load model architecture
    arch_path = MODEL_DIR / 'model_architecture.pkl'
    with open(arch_path, 'rb') as f:
        model_params = pickle.load(f)
    
    # Build model
    model = CNN(**model_params)
    model.build(input_shape=[(None, 250, 250, 3)])
    model.call(np.zeros((1, 250, 250, 3), dtype=np.float32))
    
    # Load weights
    weights_files = list(MODEL_DIR.glob('*.weights.h5'))
    if not weights_files:
        raise FileNotFoundError(f"No weights file found in {MODEL_DIR}")
    weights_path = weights_files[0]
    model.load_weights(str(weights_path))
    
    print(f"Model loaded successfully: {MODEL_DIR}")
    print(f"  Number of classes: {len(class_to_id)}")
    
    return model, id_to_class

def load_board_crops(crops_dir, board_name, target_size=(250, 250)):
    """Load all crops from a board"""
    board_crops_dir = crops_dir / board_name
    
    if not board_crops_dir.exists():
        return None, None
    
    images = []
    positions = []
    
    # Read all positions in order
    for r in range(1, 10):  # 9 rows
        for c in range(1, 11):  # 10 columns
            img_path = board_crops_dir / f"r{r}_c{c}.jpg"
            
            if not img_path.exists():
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize and pad
            img = resize_and_pad(img, target_size)
            
            images.append(img)
            positions.append((r, c))
    
    if len(images) == 0:
        return None, None
    
    return np.array(images, dtype=np.float32), positions

def predict_board(model, images, id_to_class):
    """Predict all positions on a board"""
    y_prob = model.predict(images, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    
    predictions = []
    for pred_id, prob in zip(y_pred, y_prob):
        pred_label = id_to_class[pred_id]
        confidence = prob[pred_id]
        predictions.append({
            'label': pred_label,
            'confidence': confidence
        })
    
    return predictions

def draw_board(ax, df, title, label_col):
    """Draw chess board"""
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 9.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 10))
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for _, row in df.iterrows():
        x, y = row['col'], row['row']
        label = row[label_col]
        color = piece_colors.get(label, 'gray')
        abbr = piece_abbr.get(label, '?')
        
        circle = plt.Circle((x, y), 0.35, color=color, ec='black', linewidth=2, zorder=2)
        ax.add_patch(circle)
        
        text_color = 'black' if label == 'empty' else 'white'
        ax.text(x, y, abbr, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=text_color, zorder=3)

def draw_diff_board(ax, df):
    """Draw difference comparison"""
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 9.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 10))
    ax.set_title('Difference Comparison (Green=Correct, Red=Error)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    correct_count = 0
    error_count = 0
    
    for _, row in df.iterrows():
        x, y = row['col'], row['row']
        pred = row['label_pred']
        correct = row['label_correct']
        is_correct = pred == correct
        
        if is_correct:
            correct_count += 1
            color = 'lightgreen'
            marker = '○'
            ec = 'green'
        else:
            error_count += 1
            color = 'lightcoral'
            marker = '×'
            ec = 'red'
        
        rect = mpatches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                                  facecolor=color, edgecolor=ec, linewidth=3, zorder=2)
        ax.add_patch(rect)
        
        ax.text(x, y, marker, ha='center', va='center', 
                fontsize=20, fontweight='bold', color=ec, zorder=3)
    
    accuracy = correct_count / (correct_count + error_count) * 100
    ax.text(5.5, 10.2, f'Correct: {correct_count}  Errors: {error_count}  Accuracy: {accuracy:.2f}%', 
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def visualize_comparison(df_merged, board_name, output_path):
    """Generate visual comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    draw_board(axes[0], df_merged, 'Ground Truth', 'label_correct')
    draw_board(axes[1], df_merged, f'{board_name} Recognition Result', 'label_pred')
    draw_diff_board(axes[2], df_merged)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def recognize_single_board(model, id_to_class, scene_dir, board_name, csv_path):
    """Recognize a single board"""
    crops_dir = scene_dir / "crops"
    
    # Load crops
    images, positions = load_board_crops(crops_dir, board_name)
    
    if images is None:
        return None
    
    # Predict
    predictions = predict_board(model, images, id_to_class)
    
    # Create prediction result DataFrame
    df_pred = pd.DataFrame([
        {'row': r, 'col': c, 'label': pred['label'], 'confidence': pred['confidence']}
        for (r, c), pred in zip(positions, predictions)
    ])
    
    # Read ground truth
    if not csv_path.exists():
        print(f"  Annotation file not found: {csv_path}")
        return None
    
    df_correct = pd.read_csv(csv_path)
    
    # Merge
    df_merged = df_pred.merge(df_correct, on=['row', 'col'], suffixes=('_pred', '_correct'))
    
    # Calculate accuracy
    correct_count = (df_merged['label_pred'] == df_merged['label_correct']).sum()
    total_count = len(df_merged)
    accuracy = correct_count / total_count * 100
    
    return {
        'board_name': board_name,
        'df_merged': df_merged,
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total_count
    }

def main():
    print("\n" + "="*80)
    print("Recognizing All Boards Using CNN2 Model")
    print("="*80)
    
    # Load model
    model, id_to_class = load_model()
    
    # Statistics
    all_results = []
    scene_stats = {}
    
    # Iterate through all scenes
    for scene_dir in sorted(PROCESSED_BOARDS.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_name = scene_dir.name
        print(f"\n{'='*80}")
        print(f"Scene: {scene_name}")
        print("="*80)
        
        # Determine CSV file
        csv_name = f"Correct_Xiangqi_board_layout_{scene_name}.csv"
        csv_path = CORRECT_CSV_DIR / csv_name
        
        if not csv_path.exists():
            print(f"  Skip (annotation file not found: {csv_name})")
            continue
        
        # Create output directory
        scene_output_dir = OUTPUT_DIR / scene_name
        scene_output_dir.mkdir(exist_ok=True)
        
        scene_results = []
        
        # Get all boards
        crops_dir = scene_dir / "crops"
        if not crops_dir.exists():
            continue
        
        board_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir()])
        
        for board_dir in board_dirs:
            board_name = board_dir.name
            print(f"\n  Processing: {board_name}")
            
            # Recognize
            result = recognize_single_board(model, id_to_class, scene_dir, board_name, csv_path)
            
            if result is None:
                print(f"    Recognition failed")
                continue
            
            # Save result CSV
            csv_output = scene_output_dir / f"{board_name}_recognition.csv"
            result['df_merged'][['row', 'col', 'label_pred', 'label_correct']].to_csv(
                csv_output, index=False
            )
            
            # Generate visualization
            vis_output = scene_output_dir / f"{board_name}_comparison.png"
            visualize_comparison(result['df_merged'], board_name, vis_output)
            
            print(f"    Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
            print(f"    - Result: {csv_output}")
            print(f"    - Visualization: {vis_output}")
            
            scene_results.append(result)
            all_results.append(result)
        
        # Scene statistics
        if scene_results:
            scene_total = sum(r['total'] for r in scene_results)
            scene_correct = sum(r['correct'] for r in scene_results)
            scene_acc = scene_correct / scene_total * 100
            
            scene_stats[scene_name] = {
                'boards': len(scene_results),
                'accuracy': scene_acc,
                'correct': scene_correct,
                'total': scene_total
            }
            
            print(f"\n  【{scene_name} Summary】")
            print(f"    Number of images: {len(scene_results)}")
            print(f"    Overall accuracy: {scene_acc:.2f}% ({scene_correct}/{scene_total})")
    
    # Overall statistics
    print("\n" + "="*80)
    print("Overall Recognition Statistics")
    print("="*80)
    
    if all_results:
        total_boards = len(all_results)
        total_positions = sum(r['total'] for r in all_results)
        total_correct = sum(r['correct'] for r in all_results)
        overall_accuracy = total_correct / total_positions * 100
        
        print(f"\nTotal images: {total_boards}")
        print(f"Total positions: {total_positions}")
        print(f"Total correct: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.2f}%")
        
        print(f"\n{'Scene':<15} {'Images':<8} {'Accuracy':<12} {'Correct/Total':<15}")
        print("-" * 55)
        for scene_name, stats in scene_stats.items():
            print(f"{scene_name:<15} {stats['boards']:<8} {stats['accuracy']:>6.2f}%     {stats['correct']:>4}/{stats['total']:<4}")
        
        # Save overall statistics
        summary = {
            'total_boards': total_boards,
            'total_positions': total_positions,
            'total_correct': total_correct,
            'overall_accuracy': overall_accuracy,
            'scene_stats': scene_stats,
            'board_results': [
                {
                    'board': r['board_name'],
                    'accuracy': r['accuracy'],
                    'correct': r['correct'],
                    'total': r['total']
                }
                for r in all_results
            ]
        }
        
        summary_path = OUTPUT_DIR / 'overall_summary.pkl'
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        
        print(f"\nOverall statistics saved: {summary_path}")
        print(f"All results saved to: {OUTPUT_DIR}")
    
    print("\n" + "="*80)
    print("Recognition Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

