# Chinese Chess (Xiangqi) Board Detection and Piece Recognition

## Project Structure

```
chess_board_detection/
├── data/
│   ├── train_cnn2_image/          # Standard board images (01-018)
│   │   ├── crops/                 # Extracted patches (90 per board)
│   │   └── aligned_output/        # Aligned board images
│   ├── indoor/                    # Indoor scene images (5 scenes)
│   └── outdoor/                   # Outdoor scene images (5 scenes)
│
├── processed_boards/               # Aligned real-world boards
│   ├── indoor_1/ ... indoor_5/
│   └── outdoor_1/ ... outdoor_5/
│
├── model/                         # Trained models
│   ├── CNN_from_board_crops/     # Scene-specific model (91.32% acc)
│   ├── CNN_01_018_retrained/     # Standard board model (100% acc)
│   └── CNN_indoor_outdoor/       # Domain adapted model
│
├── correct_layout_csv/            # Ground truth labels
│   ├── Correct_Xiangqi_board_layout_01.csv
│   ├── Correct_Xiangqi_board_layout_012.csv
│   └── Correct_Xiangqi_board_layout_[scene].csv
│
├── recognition_results_*/         # Recognition outputs
│   └── [scene]/
│       ├── [image]_recognition.csv
│       └── [image]_comparison.png
│
│
├── xiangqi.ipynb                  # Main development notebook
├── process_indoor_outdoor.py      # Batch processing script
├── train_with_crops.py            # Standard board training
├── train_with_inoutdoor_crops.py  # Scene-specific training
├── recognize_all_boards.py        # Batch recognition script
├── nn_models.py                   # CNN model definitions
├── major_2.py                     # Image processing utilities
└── blank_temp.jpg                 # Blank board template
```


##  Dataset

### Standard Board Dataset
- **Images**: 18 boards (01-018)
- **Patches**: 1,620 (18 × 90)
- **Layouts**: 2 configurations (layout_01, layout_012)
- **Conditions**: Controlled lighting, frontal view

### Real-World Scene Dataset
- **Images**: 42 boards
  - Indoor: 20 images (5 scenes × 4 images)
  - Outdoor: 22 images (5 scenes × 4-6 images)
- **Patches**: 3,780 (42 × 90)
- **Challenges**: Variable lighting, shadows, background clutter

### Piece Classes (15 total)

| Category | Red (r) | Green (g) | Count per Board |
|----------|---------|-----------|-----------------|
| Bing (兵/卒) | bing_r | bing_g | 5 + 5 = 10 |
| Che (車/车) | che_r | che_g | 2 + 2 = 4 |
| Ma (馬/马) | ma_r | ma_g | 2 + 2 = 4 |
| Pao (砲/炮) | pao_r | pao_g | 2 + 2 = 4 |
| Xiang (相/象) | xiang_r | xiang_g | 2 + 2 = 4 |
| Shi (仕/士) | shi_r | shi_g | 2 + 2 = 4 |
| Jiang (帅/将) | jiang_r | jiang_g | 1 + 1 = 2 |
| Empty | - | empty | 58 |
