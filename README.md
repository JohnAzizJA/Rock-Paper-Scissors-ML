# ✊✋✌️ Rock-Paper-Scissors Classifier

A machine learning project that classifies hand gestures (rock, paper, scissors) using **MediaPipe hand landmark detection** and traditional ML algorithms.

## 🎯 Project Overview

This project uses a hybrid approach combining:
- **AI-based feature extraction** - MediaPipe extracts 21 hand landmarks (x, y, z coordinates)
- **Geometric feature engineering** - Calculates distances between fingers and angles
- **Traditional ML classifiers** - Random Forest, XGBoost, SVM, Neural Networks, etc.

### Why This Approach?

Instead of using traditional image processing (color histograms, edge detection), we use MediaPipe's pre-trained deep learning model to detect hand landmarks. This gives us:
- ✅ **Higher accuracy** - Focus on hand geometry, not background/lighting
- ✅ **Robustness** - Works across different skin tones, backgrounds
- ✅ **Interpretable features** - Distances and angles have clear meaning

## 📊 Features Extracted

### 1. Hand Landmarks (63 features)
- 21 hand joints with (x, y, z) coordinates
- MediaPipe automatically detects wrist, thumb, fingers, etc.

### 2. Relative Distances (15 features)
- Distance from each fingertip to wrist
- Distance between fingertips (crucial for scissors detection)
- Distance from fingertips to palm center

### 3. Finger Angles (5 features)
- Bend angle for each finger (thumb, index, middle, ring, pinky)
- Helps distinguish extended fingers (paper) from folded (rock)

**Total: 83 features per image**

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**

2. **Create virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install mediapipe opencv-python scikit-learn xgboost pandas matplotlib seaborn pillow scipy numpy
```

## 📁 Project Structure

```
Rock-Paper-Scissors-ML/
├── data/
│   ├── training/           # Training images
│   │   ├── rock/
│   │   ├── paper/
│   │   └── scissors/
│   ├── testing/            # Testing images (optional)
│   └── predict/            # Images for prediction
├── src/
│   ├── feature_extraction.py  # Extract hand landmarks
│   ├── train.py               # Train ML models
│   └── predict.py             # Predict on new images
├── features/                   # Extracted features (auto-generated)
├── models/                     # Trained models (auto-generated)
├── results/                    # Confusion matrices, plots (auto-generated)
└── README.md
```

## 🎓 Usage

### Step 1: Prepare Your Dataset

Organize your images in the following structure:
```
data/training/
├── rock/       # Put rock hand images here
├── paper/      # Put paper hand images here
└── scissors/   # Put scissors hand images here
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`

### Step 2: Extract Features

```bash
python src/feature_extraction.py
```

This will:
- Load all images from `data/training/`
- Detect hands using MediaPipe
- Extract 83 features per image
- Split into training (80%) and testing (20%) sets
- Save features to `features/train_features.pkl` and `features/test_features.pkl`

**Output:**
```
======================================================================
HAND LANDMARK FEATURE EXTRACTION - MediaPipe
======================================================================
Reading from: data/training

Processing rock...
  Found 500 images
  Successfully processed: 495/500 images
...
Total samples: 1485
Feature dimensions: 83
```

### Step 3: Train Models

```bash
python src/train.py
```

This will:
- Train 7 different classifiers (Decision Tree, Random Forest, XGBoost, KNN, SVM RBF, SVM Linear, Neural Network)
- Evaluate each model on test set
- Generate confusion matrices and comparison plots
- Save the best model to `models/best_model.pkl`

**Output:**
```
======================================================================
TRAINING AND EVALUATION
======================================================================
Training: Random Forest
Training Time: 2.145s
Test Accuracy: 98.50%
Precision: 0.9853
Recall: 0.9850
F1-Score: 0.9851
```

**Generated files:**
- `models/best_model.pkl` - Best performing model
- `results/accuracy_comparison.png` - Accuracy comparison chart
- `results/cm_*.png` - Confusion matrix for each classifier
- `results/results.csv` - Detailed metrics table

### Step 4: Predict on New Images

```bash
python src/predict.py
```

**Option 1: Single Image Prediction**
```
Select option (1/2): 1
Enter image path: data/predict/rock.jpg

🎯 PREDICTION RESULT
✊ Predicted: ROCK

📊 Confidence Scores:
   ✊ Rock     ██████████████████████████████ 98.0%
   ✋ Paper    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.5%
   ✌️ Scissors ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.5%
```

**Option 2: Batch Testing on Folder**
```
Select option (1/2): 2
Enter folder path: data/predict

📂 TESTING FOLDER: predict
📊 Found 15 images

[1/15] ✅ Paper_1.jpg    → ✋ PAPER    (99.5%) [True: paper]
[2/15] ✅ Rock_1.jpg     → ✊ ROCK     (98.0%) [True: rock]
...

======================================================================
📊 RESULTS SUMMARY
======================================================================
✅ Accuracy: 100.0% (13/13 correct)

📈 Predictions by class:
   ✊ Rock    : 6 images
   ✋ Paper   : 4 images
   ✌️ Scissors: 5 images
```

## 🧠 How It Works

### 1. Hand Landmark Detection (MediaPipe)
MediaPipe detects 21 key points on the hand:
- **Wrist** (landmark 0)
- **Thumb** (landmarks 1-4)
- **Index finger** (landmarks 5-8)
- **Middle finger** (landmarks 9-12)
- **Ring finger** (landmarks 13-16)
- **Pinky** (landmarks 17-20)

### 2. Feature Engineering

**For Rock (Fist):**
- Short distances from fingertips to palm
- High finger bend angles
- All fingers close together

**For Paper (Open Hand):**
- Long distances from fingertips to palm
- Low finger bend angles (extended)
- Fingers spread apart

**For Scissors (Peace Sign):**
- Large distance between index and middle fingertips
- Index and middle fingers extended
- Ring and pinky fingers folded

### 3. Classification

Multiple classifiers are trained and compared:
- **Random Forest** - Ensemble of decision trees
- **XGBoost** - Gradient boosting (often best performer)
- **SVM (RBF & Linear)** - Support Vector Machines
- **Neural Network** - Multi-layer perceptron
- **KNN** - K-Nearest Neighbors
- **Decision Tree** - Single decision tree

The best model (highest test accuracy) is automatically selected and saved.

## 📈 Expected Performance

With a good dataset (500+ images per class), you can expect:
- **Accuracy**: 95-99%
- **Training time**: 1-5 seconds (depending on dataset size)
- **Prediction time**: <100ms per image

## 🛠️ Troubleshooting

### "No hand detected"
- Make sure hand is clearly visible in the image
- Hand should be centered and not cut off
- Good lighting helps
- Try different hand positions

### "MediaPipe not available"
```bash
pip uninstall mediapipe
pip install mediapipe
```

### Low accuracy
- Increase dataset size (500+ images per class recommended)
- Ensure diverse hand positions in training data
- Check if hands are properly detected in training images
- Try different classifiers (XGBoost often works best)

## 📊 Model Comparison

Typical performance on a well-balanced dataset:

| Model | Accuracy | Training Time | Strengths |
|-------|----------|---------------|-----------|
| **Random Forest** | 96-98% | Fast | Robust, minimal tuning |
| **XGBoost** | 97-99% | Medium | Best accuracy, handles imbalance |
| **Neural Network** | 95-98% | Slow | Complex patterns |
| **SVM (RBF)** | 94-97% | Medium | Good with scaled features |
| **SVM (Linear)** | 92-95% | Fast | Fast, linearly separable |
| **KNN** | 90-94% | Very Fast | Simple, distance-based |
| **Decision Tree** | 85-90% | Very Fast | Interpretable |

## 🎯 Use Cases

- **Educational tool** - Learn hand gesture recognition
- **Game controller** - Control games with hand gestures
- **Accessibility** - Alternative input method
- **Research** - Baseline for gesture recognition studies

## 🔧 Customization

### Add More Gestures

1. Create new folders in `data/training/` (e.g., `thumbs_up/`)
2. Update `CLASS_MAP` in `feature_extraction.py`:
```python
CLASS_MAP = {'rock': 0, 'paper': 1, 'scissors': 2, 'thumbs_up': 3}
```
3. Re-run feature extraction and training

### Adjust Train/Test Split

In `feature_extraction.py`:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # 70/30 split
)
```

### Tune Hyperparameters

In `train.py`, modify classifier parameters:
```python
'Random Forest': RandomForestClassifier(
    n_estimators=300,  # More trees
    max_depth=25,      # Deeper trees
    random_state=42
)
```

## 📚 Technologies Used

- **MediaPipe** - Hand landmark detection
- **OpenCV** - Image processing
- **scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **NumPy** - Numerical operations

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request



## 🙏 Acknowledgments

- **MediaPipe** by Google for hand detection
- Rock-Paper-Scissors datasets from Kaggle
- scikit-learn community for ML algorithms


