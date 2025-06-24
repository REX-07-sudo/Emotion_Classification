# Emotion Classification from Audio using MFCC and LightGBM


This project focuses on classifying human emotions from audio recordings using MFCC (Mel Frequency Cepstral Coefficients) features and a LightGBM classifier. The system is trained on the RAVDESS dataset containing emotional speech and song audio files. A Streamlit-based web app allows users to upload `.wav` files and get real-time emotion predictions.

---

##  Pre-processing Methodology

1. **Dataset**:  
   - Combined **speech and song files** from RAVDESS .
   - Each audio filename encodes the emotion (01 to 08), mapped to:
     - `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

2. **Feature Extraction**:
   - Used `librosa` to extract **40 MFCC features** from each audio.
   - Mean of MFCCs taken to generate a fixed-length vector for every file.

3. **Filtering**:
   - Dropped sample with label: `surprised` due to consistently poor performance (<75% precision).

4. **Encoding and Scaling**:
   - Applied `LabelEncoder` on emotion labels.
   - Scaled features using `StandardScaler`.

5. **Train-Test Split**:
   - 80-20 split with **stratified sampling** to maintain class balance.

6. **SMOTE**:
   - Applied **SMOTE oversampling** only to training data to handle class imbalance.

---

##  Model Pipeline

- **Model Used**: LightGBM (`LGBMClassifier`)
- **Hyperparameter Tuning**: Done via **Optuna**, optimizing `f1-score` (macro average)  
- **Key Steps**:
  - Feature extraction  Label encoding  Scaling  Train/Test split
  - SMOTE on train set
  - Optuna hyperparameter search
  - Final model training with best parameters

---

## Accuracy Metrics (Before Drop)

**Macro F1-Score:** `0.7738`  
**Overall Accuracy:** `78%`

#### Classification Report (8 Classes)

| Emotion    | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Angry      | 0.89      | 0.85   | 0.87     | 75      |
| Calm       | 0.81      | 0.87   | 0.84     | 75      |
| Disgust    | 0.75      | 0.77   | 0.76     | 39      |
| Fearful    | 0.75      | 0.83   | 0.78     | 75      |
| Happy      | 0.86      | 0.76   | 0.81     | 75      |
| Neutral    | 0.76      | 0.82   | 0.78     | 38      |
| Sad        | 0.76      | 0.67   | 0.71     | 75      |
| Surprised  | 0.60      | 0.67   | 0.63     | 39      |

#### Averages

- **Macro Avg**: Precision: `0.77` | Recall: `0.78` | F1-Score: `0.77`
- **Weighted Avg**: Precision: `0.79` | Recall: `0.78` | F1-Score: `0.78`

---

##  Accuracy Metrics (After dropping **surprised** class)

**Macro F1-Score:** `0.80`  
**Overall Accuracy:** `80%`

#### Classification Report (7 Classes)

| Emotion    | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Angry      | 0.86      | 0.87   | 0.86     | 75      |
| Calm       | 0.80      | 0.91   | 0.85     | 75      |
| Disgust    | 0.73      | 0.77   | 0.75     | 39      |
| Fearful    | 0.77      | 0.80   | 0.78     | 75      |
| Happy      | 0.87      | 0.83   | 0.85     | 75      |
| Neutral    | 0.76      | 0.82   | 0.78     | 38      |
| Sad        | 0.78      | 0.63   | 0.70     | 75      |

#### Averages

- **Macro Avg**: Precision: `0.80` | Recall: `0.80` | F1-Score: `0.80`
- **Weighted Avg**: Precision: `0.80` | Recall: `0.80` | F1-Score: `0.80`

---

##  Streamlit Web App

- Upload any `.wav` file
- The app will extract features and predict the **emotion**
- Hosted locally or can be deployed via platforms like **Streamlit Cloud**

---

## Note:

- Trained model uses only 7 emotions.
- Both confusion matrices are in .ipynb file.

