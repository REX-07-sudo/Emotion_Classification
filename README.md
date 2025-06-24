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






---






##  Accuracy Metrics (After dropping **surprised** class)



---



##  Streamlit Web App

- Upload any `.wav` file
- The app will extract features and predict the **emotion**
- Hosted locally or can be deployed via platforms like **Streamlit Cloud**

---


## Note

- Trained model uses only 7 emotions.

