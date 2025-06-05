
# Fairplay Moment Detection in Cricket Matches

## Overview
This project focuses on detecting fairplay moments in cricket matches using computer vision and machine learning. Moments like handshakes, hugs, and sportsmanlike gestures are identified to aid sponsor placement during meaningful interactions.

## Dataset
- 10 Handshake images
- 11 Hug images
- 10 Lace-tying/helping gestures
- 21 Normal match moments (non-fairplay)
Each frame was manually labeled and pose keypoints were extracted using YOLOv8-pose.

## Approach
1. **Pose Detection**: YOLOv8-pose model extracts 17 human body keypoints per frame.
2. **Feature Engineering**: Keypoints (x, y) are flattened into 34-length vectors.
3. **Classification**: RandomForestClassifier predicts whether a frame contains a fairplay moment.

## Tools & Libraries
- YOLOv8 (Ultralytics)
- OpenCV
- Scikit-learn
- Pandas, NumPy
- Seaborn, Matplotlib

## Results
- **Accuracy**: 81%
- **Precision (Fairplay)**: 84%
- **Recall (Fairplay)**: 89%

## Files
- `Data Frames/` – Folder containing extracted frames used in the dataset  
- `Confusion Matrix.png` – Visualization of model prediction vs actual  
- `Fairplay_Detection_Project_Documentation.pdf` – Formal project documentation  
- `Manual Labelling.ipynb` – Notebook used to label frames manually  
- `manual_labels.csv` – CSV of manually labeled fairplay/not fairplay frames  
- `pose_dataset.csv` – Feature dataset with keypoints and labels  
- `train_test model.ipynb` – Notebook for training and evaluating the ML model  
- `yolov8n-pose.pt` – Pretrained YOLOv8-pose model used for pose extraction 

---

