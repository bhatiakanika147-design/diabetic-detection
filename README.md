# diabetic-detection
Diabetic Detection using Neural Network
📌 Project Description

This project predicts whether a person is diabetic or not using a Machine Learning model based on a Neural Network.
It uses medical data such as glucose level, blood pressure, insulin, etc. to make predictions.

📊 Dataset
PIMA Indians Diabetes Dataset
Contains 8 input features and 1 output (diabetic / non-diabetic)
⚙️ Technologies Used
Python
NumPy
Scikit-learn
Keras (Deep Learning)
🧠 Model Details
Neural Network (Sequential Model)
Input Layer: 8 features
Hidden Layers: Dense layers with ReLU activation
Output Layer: Sigmoid activation
Optimizer: Adam
Loss Function: Binary Crossentropy
🚀 Features
Data preprocessing (normalization using StandardScaler)
Train-test split for proper evaluation
Improved neural network architecture
Dropout used to reduce overfitting
Early stopping for better performance
📈 Accuracy
Training Accuracy: ~75%
Testing Accuracy: 78.35%

This performance is considered good for the PIMA Indians Diabetes Dataset using a Neural Network model.

📊 Evaluation Metrics
Accuracy: 78.35%
Confusion Matrix used for detailed analysis
Recall is emphasized to minimize false negatives (important in medical diagnosis)
