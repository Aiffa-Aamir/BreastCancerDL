🧠 Breast Cancer Detection Using Deep Learning

📌 Problem Statement

Breast cancer is one of the most common cancers globally. Early and accurate detection is critical to improving survival rates. Traditional diagnostic techniques are often time-consuming, subjective, and limited by human error. This project aims to automate and improve the accuracy of breast cancer diagnosis using a deep learning-based image classification pipeline.

📂 Dataset

We use the Breast Histopathology Images dataset, which includes high-resolution histology images of breast tissue. Each image is labeled as either:

Benign: Non-cancerous

Malignant: Cancerous

Dataset Characteristics:
RGB images of size 50x50 (resized during preprocessing)

Two classes: benign, malignant

Substantial class imbalance addressed using class weights

🧪 Methodology / Pipeline

This project follows a structured deep learning pipeline:

1. Data Loading and Preprocessing
   
Image files are read and resized to a standard 50x50x3 shape.

Labels are encoded into binary format.

Train/validation/test split is performed.

Class imbalance is addressed using class_weight.

2. Data Augmentation

Techniques such as rotation, flipping, and zooming are applied using ImageDataGenerator to increase data diversity and reduce overfitting.

3. Model Building
   
A custom CNN architecture is used:

Convolutional layers with ReLU activation

MaxPooling layers to reduce spatial dimensions

Dropout for regularization

Dense output layer with sigmoid activation

4. Model Training
   
Trained with binary cross-entropy loss

Adam optimizer

EarlyStopping and ReduceLROnPlateau callbacks used to optimize training

5. Evaluation
   
Model is evaluated on the test set using:

Accuracy

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

ROC curve and AUC score are also plotted to evaluate performance

⚙️ Requirements

Install the required dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
Minimal packages:

bash
Copy
Edit
tensorflow
numpy
pandas
matplotlib
scikit-learn

🚀 Results

Test Accuracy: ~90%

AUC Score: 0.96+

The model generalizes well and performs competitively with state-of-the-art methods.

🔍 Key Learnings

Handling class imbalance is crucial in medical imaging.

Data augmentation significantly improves generalization.

Simpler CNN architectures can still yield high performance with proper tuning.

📘 How to Use

Clone the repository.

Download and extract the dataset into the data/ directory.

Run the notebook: BreastCancerDL.ipynb.

