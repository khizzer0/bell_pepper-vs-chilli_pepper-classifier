# 🌶️ Bell Pepper vs Chili Pepper Classifier (Deep Learning Project)

This project is a deep learning-based image classifier that distinguishes between **Bell Peppers** and **Chili Peppers** using transfer learning (ResNet18). It was developed as part of a final project for the Pepper Classification module on CV Studio.
---
## 📌 Problem Statement

Customers at Amarach Farms were frequently receiving the wrong kind of pepper due to visual similarity between **Bell Peppers** and **Chili Peppers**. This led to complaints — for example, one customer’s salad became inedible due to mistakenly receiving hot peppers.

To automate the sorting process, this project uses **Computer Vision (CNN)** to correctly classify pepper types.
---
## 📂 Project Structure

📁 pepper-classification-final-project
│
├── runs/
│ └── transfer-learning-for-pepper-classification/
│ └── bell-vs-chilli-classifier/ # Trained models and results
│
├── BellPepper1.jpeg → BellPepper4.jpeg # Sample test images
├── ChiliPepper1.jpeg → ChiliPepper4.jpeg # Sample test images
├── model.pt # Trained PyTorch model
├── pepper_classifier.ipynb # Main training + inference notebook
└── README.md # Project documentation

---

## 🧠 Model Details

- **Model**: ResNet18 (pretrained)
- **Classifier Head**: Final FC layer replaced with 2-output layer (Bell & Chili)
- **Framework**: PyTorch
- **Device**: GPU/CPU (Colab Compatible)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: SGD with Momentum
- **Scheduler**: Cyclic Learning Rate (triangular2)

---

## 📊 Training Details

- **Epochs**: 10  
- **Training Set**: 100 images per class  
- **Validation Accuracy**: ~95%  
- **Augmentations**: Resize, Normalize

---

## 📈 Results

Images were successfully predicted and labeled as:

- ✅ BellPepper1.jpeg → **Bellpepper**
- ✅ ChiliPepper2.jpeg → **Chilipepper**
- ✅ Mixed peppers classified correctly.

All test predictions were visualized with matplotlib in the notebook.

---

## 🧪 Testing Code Sample

```python
image = Image.open("ChiliPepper1.jpeg").convert("RGB")
x = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    z = model(x)
    _, yhat = torch.max(z.data, 1)

prediction = "Bellpepper" if yhat.item() == 0 else "Chilipepper"
🚀 Deployment
Model was deployed via IBM Code Engine and tested using browser-based classification tool built on CV Studio.

📦 How to Use
Clone this repo:

git clone https://github.com/your-username/pepper-classifier.git
Open pepper_classifier.ipynb in Google Colab.

Run all cells to train or test the model.

Replace image paths with your own test images to predict.

📷 Sample Screenshot
(Include screenshots of Bell Pepper & Chili Pepper classification here)

👤 Author
Khizzer ul Islam
BS Artificial Intelligence | Final  Project
Email: khizzerjadoon890@gmail.com

🏁 Credits
CV Studio - IBM Skills Network

PyTorch

IBM Cloud Code Engine

📃 License
This project is for educational purposes only and released under the MIT License.
