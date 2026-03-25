![Malaria Detection Header](file:///C:/Users/JAANI%20BEE/.gemini/antigravity/brain/6d158ae8-143f-47b3-835a-73a82c97ef44/malaria_detection_header_1774445458734.png)

# 🔬 Malaria Detection System

### **Automated Diagnosis using Deep Learning**
This system provides a high-accuracy solution for classifying blood cell images as **Parasitized** or **Uninfected**. Built with TensorFlow and MobileNetV2, it leverages transfer learning for rapid and reliable medical diagnosis.

---

## 🚀 Quick Run (3 Steps)

### 1. **Setup & Activate**
```powershell
cd malaria_detection
..\venv\Scripts\activate
```

### 2. **Train & Evaluate**
```powershell
python main.py
```

### 3. **Launch Dashboard**
```powershell
python run_dashboard.py
```
*The dashboard will automatically open at `http://127.0.0.1:5000`*

---

## 🛠️ Project Structure
- `📂 data/raw/`: Place your images here (`Parasitized/` & `Uninfected/`).
- `📂 models/`: Stores the trained `best_model.h5`.
- `📂 results/`: Contains training plots and prediction reports.
- `🐍 app.py`: Core Flask web application.

---

## 📤 GitHub Integration
To save your work or push changes:
```powershell
git add .
git commit -m "Update project models and results"
git push origin main
```

---
**Version:** 1.2 | **Status:** Ready for Deployment
