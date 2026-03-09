# 🌐 Web Dashboard Setup & Usage

## 📦 Installation

### Step 1: Install Flask

```powershell
python -m pip install Flask>=2.3.0
```

Or install all packages again:

```powershell
python -m pip install -r requirements.txt
```

---

## 🚀 Running the Web Dashboard

### Option 1: Using Launcher Script (Easiest)

```powershell
cd "C:\Users\JAANI BEE\OneDrive\Desktop\hackathon\malaria_detection"
python run_dashboard.py
```

This will:
1. Check all dependencies
2. Start the web server
3. Open your browser automatically

---

### Option 2: Direct Run

```powershell
cd "C:\Users\JAANI BEE\OneDrive\Desktop\hackathon\malaria_detection"
python app.py
```

Then open: **http://127.0.0.1:5000**

---

## 📋 Complete Workflow

### Step 1: Train the Model

```powershell
python main.py
```

This creates:
- `models/best_model.h5` - Trained model
- `results/training_history.png` - Training curves
- `results/confusion_matrix.png` - Performance matrix
- `results/predictions_report.txt` - Detailed report

---

### Step 2: View Results in Dashboard

```powershell
python run_dashboard.py
```

Or directly:

```powershell
python app.py
```

Your browser will open automatically showing:
- ✓ Training results and plots
- ✓ Confusion matrix
- ✓ Full report
- ✓ System information

---

### Step 3: Make Predictions

In the dashboard:
1. Go to **"Make Prediction"** section
2. Click **"Upload"** or drag-drop a blood cell image
3. See diagnosis and confidence score instantly

---

## 🎯 Dashboard Features

### 📊 Results Section
- View all training plots
- See confusion matrix
- Read full report

### 🔬 Prediction Section
- Upload blood cell images
- Real-time diagnosis
- Confidence scores
- Visual feedback

### ℹ️ System Information
- TensorFlow version
- OpenCV version
- NumPy version
- Model status

---

## 🔧 Troubleshooting

### Issue: "Flask is not installed"
```powershell
python -m pip install Flask
```

### Issue: "Port 5000 already in use"
Edit `app.py` line 115:
```python
app.run(debug=False, port=5001, use_reloader=False)  # Change 5000 to 5001
```

### Issue: "Browser doesn't open"
Manually go to: **http://127.0.0.1:5000**

### Issue: "Model not found"
First train the model:
```powershell
python main.py
```

---

## 📁 Project Structure

```
malaria_detection/
├── app.py                  # Flask web server
├── run_dashboard.py        # Launcher script
├── main.py                # Training script
├── prediction.py          # Prediction module
├── requirements.txt       # Dependencies
│
├── templates/
│   └── index.html        # Dashboard UI
│
├── models/
│   └── best_model.h5     # Trained model
│
└── results/
    ├── training_history.png
    ├── confusion_matrix.png
    └── predictions_report.txt
```

---

## 📝 Step-by-Step Summary

```powershell
# 1. Navigate to project
cd "C:\Users\JAANI BEE\OneDrive\Desktop\hackathon\malaria_detection"

# 2. Install dependencies (if not done)
python -m pip install -r requirements.txt

# 3. Train the model (first time only)
python main.py

# 4. Start the web dashboard
python run_dashboard.py

# 5. Browser opens automatically at http://127.0.0.1:5000
```

---

## 🎨 Dashboard Preview

The web dashboard includes:

**Home Page:**
- System status indicator
- Quick links to all features

**Results Tab:**
- Training history plots
- Confusion matrix visualization
- Complete performance report

**Prediction Tab:**
- Drag-and-drop file upload
- Real-time prediction
- Confidence visualization

**System Info Tab:**
- Installed package versions
- Model status
- System health

---

## 💡 Tips

1. **First Time?** Run `python main.py` first to train
2. **Quick Test?** Use `python test_system.py` to verify setup
3. **Make Predictions?** Use the web dashboard for easy uploads
4. **View Results?** All outputs saved in `results/` folder

---

## 🆘 Help

**Error during training?**
→ Check `main.py` output for specific issues

**Dashboard won't open?**
→ Manually visit http://127.0.0.1:5000

**Model predictions wrong?**
→ Train with more images using `python main.py`

---

## 📞 Commands Quick Reference

```powershell
# Train model
python main.py

# Open dashboard
python run_dashboard.py

# Or direct run
python app.py

# Test setup
python test_system.py

# View results
explorer results
```

---

**Enjoy your Malaria Detection Dashboard! 🎉**
