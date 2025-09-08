# 🚗 Car Price Predictor

A Machine Learning project that predicts the **market value of a used car** based on user inputs such as brand, manufacturing year, engine power, kilometers driven, fuel type, and transmission.  

This project is built with **Python, Flask, and Scikit-learn**, and is deployed for live use.

---

## 🔗 Live Demo
Try the project here: [Car Price Predictor Live](https://car-price-predictor-production-b243.up.railway.app/)  

---

## 📌 Features
- User-friendly web interface built with **Flask**.  
- Select from popular car brands or enter a **custom brand**.  
- Predict car prices using a trained **Machine Learning regression model**.  
- Responsive design (works on desktop and mobile).  
- Deployed online for easy access.  

---

## ⚙️ Tech Stack
- **Frontend:** HTML, CSS, JavaScript, Bootstrap  
- **Backend:** Flask (Python)  
- **Machine Learning:** Scikit-learn, Pandas, NumPy  
- **Deployment:** Railway.app  

---

## 📊 Dataset
The dataset contains details of cars such as:
- Brand  
- Year of manufacture  
- Engine capacity (CC)  
- Power (BHP)  
- Mileage (KMPL)  
- Transmission type  
- Fuel type  
- Kilometers driven  
- Owner type  

The target variable is the **selling price**.  
(Data source: [Kaggle Car Price Dataset](https://www.kaggle.com/datasets) or your custom dataset)

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/esfanmerchant/car-price-predictor.git
   cd car-price-predictor

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
4. **Run the Flask app**
   ```bash
   python app.py
   
5. **Open your browser at:**
   ```bash
  http://127.0.0.1:5000/

## 📷 Screenshots

(Add some screenshots of your web app UI here, e.g. homepage, form, and prediction result)

---

## 📚 Learnings
- Practiced **data preprocessing and cleaning**.  
- Built and trained **machine learning regression models**.  
- Integrated ML model with a **Flask web application**.  
- Deployed project on **Railway.app**.  

---

## 💡 Future Improvements
- Add support for more car brands and features.  
- Use advanced models (**Random Forest, XGBoost**, etc.) for better accuracy.  
- Enhance UI/UX with charts and better visualization.  
- Build an **API endpoint** for programmatic predictions.  

---

## 👨‍💻 Author
**Esfan Merchant**  
Frontend Developer & Data Science Enthusiast  
🔗 [GitHub Profile](https://github.com/esfanmerchant)  

