# 🌾 Crop and Fertilizer Recommendation System

This web-based application provides intelligent recommendations for both **crops** and **fertilizers** based on soil and environmental parameters.

🔗 **Live App**: [https://crop-and-fertilizer-recommendation-zd8g.onrender.com/](https://crop-and-fertilizer-recommendation-zd8g.onrender.com/)

---

## 🚀 Features

- ✅ **Crop Recommendation** based on:
  - Nitrogen (N), Phosphorus (P), Potassium (K)
  - Temperature
  - Humidity
  - pH
  - Rainfall

- ✅ **Fertilizer Suggestion** based on:
  - Ideal vs actual soil nutrient levels
  - Logic-based comparison for actionable advice

- 🧠 Uses Machine Learning for crop classification
- 🌐 User-friendly web interface built with Flask

---

## 🧰 Tech Stack

- **Frontend**: HTML, CSS (via Flask templating)
- **Backend**: Python, Flask
- **ML Libraries**: scikit-learn, pandas, numpy
- **Deployment**: Render

---

## 📦 Requirements

- Python 3.7 or higher
- Flask
- scikit-learn
- pandas
- numpy

---

## 📊 How It Works

### Crop Recommendation:
- Trained ML model takes in environmental features and predicts the best crop to grow.

### Fertilizer Suggestion:
- Compares the input soil values with ideal nutrient ranges for the chosen crop.
- Returns a text-based recommendation using a dictionary logic.

---

## 🙏 Acknowledgments

- Dataset sources: Kaggle and open agricultural datasets
- Thanks to edunet Foundation.

---

🌱 Happy Farming!
