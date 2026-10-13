# 🏡 California House Price Prediction

A Machine Learning project that predicts California house prices using housing data and regression techniques. The model is integrated with a Flask web application and deployed on Render.

## 🚀 Live Demo

[**Try the House Price Predictor**](https://california-house-price-prediction-model-jlk6.onrender.com/)

## 📌 Project Overview

This project uses machine learning to estimate house prices based on housing features such as median income, house age, average rooms, population, location, and ocean proximity.

## ✨ Features

- Predict California house prices using a trained ML model.
- Data preprocessing and feature transformation.
- Regression model evaluation.
- User-friendly web interface built with Flask.
- Online deployment using Render.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- HTML and CSS
- Gunicorn
- Render

## 📊 Model Performance

- **R² Score:** 0.8291 (82.91%)

The R² score indicates how well the model explains the variation in house prices on the evaluated dataset.

## 📂 Project Structure

```text
California-House-Price-Prediction/
├── data/
├── model/
├── static/
├── templates/
├── main.py
├── main_california_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Run Locally

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd California-House-Price-Prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows:

   ```bash
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask application:

   ```bash
   python main.py
   ```

5. Open the local URL shown in the terminal, usually `http://127.0.0.1:5000`.

## 🎯 Objective

To apply machine learning concepts to a real-world regression problem and make house price predictions accessible through a web application.

## 👨‍💻 Author

**Sachin Kumar**

- GitHub: [Sachinkr606](https://github.com/Sachinkr606)
- Portfolio: [Visit My Portfolio](https://Sachinkr606.github.io/)

---

*Built as a Machine Learning project to practice data preprocessing, regression, model evaluation, and web deployment.*
