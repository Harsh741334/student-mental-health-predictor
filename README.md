# 🧠 Student Mental Health Prediction App

A machine learning-powered web application that predicts depression risk in students based on various lifestyle and academic factors.

## Features

- **Interactive Web Interface**: Built with Streamlit for easy data input
- **Deep Learning Model**: Keras neural network trained on student mental health data
- **Real-time Predictions**: Get instant depression risk assessment
- **Multiple Input Factors**: Considers academic, lifestyle, and personal factors

## Live Demo

[View Live App](https://studenthealthpridiction.streamlit.app/) *(Update with your deployed URL)*

## Input Features

The model considers the following factors:

| Category | Features |
|----------|----------|
| **Demographics** | Gender, Age, City |
| **Academic** | Profession, Degree, CGPA, Academic Pressure, Study Satisfaction |
| **Work** | Work Pressure, Job Satisfaction, Work/Study Hours |
| **Lifestyle** | Sleep Duration, Dietary Habits |
| **Mental Health** | Suicidal Thoughts History, Family History of Mental Illness |
| **Financial** | Financial Stress |

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Model**: Neural Network (Dense layers with Dropout)

##  Project Structure

```
├── MAIN.py                      # Streamlit application
├── mental_health_model.keras    # Trained Keras model
├── preprocessor.pkl             # Data preprocessor (StandardScaler + OneHotEncoder)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
```

##  Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Harsh741334/student-mental-health-predictor.git
   cd student-mental-health-predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run MAIN.py
   ```

5. **Open in browser**
   ```
   https://studenthealthpridiction.streamlit.app/
   ```

## Deployment

### Streamlit Community Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file as `MAIN.py`
5. Deploy!

### Other Platforms

- **Render**: Set start command to `streamlit run MAIN.py --server.port $PORT --server.address 0.0.0.0`
- **Heroku**: Add `Procfile` with `web: streamlit run MAIN.py`

##  Model Performance

- **Architecture**: 64 → 32 → 16 → 1 (Dense layers with ReLU activation)
- **Output**: Sigmoid activation for binary classification
- **Training**: Early stopping with patience=6

## Disclaimer

This application is for **educational purposes only** and should not be used as a substitute for professional mental health diagnosis or treatment. If you or someone you know is struggling with mental health issues, please seek help from qualified healthcare professionals.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you have any questions or need help, please open an issue on GitHub.

---

Made with ❤️ for student mental health awareness
