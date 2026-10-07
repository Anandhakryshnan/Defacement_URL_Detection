# 🌐 Defaced URL Detector 🛡️

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/UI-Cyberpunk%20Glassmorphism-00ff66.svg)]()

**Defaced URL Detector** is an advanced machine learning web application built with Flask. It instantly analyzes raw URLs to detect malicious intent (defacement, phishing, or malware) using a fully integrated Random Forest Classifier.

Equipped with a stunning, highly responsive **Cyberpunk/Hacker aesthetic** frontend, the app visually reacts to threats in real-time.

---

## ✨ Key Features

- **🧠 Machine Learning Engine**: Powered by a robust `RandomForestClassifier` trained on a rich dataset of URLs.
- **⚙️ 25 Unique Feature Extractions**: The backend algorithm breaks down the URL to analyze entropy, path-to-length ratios, delimiters, top-level domains, and specific symbol occurrences to accurately flag anomalies.
- **⚡ Instant Predictions**: The AI model is trained during server initialization, resulting in instantaneous zero-latency predictions for end users.
- **🎨 Reactive Cyberpunk UI**: Features CRT scanlines (optional), dynamic particle data-networks, neon-glowing active states, and aggressive danger-pulsing when a threat is found. Fully responsive for mobile devices.
- **🛡️ Crash-Proof Backend**: Automatically sanitizes input (e.g., prepending missing protocols) and safely catches mathematical extraction errors.

---

## 🚀 Live Demo

*(Once deployed to Render, place your live application link here!)*  
`https://defacement-url-detection.onrender.com/`

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Anandhakryshnan/Defacement_URL_Detection.git
cd Defacement_URL_Detection
```

2. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python app.py
```

4. **Access the Web UI:**  
Open your browser and navigate to `http://localhost:5000`

---

## 🧠 How the AI Works

The system does not rely on simple blacklists. Instead, it mathematically analyzes the structure of the URL to detect patterns commonly used by hackers.

When a URL is submitted, the backend script evaluates **25 structural features**, including:
- **Lexical Lengths:** Overall URL length, domain length, filename length.
- **Entropy Analytics:** Shannon entropy of the URL and Domain to detect randomly generated/obfuscated strings.
- **Symbol Densities:** Frequency counts of delimiters (`-`, `?`, `=`, `@`, `~`, etc.) commonly abused in defacement attacks.
- **Path Token Ratios:** The ratio of the path length compared to the domain length.

These features are fed into the **Random Forest** algorithm, which mathematically predicts the classification (`benign` or `defacement`).

---

## 📁 Repository Structure

- `app.py`: The core Flask server and feature extraction algorithms.
- `random forest.py`: The standalone script used to evaluate the model's accuracy, precision, and f1-scores.
- `enhanced_feature_set.csv`: The dataset used to train the model on server boot.
- `templates/index.html`: The HTML structure for the Web UI.
- `static/styles.css`: The Cyberpunk CSS design system.
- `static/app.js` & `particles.js`: Logic for the interactive background network effects.

---

## 🤝 Contributing
Contributions are welcome! If you have ideas to optimize the feature extraction or improve the UI, feel free to open a Pull Request.

## 📝 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
