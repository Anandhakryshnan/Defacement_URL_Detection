# Defaced URL Detector 💻🔍

This is a simple web application built with Flask that detects defacement URLs using machine learning. It takes a URL as input and predicts whether it is a defacement URL or not.

## Installation 🚀

1. Clone the repository:

```bash
git clone https://github.com/Anandhakryshnan/Defacement_URL_Detection
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage 🎯

1. Run the Flask application:

```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`.

3. Enter a URL in the provided input field and click "Detect".

4. The application will predict whether the URL is a defacement URL or not and display the result.

## How it Works ⚙️

- The application extracts various features from the input URL, such as URL length, domain length, number of digits in the query string, etc.
- These features are then used as input to a Random Forest Classifier machine learning model.
- The model predicts whether the URL is a defacement URL based on the extracted features.
- The prediction is displayed to the user on the web interface.

## Files 📁

- `app.py`: Contains the Flask application code.
- `index.html`: HTML template for the web interface.
- `styles.css`: CSS file for styling the web interface.
- `particles.js`: JavaScript library for the background particle animation.
- `app.js`: JavaScript file for handling form submission and displaying prediction results.
- `random_forest_model.pkl`: Pre-trained Random Forest Classifier model (not included in this repository).

## Contributing 🤝

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License 📝

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
