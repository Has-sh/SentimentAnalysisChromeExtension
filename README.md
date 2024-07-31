# Sentiment Analysis Chrome Extension

This repository contains a Chrome extension for sentiment analysis, using a Django backend to perform the analysis. The extension allows users to input text and receive sentiment analysis results directly in their browser.

## Project Structure

- **`requirements.txt`**: Lists the Python dependencies required for the Django backend.
- **`sentimentAnalysis/`**: Contains the Django project for sentiment analysis, including `manage.py` for managing the Django application.
- **`Chrome Extension/`**: Contains the unpacked Chrome extension files for manual loading.
- **`sentiment_analysis.py`**: Python script for training the sentiment analysis model.

## Setup and Installation

### Django Backend

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Has-sh/SentimentAnalysisChromeExtension.git
    cd SentimentAnalysisChromeExtension
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Navigate to the Django Project Directory:**

    ```bash
    cd sentimentAnalysis
    ```

5. **Apply Migrations:**

    ```bash
    python manage.py migrate
    ```

6. **Run the Django Development Server:**

    ```bash
    python manage.py runserver
    ```

   The Django server will be running on `http://127.0.0.1:8000/`.

### Chrome Extension

1. **Open Chrome and Navigate to `chrome://extensions/`.**

2. **Enable "Developer mode"** using the toggle switch in the top-right corner.

3. **Click "Load unpacked"** and select the `Chrome Extension` directory.

4. **Test the Extension**: The extension should now appear in your list of extensions and be available for use.

## Code Overview

### `sentiment.py`

This script handles sentiment prediction using a pre-trained TensorFlow model. The `predict_sentiment` function tokenizes and pads input text, then predicts and returns the sentiment label.

### `views.py`

This Django view handles POST requests to analyze text sentiment. It uses the `predict_sentiment` function to get sentiment results and returns them in JSON format.

### `manifest.json`

Defines the Chrome extension's metadata and permissions. It includes background and popup configuration.

### `background.js`

Handles messages from the popup, performs sentiment analysis by sending a request to the Django backend, and then sends the result back to the popup.

### `popup.js`

Manages user interactions in the Chrome extension popup. It sends text for analysis and displays the result when received.

### `popup.html`

Provides the user interface for the Chrome extension popup. It includes a form for text input and a section to display the sentiment analysis result.

## Usage

1. **Input Text in the Extension Popup**: Open the extension popup, enter text into the textarea, and click "Analyze."

2. **View Sentiment Result**: The result will be displayed in the popup.

## Notes

- Ensure that the Django server is running for the extension to communicate with the backend.
- The TensorFlow model should be placed in the `static/file` directory of the Django project for predictions to work.
- If you encounter CORS issues, ensure that your Django server is configured to handle CORS requests in `settings.py`.
