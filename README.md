# Hopfield Network for MNIST Digit Recognition

This is a web-based demonstration of a Hopfield network, a recurrent neural network that serves as a content-addressable memory system. The application uses Python with Flask and NumPy to visualize how the network recalls noisy handwritten digits from the MNIST dataset.

## Features

* **Memory Storage**: The network is trained on a set of clean, binarized handwritten digits (0-9).
* **Noise Introduction**: Users can add varying levels of noise to the digits.
* **Pattern Recall**: The network recalls the original, clean pattern from the noisy input.
* **Visual Demonstration**: The app plots the original, noisy, and recalled images, highlighting the pixels that were changed.

## How to Run

1.  **Clone the repository**:
    ```sh
    git clone [your-repository-url]
    cd [your-project-directory]
    ```
2.  **Install dependencies**:
    ```sh
    pip install Flask numpy matplotlib keras tensorflow
    ```
3.  **Start the server**:
    ```sh
    python app.py
    ```
4.  Open your browser and navigate to `http://127.0.0.1:5000`.
