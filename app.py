import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random
import os
import io
import base64
from flask import Flask, render_template, request, jsonify

# --- Helper Functions for Data and Visualization ---
def preprocess_image(image):
    binary_image = (image > 128).astype(int)
    bipolar_vector = 2 * binary_image.flatten() - 1
    return bipolar_vector

def show_image(vector, ax, diff_vector=None):
    image = vector.reshape(28, 28)
    
    ax.imshow(image, cmap='gray', vmin=-1, vmax=1)
    
    if diff_vector is not None:
        diff_image = diff_vector.reshape(28, 28)
        highlight_mask = np.abs(diff_image) > 0
        
        normalized_image = (image + 1) / 2
        colored_image = np.stack([normalized_image] * 3, axis=-1)
        
        colored_image[highlight_mask] = [1, 0, 0] # R=1 (red), G=0, B=0
        
        ax.imshow(colored_image, vmin=0, vmax=1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, which='both', color='#444', linestyle='-', linewidth=0.5)
    ax.set_facecolor('none')
    ax.spines['left'].set_color('#c9d1d9')
    ax.spines['bottom'].set_color('#c9d1d9')
    ax.spines['right'].set_color('#c9d1d9')
    ax.spines['top'].set_color('#c9d1d9')
    ax.axis('off')

# --- Hopfield Network Class ---
class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def store_patterns(self, patterns):
        num_patterns = len(patterns)
        if num_patterns == 0:
            return
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= num_patterns

    def recall(self, initial_state, max_iterations=100):
        state = initial_state.copy()
        for _ in range(max_iterations):
            neuron_idx = random.randint(0, self.num_neurons - 1)
            weighted_sum = np.dot(self.weights[neuron_idx], state)
            new_state = 1 if weighted_sum >= 0 else -1
            if new_state == state[neuron_idx]:
                continue
            state[neuron_idx] = new_state
        return state

def recognize_digit(recalled_pattern, stored_patterns_dict):
    min_distance = float('inf')
    best_match_digit = None
    for digit, patterns in stored_patterns_dict.items():
        for stored_pattern in patterns:
            distance = np.sum(recalled_pattern != stored_pattern)
            if distance < min_distance:
                min_distance = distance
                best_match_digit = digit
    return best_match_digit

# --- Flask App Setup ---
app = Flask(__name__)

# Global variables to store the network and data
hn = None
stored_patterns_dict = {}
x_train, y_train = None, None

def initialize_network():
    global hn, stored_patterns_dict, x_train, y_train
    if hn is None:
        (x_train, y_train), (_, _) = mnist.load_data()
        
        patterns_to_store = 5
        for i in range(10):
            digit_images = x_train[y_train == i][:patterns_to_store]
            bipolar_patterns = [preprocess_image(img) for img in digit_images]
            stored_patterns_dict[i] = bipolar_patterns

        all_patterns_to_store = []
        for patterns in stored_patterns_dict.values():
            all_patterns_to_store.extend(patterns)

        num_neurons = all_patterns_to_store[0].shape[0]
        hn = HopfieldNetwork(num_neurons)
        hn.store_patterns(all_patterns_to_store)

@app.before_request
def ensure_network_initialized():
    initialize_network()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize_single', methods=['POST'])
def recognize_single():
    try:
        data = request.get_json()
        test_digit_idx = int(data['digit'])
        noise_level = float(data['noise'])

        clean_vector = random.choice(stored_patterns_dict[test_digit_idx])
        
        noisy_vector = clean_vector.copy()
        num_neurons = hn.num_neurons
        num_noisy_pixels = int(num_neurons * noise_level)
        noisy_indices = np.random.choice(num_neurons, num_noisy_pixels, replace=False)
        noisy_vector[noisy_indices] *= -1

        recalled_vector = hn.recall(noisy_vector)
        recognized_digit = recognize_digit(recalled_vector, stored_patterns_dict)

        buf = io.BytesIO()
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=100, facecolor='none', edgecolor='none')
        
        show_image(clean_vector, axs[0])
        show_image(noisy_vector, axs[1], diff_vector=noisy_vector - clean_vector)
        show_image(recalled_vector, axs[2], diff_vector=recalled_vector - noisy_vector)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(buf, format='png', transparent=True)
        plt.close(fig)
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        img_url = "data:image/png;base64," + img_base64

        return jsonify({'recognized_digit': recognized_digit, 'image_url': img_url})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recognize_all', methods=['POST'])
def recognize_all():
    try:
        data = request.get_json()
        noise_level = float(data['noise'])
        
        all_results = []
        for test_digit_idx in range(10):
            test_image_index = np.where(y_train == test_digit_idx)[0][0]
            clean_vector = preprocess_image(x_train[test_image_index])
            
            noisy_vector = clean_vector.copy()
            num_neurons = hn.num_neurons
            num_noisy_pixels = int(num_neurons * noise_level)
            noisy_indices = np.random.choice(num_neurons, num_noisy_pixels, replace=False)
            noisy_vector[noisy_indices] *= -1

            recalled_vector = hn.recall(noisy_vector)
            recognized_digit = recognize_digit(recalled_vector, stored_patterns_dict)

            buf = io.BytesIO()
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=100, facecolor='none', edgecolor='none')
            
            show_image(clean_vector, axs[0])
            show_image(noisy_vector, axs[1], diff_vector=noisy_vector - clean_vector)
            show_image(recalled_vector, axs[2], diff_vector=recalled_vector - noisy_vector)

            plt.tight_layout(pad=0.5)
            plt.savefig(buf, format='png', transparent=True)
            plt.close(fig)
            buf.seek(0)
            
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            img_url = "data:image/png;base64," + img_base64

            all_results.append({
                'original_digit': test_digit_idx,
                'recognized_digit': recognized_digit,
                'image_url': img_url
            })
            
        return jsonify(all_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    initialize_network()
    app.run(debug=True)
