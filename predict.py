import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json
from PIL import Image
import warnings
import logging
import tf_keras

# Suppress warnings and limit logging output
warnings.filterwarnings('ignore')
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (224, 224))
    return image.numpy()

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)
    predictions = model.predict(processed_test_image)
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_k_probabilities = predictions[0][top_k_indices]
    top_k_classes = [str(i) for i in top_k_indices]
    return top_k_probabilities, top_k_classes

def create_percentage_bar(percentage):
    """
    Generate a text-based progress bar to represent the given percentage.
    """
    return "#" * int(percentage)

def main(args):
    """
    Main function to load the model, make predictions, and display results.
    """
    # Load the pre-trained model
    model = tf_keras.models.load_model(
        args.model_path, custom_objects={'KerasLayer': hub.KerasLayer}
    )
    
    # Predict the top K classes
    probabilities, classes = predict(args.image_path, model, args.top_k)

    # Map class indices to names if a category mapping file is provided
    if args.category_names:
        with open(args.category_names, 'r') as file:
            label_map = json.load(file)
        labels = [label_map.get(cls, f"Class {cls}") for cls in classes]
    else:
        labels = classes

    # Display the predictions
    print(f"Predictions for image: {args.image_path}")
    for idx, (probability, label) in enumerate(zip(probabilities, labels), start=1):
        percentage = probability * 100
        bar = create_percentage_bar(percentage)
        print(f"{idx}: {label} with probability {percentage:.2f}%")
        print(f"{bar}\n")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=(
            "Flower Image Classifier Application.\n"
            "Example: predict.py [image.jpg] [model.h5] [--top_k 3] [--category_names map.json]"
        )
    )
    parser.add_argument("image_path", help="Path to the flower image to be classified.")
    parser.add_argument("model_path", help="Path to the pre-trained .h5 model file.")
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top classes to display. Default is 5."
    )
    parser.add_argument(
        "--category_names",
        help="Path to JSON file mapping class indices to names."
    )

    args = parser.parse_args()
    main(args)
