import tensorflow as tf
import tensorflow_hub as hub

detector = hub.load("https://tfhub.dev/tensorflow/face_detector/1")

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])  # Resize based on the model's requirement
    img = img / 255.0  # Normalize pixel values
    return img

img_tensor = preprocess_image('test.jpg')

detections = detector(img_tensor)
# Process detections to extract bounding box coordinates


