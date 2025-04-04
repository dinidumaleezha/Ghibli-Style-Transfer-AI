import tensorflow as tf  # type: ignore
import tensorflow_hub as hub  # type: ignore
from PIL import Image, ImageFilter  # type: ignore
import numpy as np  # type: ignore

# Load pre-trained style transfer model
hub_module = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

def load_image(image_path, resize_to=None):
    """
    Load and preprocess an image.
    If resize_to is provided, image will be resized to that size.
    """
    img = Image.open(image_path).convert('RGB')

    if resize_to:
        img = img.resize(resize_to, resample=Image.BICUBIC)  # High-quality resize

    img_np = np.array(img).astype(np.float32) / 255.0  # Normalize
    img_np = img_np[tf.newaxis, ...]  # Add batch dim
    return img_np, img.size  # return image and its size

def apply_style_transfer(content_image_path, style_image_path, output_path="results/output_ghibli_style.png"):
    """
    Apply Ghibli-style transfer to the content image.
    """
    # Load original content image for size reference
    content_image = Image.open(content_image_path).convert("RGB")
    original_size = content_image.size  # (width, height)

    # Load images
    content_tensor, _ = load_image(content_image_path)
    style_tensor, _ = load_image(style_image_path, resize_to=(256, 256))

    # Apply style transfer
    stylized_image = hub_module(tf.constant(content_tensor), tf.constant(style_tensor))[0]

    # Convert and upscale to original size
    stylized_image = np.squeeze(stylized_image, axis=0)
    stylized_image = (stylized_image * 255).clip(0, 255).astype(np.uint8)
    result_img = Image.fromarray(stylized_image)

    # Resize to original with LANCZOS (best quality)
    result_img = result_img.resize(original_size, resample=Image.LANCZOS)

    # Optional: apply slight sharpening for clarity
    result_img = result_img.filter(ImageFilter.SHARPEN)

    # Save as lossless PNG
    result_img.save(output_path, format="PNG", quality=100, optimize=True)
    print(f"✅ Stylized image saved: {output_path}")

    return result_img

def export_tflite_model(model_path="model/ghibli_style_model.tflite"):
    """
    Export TensorFlow Hub model as TFLite file.
    """
    @tf.function
    def model_fn(content_img, style_img):
        return hub_module(content_img, style_img)

    dummy_content = tf.zeros([1, 256, 256, 3], dtype=tf.float32)
    dummy_style = tf.zeros([1, 256, 256, 3], dtype=tf.float32)
    concrete_func = model_fn.get_concrete_function(dummy_content, dummy_style)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    print("✅ TFLite model saved:", model_path)

# Run
if __name__ == "__main__":
    content_image_path = "inputs/image/v1.jpg"
    style_image_path = "style_image/ghibli_style_reference.png"  # Use uploaded reference image

    apply_style_transfer(content_image_path, style_image_path)
    export_tflite_model()


