from inference.image_inference import image_inference
import numpy as np
from PIL import Image
import io

# Create a dummy black image
img = Image.new("RGB", (224, 224), color="black")
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format="PNG")

preds = image_inference.predict(img_byte_arr.getvalue())
print("Predictions:", preds)