import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

def show_example_photos(image_urls):

    def fetch_image(url):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image.rotate(270, expand=True)  # Rotate 90 degrees left (counterclockwise)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, url in enumerate(image_urls):
        ax = axes[i // 3, i % 3]
        image = fetch_image(url)
        ax.imshow(image)
        ax.axis('off')

    plt.show()

image_urls = [
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_0801.jpeg', 
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_0802.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_0803.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_0804.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_0805.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_0806.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_0807.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_0808.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_0809.jpeg',
    ]

import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO

# load clip model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image.rotate(270, expand=True).convert("RGB")

# Define ranking criteria as text prompts
text_prompts = ["high quality", "clear picture", "faces are easy to see", "great composition"]

# Load images
images = [load_image_from_url(url) for url in image_urls]

# Process images and text prompts
inputs = processor(text=text_prompts, images=images, return_tensors="pt", padding=True)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Get the similarity scores from CLIP
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # similarity scores between images and prompts

# Sum scores across prompts for each image to get an overall ranking
scores = logits_per_image.sum(dim=1).cpu().numpy()

# Sort and get indices of top-ranked images
sorted_images = [img for _, img in sorted(zip(scores, images), reverse=True)]
top_indices = scores.argsort()[::-1]  # Descending order
top_images = [(images[i],top_indices[i]) for i in top_indices[:5]]  # Top 5 images

# Display the top-ranked images with PLT
plt.figure(figsize=(10, 10))
for idx, img in enumerate(top_images):
    plt.subplot(1, len(top_images), idx + 1)
    plt.imshow(img[0])
    plt.axis('off')
    plt.title(f"Rank {idx + 1}")
    plt.text(0, -40, img[1], ha='center', va='top')
plt.show()