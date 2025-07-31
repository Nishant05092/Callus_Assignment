# Callus_Assignment
🔧 Project Overview
This project leverages the zero-shot learning capability of CLIP (Contrastive Language-Image Pretraining) with the ViT-B/32 architecture to classify images as either medical or non-medical. Instead of training a custom model, it uses natural language descriptions to guide classification decisions.

📌 Problem Statement
Automatically classify a given image (from file, URL, or PDF) into:

Medical Image: X-ray, MRI, CT scan, etc.

Non-Medical Image: Landscape, animal, human, object, etc.

🧠 Model Used
Model: OpenAI CLIP (ViT-B/32)

Type: Pretrained Transformer (Vision + Text)

Key Feature: Joint image-text embedding space enabling zero-shot classification.

🛠️ Approach & Reasoning
Why CLIP?

It understands natural language prompts.

Ideal for zero-shot tasks — no retraining or labeling needed.

Compatible with open-source implementations like openai/CLIP and laion/CLIP.

Process Flow:

Input is processed from 3 formats: image file, URL, or PDF.

Image is preprocessed using CLIP's required normalization and resizing.

Class descriptions (in natural language) are tokenized and encoded into embeddings.

The similarity between image and each text prompt is calculated using cosine similarity.

The class with the highest similarity is the predicted label.

Text Prompts:

"a medical image like an X-ray, MRI, or CT scan"

"a non-medical image like a landscape, animal, or object"

🧪 Evaluation
Category	Correct Predictions	Total	Accuracy
Medical	9	10	90%
Non-Medical	10	10	100%
Overall	19	20	95%

Note: Tested on a small manually labeled dataset.

🖼️ Input Handling
Local File: .jpg, .png, etc.

URL: Uses requests and PIL to load image from web.

PDFs: Uses PyMuPDF to extract the first page and convert it to an image.

⚙️ Performance & Efficiency
No training time: Zero-shot inference.

Inference time: <1 sec per image on GPU.

Model Size: ~400 MB.

PDF rendering: Lightweight and fast using fitz (PyMuPDF).

✅ Advantages
No need for dataset creation, labeling, or model training.

Works on real-world inputs without preprocessing.

Expandable to more classes by changing or adding descriptions.

Easy to deploy using a Flask or Streamlit web app.

⚠️ Limitations
Limited to the descriptive power of natural language prompts.

May misclassify ambiguous or mixed-content images.

Performance drops if input images are too noisy, blurred, or low resolution.

Currently binary classification (medical vs non-medical).

📦 Dependencies
torch, clip, Pillow, requests, PyMuPDF, torchvision
