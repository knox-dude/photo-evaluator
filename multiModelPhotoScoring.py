import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np
from dataclasses import dataclass
import cv2
from facenet_pytorch import MTCNN
import requests
from dotenv import load_dotenv
import os
import base64

@dataclass
class PhotoScore:
    technical_score: float
    composition_score: float
    expression_score: float
    overall_score: float
    explanation: str

class PhotoScoringSystem:
    def __init__(self):
        # Initialize vision-language model (e.g., BLIP-2)

        # uncomment when on home WIFI - big download

        # self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        # self.vlm = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        # Initialize face detection
        # self.face_detector = MTCNN(keep_all=True)
        
        # Technical quality assessment weights
        self.weights = {
            'sharpness': 0.3,
            'exposure': 0.2,
            'composition': 0.25,
            'expression': 0.25
        }

    def analyze_technical_quality(self, image):
        # Convert PIL to cv2
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Analyze sharpness
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500, 1.0)
        
        # Analyze exposure
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        brightness = hsv[:,:,2].mean()
        exposure_score = 1.0 - abs(brightness - 128) / 128
        
        # Rule of thirds analysis
        height, width = cv_image.shape[:2]
        third_h, third_w = height // 3, width // 3
        
        # Calculate points of interest using edge detection
        edges = cv2.Canny(cv_image, 100, 200)
        thirds_score = self._calculate_thirds_score(edges, third_h, third_w)
        
        return {
            'sharpness': sharpness_score,
            'exposure': exposure_score,
            'composition': thirds_score
        }

    def analyze_expressions(self, image):
        # Detect faces and get bounding boxes
        boxes, probs = self.face_detector.detect(image)
        
        if boxes is None:
            return 0.0
        
        # Analyze each face for expression quality
        expression_scores = []
        for box in boxes:
            face = image.crop((box[0], box[1], box[2], box[3]))
            score = self._analyze_single_expression(face)
            expression_scores.append(score)
            
        return np.mean(expression_scores) if expression_scores else 0.0

    def get_llm_analysis(self, image_url, api_key):
        """
        Use GPT-4 Vision or similar to get qualitative analysis
        """
        prompt = """
        Analyze this photo and rate it on a scale of 1-10 for:
        1. Technical quality (focus, lighting, exposure)
        2. Composition and framing
        3. Subject expression and emotion
        4. Overall appeal
        
        Provide a brief explanation for each score.
        """
        from openai import OpenAI

        client = OpenAI(
            organization=os.environ.get("OPENAI_ORG_ID"),
            project=os.environ.get("OPENAI_PROJECT_ID"),
            api_key=api_key
        )
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
            }
        ],
        )

        return response.choices[0].message.content
    
    def get_llm_analysis_multiple(self, image_urls, api_key):
        """
        Use GPT-4 Vision or similar to get qualitative analysis of multiple photos
        """
        prompt = f"""
        Analyze these {len(image_urls)} photos and give me the index of the best two photos in number form, eg. 'photo 1' and 'photo 2'.
        Provide a brief explanation on why you chose these as the best two.
        """

        from openai import OpenAI

        client = OpenAI(
            organization=os.environ.get("OPENAI_ORG_ID"),
            project=os.environ.get("OPENAI_PROJECT_ID"),
            api_key=api_key
        )
        content = [
            {"type": "text", "text": prompt},
            *self._generate_image_list(image_urls),
        ]
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": content,
            }
        ],
        )

        return response.choices[0].message.content

    def score_photo(self, image_path, api_key):
        """
        Combine all scoring methods for final analysis
        """
        image = Image.open(image_path)
        
        # Get technical scores
        tech_scores = self.analyze_technical_quality(image)
        
        # Get expression scores
        expr_score = self.analyze_expressions(image)
        
        # Get LLM analysis
        llm_scores = self.get_llm_analysis(image, api_key)
        
        # Combine scores with weights
        final_score = (
            tech_scores['sharpness'] * self.weights['sharpness'] +
            tech_scores['exposure'] * self.weights['exposure'] +
            tech_scores['composition'] * self.weights['composition'] +
            expr_score * self.weights['expression']
        )
        
        return PhotoScore(
            technical_score=np.mean([tech_scores['sharpness'], 
                                   tech_scores['exposure']]),
            composition_score=tech_scores['composition'],
            expression_score=expr_score,
            overall_score=final_score,
            explanation=llm_scores['explanation']
        )

    def _calculate_thirds_score(self, edges, third_h, third_w):
        # Implementation of rule of thirds scoring
        thirds_points = [
            (third_w, third_h), (2*third_w, third_h),
            (third_w, 2*third_h), (2*third_w, 2*third_h)
        ]
        
        score = 0
        for point in thirds_points:
            region = edges[point[1]-20:point[1]+20, 
                         point[0]-20:point[0]+20]
            score += np.sum(region) / 255
        
        return min(score / 1000, 1.0)

    # Function to encode the image
    def _base64_encode(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _analyze_single_expression(self, face_image):
        # Use BLIP-2 for expression analysis
        inputs = self.processor(face_image, return_tensors="pt")
        
        prompt = "Rate the quality of the facial expression in this photo on a scale of 0 to 1, where 1 is a perfect, natural expression."
        
        output = self.vlm.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            prompt=prompt
        )
        
        # Process output to get score
        # This is simplified - you'd need to parse the actual output
        return float(output[0].split()[0])
    
    def _generate_image_list(self, image_urls):
        image_list = []
        for image_url in image_urls:
            image_list.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            })
        return image_list

