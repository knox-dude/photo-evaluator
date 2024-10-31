import multiModelPhotoScoring
import os

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

image_urls2 = [
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_1301.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_1302.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_1303.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_1304.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_1305.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_1306.jpeg',
    'https://image-ranking-photos.s3.us-east-1.amazonaws.com/IMG_1307.jpeg'
]

if __name__ == "__main__":
    scoring_system = multiModelPhotoScoring.PhotoScoringSystem()
    results = []
    result = scoring_system.get_llm_analysis_multiple(image_urls2, os.environ.get("OPENAI_API_KEY"))
    for i in range(3):
        result = scoring_system.get_llm_analysis(image_urls[i], os.environ.get("OPENAI_API_KEY"))
        results.append(result)

    print(results)
