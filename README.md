# Photo Evaluator

This project aims to create a backend that evaluates photos via multiple criteria (not in this order):

1. Existing neural networks that support image input (chatGPT, Claude, etc.) - implemented
2. BLIP-2 model - work in progress
3. CV2 blur detection, sharpness detection, rule of thirds analysis, exposure detection - work in progress

The point of this backend is to get several similar photos and have AI choose which are the best photos. The ultimate goal is to create a mobile app that first clusters similar photos, then allows users to scan clusters for the best photos.

## Issues

1. Cost - it costs about 2 cents to upload a 5 MB image to chatGPT. I'm working towards downscaling these images and seeing if the accuracy of chatGPT's selection continues to work. I'll need some test photos for that.
2. Personal Preferences - "best photos" is a vague term. How do you rate 5 very similar photos? This is an issue that has been raised. I'm moving towards a solution for this, and I am currently thinking of training personal neural networks based on people's selections. That's a very ambitious step, so it will come after the implementation of the generalized photo evaluator.
