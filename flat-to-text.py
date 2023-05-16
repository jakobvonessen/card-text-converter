import cv2
import numpy as np
import pytesseract

# Load the flattened image
image = cv2.imread('card-flattened.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to obtain a binary image
_, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Print OCR results for the whole image
whole_image_text = pytesseract.image_to_string(gray, lang='swe')
print("OCR Results for the Whole Image:")
#print(whole_image_text)
print("--------------------")

# Split the OCR results into lines and remove empty lines
lines = [line.strip() for line in whole_image_text.split('\n') if line.strip()]

# Initialize empty lists for titles and questions
titles = []
questions = []

# Iterate over the lines and classify them as titles or questions

for i, line in enumerate(lines):
    if line.endswith("?"):
        if len(line) > len(lines[i-1]):
            questions.append(line)
            titles.append(lines[i-1])
        else:
            questions.append(lines[i-1] + " " + line)
            titles.append(lines[i-2])

# Print the extracted titles
print("Titles:")
for title in titles:
    print(title)
print("--------------------")

# Print the extracted questions
print("Questions:")
for question in questions:
    print(question)
print("--------------------")
