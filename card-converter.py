import cv2
import numpy as np
import pytesseract

from tkinter import filedialog
from tkinter import Tk

import pandas as pd

# Create the root Tkinter window
root = Tk()
root.withdraw()  # Hide the root window

# Open a file dialog for selecting card photos
card_paths = filedialog.askopenfilenames(title="Open card files")

# Open a file dialog for selecting Excel file for export
excel_path = filedialog.askopenfilename(title="Select Excel file to append to")
df = pd.read_excel(excel_path)

# Initialize empty lists for titles and questions
titles = []
questions = []

for card_path in card_paths:
    # Load the card photo
    image = cv2.imread(card_path)

    # Get the image dimensions
    height, width, _ = image.shape

    # Define the color range for green (with ±10 variance)
    target_color = np.array([167, 169, 163], dtype=np.uint8)
    color_variance = 28
    lower_green = np.array([max(0, target_color[0] - color_variance),
                        max(0, target_color[1] - color_variance),
                        max(0, target_color[2] - color_variance)], dtype=np.uint8)
    upper_green = np.array([min(255, target_color[0] + color_variance),
                        min(255, target_color[1] + color_variance),
                        min(255, target_color[2] + color_variance)], dtype=np.uint8)

    # Create a mask for the green color range
    mask = cv2.inRange(image, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it represents the green region)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Ensure the polygon has four corners
    if len(approx_polygon) == 4:
        # Order the corners of the polygon
        ordered_corners = np.zeros((4, 2), dtype=np.float32)
        approx_polygon = approx_polygon.reshape(-1, 2)
        summed = np.sum(approx_polygon, axis=1)
        ordered_corners[0] = approx_polygon[np.argmin(summed)]
        ordered_corners[2] = approx_polygon[np.argmax(summed)]
        diff = np.diff(approx_polygon, axis=1)
        ordered_corners[1] = approx_polygon[np.argmin(diff)]
        ordered_corners[3] = approx_polygon[np.argmax(diff)]

        # Define the dimensions of the output image
        output_width = max(np.linalg.norm(ordered_corners[0] - ordered_corners[1]),
                        np.linalg.norm(ordered_corners[2] - ordered_corners[3]))
        output_height = max(np.linalg.norm(ordered_corners[0] - ordered_corners[3]),
                            np.linalg.norm(ordered_corners[1] - ordered_corners[2]))

        # Define the corners of the output image
        target_corners = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
                                dtype=np.float32)

        # Compute the perspective transformation matrix
        transformation_matrix = cv2.getPerspectiveTransform(ordered_corners, target_corners)

        # Apply the perspective transformation to obtain the flat photo
        flattened_image = cv2.warpPerspective(image, transformation_matrix, (int(output_width), int(output_height)))

        # Export the flattened image
        #cv2.imwrite('card-flattened.jpg', flattened_image)

        # Convert the image to grayscale
        gray = cv2.cvtColor(flattened_image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to obtain a binary image
        _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Print OCR results for the whole image
        whole_image_text = pytesseract.image_to_string(gray, lang='swe')
        print("OCR Results for the Whole Image:")
        #print(whole_image_text)
        print("--------------------")

        # Split the OCR results into lines and remove empty lines
        lines = [line.strip() for line in whole_image_text.split('\n') if line.strip()]



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

new_data = {'Frågor': questions}
new_df = pd.DataFrame(new_data)

# Append the new DataFrame to the existing DataFrame
updated_df = pd.concat([df, new_df], ignore_index=True)

# Write the updated DataFrame back to the Excel file
updated_df.to_excel(excel_path, index=False)