#################################################################
# Load the Image
#################################################################
from flask import Flask, request, jsonify
from flask_cors import CORS
from imutils.perspective import four_point_transform
import base64
import cv2
import numpy as np
# from pathlib import Path
# import os


app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app, resources={r"/process_image": {"origins": "*"}})  # Allow requests to /process_image from any origin (*)


# Define the folder where uploaded images will be stored temporarily
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

height = 800
width = 600
green = (0, 255, 0)


@app.route('/process_image', methods=['OPTIONS', 'POST'])
def process_image():
    print("request", request.files['image'])
    try:
        image = request.files['image']

        # Check if the 'image' file is included in the request
        if image is None:
            return jsonify({'error': 'No image provided'}), 400

       # image_binary = image.read()
        # image_base64 = base64.b64encode(image_binary).decode("utf-8")
        # decoded_image = base64.b64decode(image_base64)
       # image_np = np.frombuffer(image_binary, dtype=np.uint8)
        #processed_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # image_data = request.files['image']

        # # Save the received image to the upload folder
        # image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
        # image_data.save(image_path)

        image = cv2.imread(image)
        image = cv2.resize(image, (width, height))
        orig_image = image.copy()
#################################################################
# Image Processing
#################################################################

        # convert the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Add Gaussian blur
        # Apply the Canny algorithm to find the edges
        edged = cv2.Canny(blur, 75, 200)

        # Show the image and the edges
        cv2.imshow('Original image:', image)
        cv2.imshow('Edged:', edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#################################################################
# Use the Edges to Find all the Contours
#################################################################

# If you are using OpenCV v3, v4-pre, or v4-alpha
# cv.findContours returns a tuple with 3 element instead of 2
# where the `contours` is the second one
# In the version OpenCV v2.4, v4-beta, and v4-official
# the function returns a tuple with 2 element
        contours, _ = cv2.findContours(
            edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Show the image and all the contours
        cv2.imshow("Image", image)
        cv2.drawContours(image, contours, -1, green, 3)
        cv2.imshow("All contours", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#################################################################
# Select Only the Edges of the Document
#################################################################

# go through each contour
        for contour in contours:
            # we approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            # if we found a countour with 4 points we break the for loop
            # (we can assume that we have found our document)
            if len(approx) == 4:
                doc_cnts = approx
                break

#################################################################
# Apply Warp Perspective to Get the Top-Down View of the Document
#################################################################

# We draw the contours on the original image not the modified one
        cv2.drawContours(orig_image, [doc_cnts], -1, green, 3)
        cv2.imshow("Contours of the document", orig_image)
        # apply warp perspective to get the top-down view
        warped = four_point_transform(orig_image, doc_cnts.reshape(4, 2))
        # convert the warped image to grayscale
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Scanned", cv2.resize(warped, (600, 800)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        _, buffer = cv2.imencode('.jpg', warped)
        processed_image_base64 = base64.b64encode(buffer).decode()

        # Set CORS headers in the response
        response = jsonify({'message': 'Image processed successfully'})
        response.headers.add('Access-Control-Allow-Origin',
                             'http://localhost:4200')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')

        return response

        # Return the processed image as base64
        # return jsonify({'processedImage': processed_image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

#################################################################
# Bonus
#################################################################

# valid_formats = [".jpg", ".jpeg", ".png"]
# def get_text(f): return os.path.splitext(f)[1].lower()


# img_files = ['input/' +
#              f for f in os.listdir('input') if get_text(f) in valid_formats]
# # create a new folder that will contain our images
# Path("output").mkdir(exist_ok=True)

# # go through each image file
# for img_file in img_files:
#     # read, resize, and make a copy of the image
#     img = cv2.imread(img_file)
#     img = cv2.resize(img, (width, height))
#     orig_img = img.copy()

#     # preprocess the image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(img, 75, 200)

#      find and sort the contours
#     contours, _ = cv2.findContours(
#         edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     # go through each contour
#     for contour in contours:
#         # approximate each contour
#         peri = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
#         # check if we have found our document
#         if len(approx) == 4:
#             doc_cnts = approx
#             break

#     # apply warp perspective to get the top-down view
#     warped = four_point_transform(orig_img, doc_cnts.reshape(4, 2))
#     warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#     final_img = cv2.resize(warped, (600, 800))

#     # write the image in the ouput directory
#     cv2.imwrite("output" + "/" + os.path.basename(img_file), final_img)
