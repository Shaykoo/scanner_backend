#################################################################
# Load the Image
#################################################################
from flask import Flask, request, jsonify
from flask_cors import CORS
from imutils.perspective import four_point_transform
import base64
import cv2
import numpy as np
# from pyzbar.pyzbar import decode

app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app)
#CORS(app, resources={r"/process_image": {"origins": ["https://scanner-frontend-hosting.web.app/"]}})


height = 800
width = 600
green = (0, 255, 0)

@app.route('/process_image', methods=['OPTIONS', 'POST'])
def process_image():
    global doc_cnts
    try:
        image = request.files['image']
        print("image",image)
        # Check if the 'image' file is included in the request
        if image is None:
            return jsonify({'error': 'No image provided'}), 400

        image_binary = image.read()
        # Convert the image data to a numpy array
        image_np = np.frombuffer(image_binary, dtype=np.uint8)
        # Decode the image using OpenCV
        processed_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)


        # Image Processing
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Add Gaussian blur
        edged = cv2.Canny(blur, 75, 200)

        #################################################################
        # Use the Edges to Find all the Contours
        #################################################################

        contours, _ = cv2.findContours(
            edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        #################################################################
        # Select Only the Edges of the Document
        #################################################################

        # Go through each contour
        for contour in contours:
            # We approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            # If we found a countour with 4 points we break the for loop
            # (we can assume that we have found our document)
            if len(approx) == 4:
                doc_cnts = approx
                break

        #################################################################
        # Apply Warp Perspective to Get the Top-Down View of the Document
        #################################################################

        # We draw the contours on the original image not the modified one
        cv2.drawContours(processed_image, [doc_cnts], -1, green, 3)
        # Apply warp perspective to get the top-down view
        warped = four_point_transform(processed_image, doc_cnts.reshape(4, 2))
        # Convert the warped image to grayscale
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Enhance image quality
        # alpha = 1.5  # Contrast control (1.0-3.0)
        # beta = 30  # Brightness control (0-100)
        #
        # enhanced_warped = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)

        _, buffer = cv2.imencode('.jpg', warped)
        processed_image_base64 = base64.b64encode(buffer).decode()

        # barcode_data_list = []
        # # Barcode scanning
        # for code in decode(warped):
        #     barcode_data = code.data.decode("utf-8")
        #     print("bbb", barcode_data)
        #     barcode_data_list.append(barcode_data)

        # Set CORS headers in the response
        # Set CORS headers in the response
        response_data = {
            'message': 'Image processed successfully',
            'processedImage': processed_image_base64,
        }

        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin',
                             'https://scanner-frontend-hosting.web.app')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')

        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000)
