from flask import Flask, render_template, request, jsonify
import cv2
import math
import numpy as np
import os

app = Flask(__name__)

def find_circle_properties(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=250)

    if circles is not None:
        circles = circles[0, :]

        # Assuming there is only one circle, get its properties
        x, y, r = circles[0]

        # Calculate center point
        center = (int(x), int(y))

        # Calculate width and height (diameter)
        width = int(2 * r)
        height = int(2 * r)

        return center, width, height
    else:
        print("No circle detected in the image.")
        return None, None, None

def convert_milli_to_inch(x):
    x = x / 10
    return x / 25.4

def calculate_object_distance(image, bbox, fx, fy, Z):
    # Unpack bounding box coordinates
    x, y, w, h = bbox

    # Calculate image points
    Image_point1x = x
    Image_point1y = y
    Image_point2x = x + w
    Image_point2y = y + h

    # Draw a line between two points
    cv2.line(image, (Image_point1x, Image_point1y-h//2), (Image_point1x, Image_point2y-h//2), (0, 0, 255), 8)

    # Convert image points to real-world coordinates
    Real_point1x = Z * (Image_point1x / fx)
    Real_point1y = Z * (Image_point1y / fy)
    Real_point2x = Z * (Image_point2x / fx)
    Real_point2y = Z * (Image_point2x / fy)

    print("Real World Co-ordinates: ")
    print("\t", Real_point1x)
    print("\t", Real_point1y)
    print("\t", Real_point2x)
    print("\t", Real_point2y)

    # Calculate the distance between two points
    dist = math.sqrt((Real_point2y - Real_point1y) ** 2 + (Real_point2x - Real_point1x) ** 2)

    val = round(convert_milli_to_inch(dist*2)*10, 2)

    # Draw text on the image with the calculated distance
    cv2.putText(image, str(val)+" mm", (Image_point1x - 200, (y + h) // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite("static/circilar_object.png", image)

    return image, val

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            # Save the uploaded image to the uploads directory
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)

            # Load camera matrix
            camera_matrix = []
            with open('../images/left/camera_matrix.txt', 'r') as f:
                for line in f:
                    camera_matrix.append([float(num) for num in line.split(' ')])

            # Define object distance from the camera in mm
            object_dist = 300

            # Extract focal length and object distance
            fx, fy, Z = camera_matrix[0][0], camera_matrix[1][1], object_dist

            # Find circle properties in the uploaded image
            (x, y), w, h = find_circle_properties(file_path)
            bbox = (x, y, w, h)

            print(f"fx: {fx}, fy: {fy}, Z: {Z}")
            print(f"Center (x, y): {x}, {y}; Width (w): {w}; Height (h): {h}")

            # If a circular object is detected, calculate its distance
            if x is not None:
                bbox = (x, y, w, h)
                image = cv2.imread(file_path)
                annotated_image, distance = calculate_object_distance(image, bbox, fx, fy, Z)
                annotated_image_path = os.path.join('static', 'annotated_image.png')
                cv2.imwrite(annotated_image_path, annotated_image)

                return render_template('result.html', distance=distance, annotated_image=annotated_image_path)
            else:
                return render_template('error.html', message="No circular object detected in the image.")
        else:
            return render_template('error.html', message="No file uploaded.")
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
