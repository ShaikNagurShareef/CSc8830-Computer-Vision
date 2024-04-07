import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify


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

# Define function to compute integral image
def compute_integral_image(frame):
    gray_clr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_clr.shape

    integral_image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            integral_image[i][j] = int(gray_clr[i][j])

    for i in range(1, width):
        integral_image[0][i] += integral_image[0][i - 1]

    for j in range(1, height):
        integral_image[j][0] += integral_image[j - 1][0]

    for i in range(1, height):
        for j in range(1, width):
            integral_image[i][j] = integral_image[i - 1][j] + integral_image[i][j - 1] - integral_image[i - 1][j - 1] + gray_clr[i][j]

    return integral_image

def image_stitch(img1, img2):
    # Convert images to Gray
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    
    # Initialize BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test to select good matches
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    matches = np.asarray(good)
    
    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find Homography matrix using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Warp image1 onto image2
        warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))
        
        # Combine images
        stitched_image = warped_img1.copy()
        stitched_image[0:img2.shape[0], 0:img2.shape[1]] = img2
        
        return stitched_image
    else:
        raise AssertionError("Can’t find enough keypoints for stitching.")

def read_images(img_paths):
    images = [cv2.imread(img_path) for img_path in img_paths]
    return images

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

            print("Camera Intrinsic Matrix: ", camera_matrix)

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
                annotated_image_path = os.path.join('static/object-realworld-dimensions', 'annotated_image.png')
                cv2.imwrite(annotated_image_path, annotated_image)

                result = f"Diameter of the circular object is: {distance} mm"

                return render_template('result.html', result=result, annotated_image=annotated_image_path)
            else:
                return render_template('error.html', message="No circular object detected in the image.")
        else:
            return render_template('error.html', message="No file uploaded.")
    else:
        return render_template('index.html')
    
@app.route('/stream_rgb_integral_imgs', methods=['POST'])
def stream_rgb_integral_imgs():
    if request.method == 'POST':
        file = request.files['image']

        if file:
            # Save the uploaded image to the uploads directory
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)
            image = cv2.imread(file_path)
            integral_image = compute_integral_image(image)
            np.savetxt('static/image_integrals/integral_image.txt', integral_image, fmt='%d')
            integral_image_path = os.path.join('static/image_integrals', 'integral_image.jpg')
            plt.imsave(integral_image_path, integral_image[..., ::-1])
            #cv2.imwrite(integral_image_path, integral_image)
            result = "Computed Integral Image is:"
        
            return render_template('result.html', result=result, annotated_image=integral_image_path)

        else:
            return render_template('error.html', message="No file uploaded.")
    else:
        return render_template('index.html')
    
@app.route('/stitch', methods=['POST'])
def stitch_images():
    # Load images
    img_files = request.files.getlist('images')
    if img_files:
        img_paths = []
        for img_file in img_files:
            img_path = os.path.join('static/uploads', img_file.filename)
            img_file.save(img_path)
            img_paths.append(img_path)

        images = read_images(img_paths)

        # Stitch images together
        panorama = images[0]
        for i in range(1, len(images)):
            stitched = image_stitch(panorama, images[i])
            if stitched is not None:
                panorama = stitched

        # Save stitched image
        panorama_image_path = os.path.join('static/stitched_images', 'annotated_image.png')
        cv2.imwrite(panorama_image_path, panorama)
        result = "Panoramic result after stitching given images"

        return render_template('result.html', result=result, annotated_image=panorama_image_path)
    else:
        return render_template('error.html', message="No images uploaded.")

if __name__ == '__main__':
    app.run(debug=True)
