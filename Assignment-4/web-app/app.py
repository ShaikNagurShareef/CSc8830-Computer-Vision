import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, Response

import imutils
import depthai as dai
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist


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

##### Assignment-4 ######

# Function to detect QR codes in an image
def detect_qr_codes(frame):
    qcd = cv2.QRCodeDetector()
    ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
    if ret_qr:
        for s, p in zip(decoded_info, points):
            if s:
                # Draw rectangle
                frame = cv2.polylines(frame, [p.astype(int)], True, (0, 255, 0), 8)
                # Add text
                cv2.putText(frame, s, (int(p[0][0]), int(p[0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                color = (0, 0, 255)
                frame = cv2.polylines(frame, [p.astype(int)], True, color, 8)
    return frame

class ObjectTracker:
    def __init__(self):
        # Dictionary to store the center points of the tracked objects
        self.object_centers = {}
        # Counter to keep track of object IDs
        self.object_id_count = 0

    def update(self, frame):
        # Extract Region of Interest (ROI)
        roi = frame

        # Background subtractor for object detection
        object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

        # 1. Object Detection
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        # 2. Object Tracking
        tracked_boxes_ids = self._track_objects(detections)
        for box_id in tracked_boxes_ids:
            x, y, w, h, obj_id = box_id
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        return roi

    def _track_objects(self, object_rectangles):
        # List to store object bounding boxes and IDs
        tracked_objects = []

        # Update center points of existing objects or assign new IDs
        for rect in object_rectangles:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Check if the object was detected previously
            object_detected = False
            for obj_id, center in self.object_centers.items():
                dist = math.hypot(cx - center[0], cy - center[1])

                # If distance is within a threshold, update center and ID
                if dist < 25:
                    self.object_centers[obj_id] = (cx, cy)
                    tracked_objects.append([x, y, w, h, obj_id])
                    object_detected = True
                    break

            # If new object detected, assign a new ID
            if not object_detected:
                self.object_centers[self.object_id_count] = (cx, cy)
                tracked_objects.append([x, y, w, h, self.object_id_count])
                self.object_id_count += 1

        # Clean up dictionary by removing unused IDs
        new_object_centers = {}
        for obj_bb_id in tracked_objects:
            _, _, _, _, obj_id = obj_bb_id
            center = self.object_centers[obj_id]
            new_object_centers[obj_id] = center

        # Update dictionary with used IDs
        self.object_centers = new_object_centers.copy()
        return tracked_objects

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

class ObjectDimensionMarker:
    def __init__(self, object_width_inches):
        self.object_width_inches = object_width_inches
        self.pixelsPerMetric = None

    def mark_object_dimensions(self, gray_frame):
        # convert the frame to grayscale and blur it slightly
        gray = cv2.GaussianBlur(gray_frame, (7, 7), 0)

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # sort the contours from left-to-right
        (cnts, _) = contours.sort_contours(cnts)
        
        # Initialize result frame
        result_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # loop over the contours individually
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                continue

            # compute the rotated bounding box of the contour
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour and draw the outline of the rotated bounding box
            box = perspective.order_points(box)
            cv2.drawContours(result_frame, [box.astype("int")], -1, (0, 255, 0), 2)

            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(result_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack the ordered bounding box
            (tl, tr, br, bl) = box

            # compute midpoints
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # draw midpoints
            cv2.circle(result_frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(result_frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(result_frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(result_frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            # draw lines between the midpoints
            cv2.line(result_frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 2)
            cv2.line(result_frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 2)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            if self.pixelsPerMetric is None:
                self.pixelsPerMetric = dB / self.object_width_inches

            # compute the size of the object in centimeters
            dimA_cm = dA / self.pixelsPerMetric * 2.54  # converting inches to centimeters
            dimB_cm = dB / self.pixelsPerMetric * 2.54

            # draw the object sizes on the image
            cv2.putText(result_frame, "{:.1f} in".format(dimA_cm),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 0), 2)
            cv2.putText(result_frame, "{:.1f} in".format(dimB_cm),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 0), 2)

        return result_frame


#########################

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
    
def process_feed():
    object_width = 2.24
    try:
        # Create object tracker instance
        tracker = ObjectTracker()

        # Initialize object dimension marker
        object_marker = ObjectDimensionMarker(object_width_inches=object_width)

        # Create a pipeline
        pipeline = dai.Pipeline()

        # Define a node for the stereo camera
        stereo = pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)

        # Define a node for the left camera
        left = pipeline.createMonoCamera()
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        # Define a node for the right camera
        right = pipeline.createMonoCamera()
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        # Connect left and right camera outputs to the stereo node
        left.out.link(stereo.left)
        right.out.link(stereo.right)

        # Define a node for the output
        xoutDepth = pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # Link stereo camera output to the output node
        stereo.depth.link(xoutDepth.input)

        # Define a node to get left camera frames
        xoutLeft = pipeline.createXLinkOut()
        xoutLeft.setStreamName("left")

        # Link left camera output to the output node
        left.out.link(xoutLeft.input)

        # Define a node to get right camera frames
        xoutRight = pipeline.createXLinkOut()
        xoutRight.setStreamName("right")

        # Link right camera output to the output node
        right.out.link(xoutRight.input)

        # Define a source - color camera
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        # Create RGB output
        xout = pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        cam.video.link(xout.input)

        # Connect to the device
        with dai.Device(pipeline) as device:
            # Output queues
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            leftQueue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
            rightQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
            rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            # Start the pipeline
            device.startPipeline()

            # OpenCV setup
            camera_id = 0
            delay = 1
            window_name = 'Object Tracking & Recgnition through QR Code'
            cap = cv2.VideoCapture(camera_id)

            while True:
                # Get the depth frame
                inDepth = depthQueue.get()

                # Get the left camera frame
                inLeft = leftQueue.get()

                # Get the right camera frame
                inRight = rightQueue.get()

                # Get the rgb camera frame
                inSrc = rgbQueue.get()

                # Access the depth data
                depthFrame = inDepth.getFrame()

                # Access the left camera frame
                leftFrame = inLeft.getCvFrame()

                # Access the right camera frame
                rightFrame = inRight.getCvFrame()
                
                # Data is originally represented as a flat 1D array, it needs to be converted into HxW form
                rgbFrame = inSrc.getCvFrame()

                # Combine left and right frames horizontally
                stereoFrame = cv2.hconcat([leftFrame, rightFrame])

                # Detect QR codes in the stereo frame
                stereoFrame = detect_qr_codes(stereoFrame)

                # Perform object detection and tracking
                stereoFrame = tracker.update(stereoFrame)

                # Convert frame to grayscale
                gray_frame = cv2.cvtColor(rgbFrame, cv2.COLOR_BGR2GRAY)

                # Mark object dimensions
                frame = object_marker.mark_object_dimensions(stereoFrame)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except:
        # Create object tracker instance
        tracker = ObjectTracker()

        # Initialize object dimension marker
        object_marker = ObjectDimensionMarker(object_width_inches=object_width)

        cap = cv2.VideoCapture(0)  # Use the correct device index
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect QR codes in the stereo frame
            frame = detect_qr_codes(frame)

            # Perform object detection and tracking
            frame = tracker.update(frame)

            # Convert frame to grayscale
            frame = object_marker.mark_object_dimensions(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_stream')
def video_stream():
    return Response(process_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


def process_objects_feed():
    object_width = 2.24
    try:
        # Create object tracker instance
        tracker = ObjectTracker()

        # Initialize object dimension marker
        object_marker = ObjectDimensionMarker(object_width_inches=object_width)

        # Create a pipeline
        pipeline = dai.Pipeline()

        # Define a node for the stereo camera
        stereo = pipeline.createStereoDepth()
        stereo.setConfidenceThreshold(255)

        # Define a node for the left camera
        left = pipeline.createMonoCamera()
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        # Define a node for the right camera
        right = pipeline.createMonoCamera()
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        # Connect left and right camera outputs to the stereo node
        left.out.link(stereo.left)
        right.out.link(stereo.right)

        # Define a node for the output
        xoutDepth = pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        # Link stereo camera output to the output node
        stereo.depth.link(xoutDepth.input)

        # Define a node to get left camera frames
        xoutLeft = pipeline.createXLinkOut()
        xoutLeft.setStreamName("left")

        # Link left camera output to the output node
        left.out.link(xoutLeft.input)

        # Define a node to get right camera frames
        xoutRight = pipeline.createXLinkOut()
        xoutRight.setStreamName("right")

        # Link right camera output to the output node
        right.out.link(xoutRight.input)

        # Define a source - color camera
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        # Create RGB output
        xout = pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        cam.video.link(xout.input)

        # Connect to the device
        with dai.Device(pipeline) as device:
            # Output queues
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            leftQueue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
            rightQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
            rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            # Start the pipeline
            device.startPipeline()

            # OpenCV setup
            camera_id = 0
            delay = 1
            window_name = 'Object Tracking & Recgnition through QR Code'
            cap = cv2.VideoCapture(camera_id)

            while True:
                # Get the depth frame
                inDepth = depthQueue.get()

                # Get the left camera frame
                inLeft = leftQueue.get()

                # Get the right camera frame
                inRight = rightQueue.get()

                # Get the rgb camera frame
                inSrc = rgbQueue.get()

                # Access the depth data
                depthFrame = inDepth.getFrame()

                # Access the left camera frame
                leftFrame = inLeft.getCvFrame()

                # Access the right camera frame
                rightFrame = inRight.getCvFrame()
                
                # Data is originally represented as a flat 1D array, it needs to be converted into HxW form
                rgbFrame = inSrc.getCvFrame()

                # Combine left and right frames horizontally
                stereoFrame = cv2.hconcat([leftFrame, rightFrame])

                # Detect QR codes in the stereo frame
                stereoFrame = detect_qr_codes(stereoFrame)

                # Perform object detection and tracking
                frame = tracker.update(stereoFrame)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except:
        # Create object tracker instance
        tracker = ObjectTracker()

        # Initialize object dimension marker
        object_marker = ObjectDimensionMarker(object_width_inches=object_width)

        cap = cv2.VideoCapture(0)  # Use the correct device index
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect QR codes in the stereo frame
            frame = detect_qr_codes(frame)

            # Perform object detection and tracking
            frame = tracker.update(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/detect_objects_video_stream')
def detect_objects_video_stream():
    return Response(process_objects_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/object_detection', methods=['GET', 'POST'])
def object_detection():
    try:
        return render_template('object_detection.html')
    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
