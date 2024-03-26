import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dust_detect import detect_blur_fft


def capture_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap


def detect_edges(roi_gray, min_threshold=50, max_threshold=150):
    # Convert the image to 8-bit unsigned integer
    roi_gray_uint8 = roi_gray.astype(np.uint8)

    blurred = cv2.GaussianBlur(src=roi_gray_uint8, ksize=(3, 5), sigmaX=0.5)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, min_threshold, max_threshold)

    return edges


def extract_resize_roi(frame, roi_coordinates, target_size=(100, 100)):
    # Extract the ROI using the provided coordinates
    roi_pts = np.array([[roi_coordinates['x1'], roi_coordinates['y1']],
                        [roi_coordinates['x2'], roi_coordinates['y2']],
                        [roi_coordinates['x3'], roi_coordinates['y3']],
                        [roi_coordinates['x4'], roi_coordinates['y4']]], dtype=np.int32)

    mask = np.zeros_like(frame[:, :, 0])
    cv2.fillPoly(mask, [roi_pts], (255, 255, 255))
    roi = cv2.bitwise_and(frame, frame, mask=mask)

    roi_resized = cv2.resize(roi, target_size)
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

    return roi_gray, mask


def apply_canny(frame):
    edges = detect_edges(frame, min_threshold=50, max_threshold=150)
    return edges


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def extract_features_and_target(frame, roi_coords):
    # Apply FFT to ROI
    roi_gray, _ = extract_resize_roi(frame, roi_coords, target_size=(100, 100))
    roi_fft = np.fft.fft2(roi_gray)
    roi_filtered = np.fft.ifft2(roi_fft).real  # Real part of inverse FFT

    # Normalize the filtered ROI
    normalized_roi = normalize_image(roi_filtered)

    # Reshape the normalized ROI to use as features
    features = normalized_roi.reshape(-1, 1)  # Reshape to a column vector

    return features


def apply_logistic_regression(frame, roi_coords):
     # Extract features
     features  = extract_features_and_target(frame, roi_coords)

     # Split the data into training and testing sets
     X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

     # Initialize logistic regression model
     model = LogisticRegression()

     # Train the model (unsupervised learning)
     model.fit(X_train, X_train)  # Using features as both input and target

     return model











def apply_fft_to_roi(frame, roi_coords):
    roi_gray, mask = extract_resize_roi(frame, roi_coords, target_size=(100, 100))

    roi_fft = np.fft.fft2(roi_gray)
    roi_filtered = np.fft.ifft2(roi_fft).real  # Real part of inverse FFT

    canny_edges = apply_canny(roi_filtered)


    canny_resized = cv2.resize(roi_filtered, (frame.shape[1], frame.shape[0]))

    return canny_resized, mask


# def get_fft_values(roi_filtered):
#     for row in roi_filtered:
#         for pixel_value in row:
#             pass
#
#     # If you want to print the values in a more structured format (e.g., row by row)
#     # You can use numpy's flatten() function to flatten the array and then iterate over it
#     flattened_roi = roi_filtered.flatten()
#
#     # Iterate over the flattened array
#     for pixel_value in flattened_roi:
#         print(pixel_value)


# Load the video
def process_video(video_path):
    cap = capture_video(video_path)

    cap.set(cv2.CAP_PROP_POS_MSEC, 816 * 1.0e3)

    # Define the ROI coordinates
    roi_coordinates = {
        "x1": 250,
        "y1": 580,
        "x2": 300,
        "y2": 300,
        "x3": 980,
        "y3": 300,
        "x4": 800,
        "y4": 580
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # # Apply logistic regression to the frame
        # model, accuracy = apply_logistic_regression(frame, roi_coordinates)
        #
        # # Get prediction probabilities for each pixel
        # features, _ = extract_features_and_target(frame, roi_coordinates)
        # prediction_probabilities = model.predict_proba(features)
        # # Reshape prediction probabilities to match frame shape
        # prediction_probabilities = prediction_probabilities[:, 1].reshape(frame.shape[0], frame.shape[1])
        #
        # # Threshold the prediction probabilities to get binary predictions
        # binary_predictions = np.where(prediction_probabilities > 0.5, 1, 0)

        # Apply FFT to ROI
        roi_filtered, mask = apply_fft_to_roi(frame, roi_coordinates)
        features  = extract_features_and_target(frame, roi_coordinates)
        print(np.sum(features.flatten())   )
        # Replace the original ROI in the frame with the filtered ROI
        frame_roi = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame_roi += cv2.cvtColor(roi_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Draw text on frame_roi
        mean, blurry = detect_blur_fft(frame, size=60, thresh=10)
        text = "Dusty ({:.4f})" if blurry else "Not Dusty ({:.4f})"
        text = text.format(mean)
        color = (0, 0, 255) if blurry else (0, 255, 0)
        cv2.putText(frame_roi, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        if np.sum(features.flatten()) < 1000:
            cv2.putText(frame_roi, f"Bridge {(np.sum(features.flatten()))}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        else:
            cv2.putText(frame_roi, "No Bridge", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        # Display the frame with ROI and text
        cv2.imshow('Filtered Frame ', frame_roi)

        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


# Call the function to process the video
process_video("data/crusher_bin_bridge2.mkv")
