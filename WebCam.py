import cv2

# نمایش لیست ورودی‌های وبکم
def show_available_webcams():
    num_cameras = cv2.getBuildInformation().count('videoio')
    for i in range(num_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Webcam ID: {i}")

show_available_webcams()