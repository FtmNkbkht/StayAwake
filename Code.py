import cv2
import dlib
import pygame
import datetime


# تعریف زمان‌های آلارم
alarm_times = [datetime.time(8, 0), datetime.time(12, 30), datetime.time(18, 45)]  # زمان‌های آلارم مورد نظر

# ایجاد شیء دوربین
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# بارگیری فایل Haarcascade چشم
eye_cascade = cv2.CascadeClassifier("D:/University/Karshenashi/Image Processing/Exercise/Project/StayAwake/models/haarcascade_eye.xml")

# بارگیری مدل تشخیص چهره
face_detector = dlib.get_frontal_face_detector()

# بارگیری مدل shape_predictor برای تشخیص چشم
predictor = dlib.shape_predictor("D:/University/Karshenashi/Image Processing/Exercise/Project/StayAwake/models/shape_predictor_68_face_landmarks.dat")

# تنظیمات پخش صدا
pygame.mixer.init()
pygame.mixer.music.load("D:/University/Karshenashi/Image Processing/Exercise/Project/StayAwake/models/sound.mp3")
is_playing = False
closed_duration = 0

# تنظیمات پیام‌های تشویقی
encouragement_messages = ["Stay awake!", "Move your arms!"]

# شمارنده برای نمایش پیام‌های تشویقی
message_counter = 0
message_duration = 0
message_interval = 30  # تعداد فریم‌ها قبل از نمایش پیام جدید

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # ساخت تاریخ و زمان فعلی
    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    # تنظیم پارامترهای متن
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    # محاسبه اندازه متن
    text_size, _ = cv2.getTextSize(current_time, font, font_scale, font_thickness)

    # مختصات متن
    text_x = 20  # مختصات x متن
    text_y = frame.shape[0] - 20  # مختصات y متن (پایین تصویر)

    # نمایش متن در ویدئو
    cv2.putText(frame, current_time, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


    # بررسی زمان آلارم و هشدار دادن در صورت تطابق
    if any(current_time == alarm_time for alarm_time in alarm_times):
        print("Wake up!")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        landmarks = predictor(gray, face)

        eyes = eye_cascade.detectMultiScale(gray[y:y + h, x:x + w])

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame[y:y + h, x:x + w], (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # بررسی بسته بودن چشم‌ها
        if len(eyes) == 0:
            closed_duration += 1
            if not is_playing:
                pygame.mixer.music.play()
                is_playing = True
        else:
            closed_duration = 0
            if is_playing:
                pygame.mixer.music.stop()
                is_playing = False

        # نمایش پیام تشویقی
        if closed_duration == message_interval:
            message_counter = (message_counter + 1) % len(encouragement_messages)
            message_duration = 0

        if message_duration < len(encouragement_messages) * message_interval:
            message = encouragement_messages[message_counter]
            text_position = (10, 50)
            font_size = 2
            text_color = (0, 255, 0)
            cv2.putText(frame, message, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, 2, cv2.LINE_AA)
            message_duration += 1

    # نمایش زمان در تصویر
    cv2.putText(frame, current_time, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # نمایش تصویر
    cv2.imshow("Stay Awake", frame)

    if cv2.waitKey(1) == 27:
      break

cap.release()
cv2.destroyAllWindows()