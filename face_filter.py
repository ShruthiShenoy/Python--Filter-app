import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
from tkinter import filedialog

# Load filters
filters = {
    "None": cv2.imread("filters/none.png", -1),
    "Glasses": cv2.imread("filters/glasses.png", -1),
    "Crown": cv2.imread("filters/crown.png", -1),
    "Heart": cv2.imread("filters/heart.png", -1),
    "Spiderman" : cv2.imread("filters/spiderman.png", -1),
}

selected_filter = None
selected_filter_name = "None"

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# OpenCV Camera Setup
cap = cv2.VideoCapture(0)

def overlay_filter(frame, filter_img, x, y, w, h):
    """Overlay filter image on the video frame at specified position."""
    if filter_img is None:
        return

    filter_img = cv2.resize(filter_img, (w, h))

    for i in range(h):
        for j in range(w):
            if filter_img[i, j][3] != 0:  # Not transparent
                if 0 <= y + i < frame.shape[0] and 0 <= x + j < frame.shape[1]:
                    frame[y + i, x + j] = filter_img[i, j][:3]

def process_frame():
    """Captures a frame and applies the selected filter."""
    global selected_filter, selected_filter_name

    success, frame = cap.read()
    if not success:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            if selected_filter_name == "Glasses":
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                x1 = int(left_eye.x * w)
                x2 = int(right_eye.x * w)
                y = int(left_eye.y * h) - 20
                filter_width = x2 - x1 + 60
                filter_height = int(filter_width * selected_filter.shape[0] / selected_filter.shape[1])
                overlay_filter(frame, selected_filter, x1 - 30, y, filter_width, filter_height)

            elif selected_filter_name == "Crown":
                forehead = face_landmarks.landmark[10]
                x = int(forehead.x * w)
                y = int(forehead.y * h) - 120
                filter_width = 200
                filter_height = 100
                overlay_filter(frame, selected_filter, x - filter_width // 2, y, filter_width, filter_height)

            elif selected_filter_name == "Heart":
                forehead = face_landmarks.landmark[10]
                x = int(forehead.x * w)
                y = int(forehead.y * h) - 120
                filter_width = 200
                filter_height = 100
                overlay_filter(frame, selected_filter, x - filter_width // 2, y, filter_width, filter_height)

            elif selected_filter_name == "Spiderman":
                if selected_filter is not None:
                    # Get face bounding box from landmarks
                    x_min = int(min(landmark.x for landmark in face_landmarks.landmark) * w)
                    y_min = int(min(landmark.y for landmark in face_landmarks.landmark) * h)
                    x_max = int(max(landmark.x for landmark in face_landmarks.landmark) * w)
                    y_max = int(max(landmark.y for landmark in face_landmarks.landmark) * h)

                    # Calculate face width and height
                    face_width = x_max - x_min
                    face_height = y_max - y_min

                    # Adjust mask size slightly bigger than face
                    mask_width = int(face_width * 1.2)
                    mask_height = int(face_height * 1.4)

                    # Center the mask over the face
                    x_mask = x_min - int((mask_width - face_width) / 2)
                    y_mask = y_min - int((mask_height - face_height) / 2)

                    overlay_filter(frame, selected_filter, x_mask, y_mask, mask_width, mask_height)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)

    label_img.config(image=img)
    label_img.image = img
    window.after(10, process_frame)

def set_filter(filter_name):
    """Change selected filter."""
    global selected_filter, selected_filter_name
    selected_filter_name = filter_name
    selected_filter = filters[filter_name]

def capture_and_save():
    success, frame = cap.read()
    if success:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, frame)
            print(f"Image saved as {file_path}")

# Tkinter UI
window = tk.Tk()
window.title("Face Filter App")

label_img = Label(window)
label_img.pack()

# Filter buttons
button_frame = tk.Frame(window)
button_frame.pack()

for filter_name, filter_img in filters.items():
    img = Image.open(f"filters/{filter_name.lower()}.png") if filter_img is not None else Image.new("RGB", (50, 50), "gray")
    img = img.resize((50, 50))
    img = ImageTk.PhotoImage(img)
    
    btn = Button(button_frame, image=img, command=lambda name=filter_name: set_filter(name))
    btn.image = img
    btn.pack(side="left", padx=5)

# Capture button
btn_capture = Button(window, text="Capture & Save", command=capture_and_save)
btn_capture.pack()

process_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()
window.destroy()

