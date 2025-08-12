import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox


def load_models():
    # Load face detection model
    face_proto = "opencv_face_detector.pbtxt"
    face_model = "opencv_face_detector_uint8.pb"
    face_net = cv2.dnn.readNet(face_model, face_proto)

    # Load age detection model
    age_proto = "age_deploy.prototxt"
    age_model = "age_net.caffemodel"
    age_net = cv2.dnn.readNet(age_model, age_proto)

    # Load gender detection model
    gender_proto = "gender_deploy.prototxt"
    gender_model = "gender_net.caffemodel"
    gender_net = cv2.dnn.readNet(gender_model, gender_proto)

    return face_net, age_net, gender_net


def get_face_box(net, frame, conf_threshold=0.7):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(
        frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False
    )

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])

    return bboxes


def predict_age_gender(face_net, age_net, gender_net, frame, face):
    # Prepare face for age and gender prediction
    (x, y, w, h) = face
    face_img = frame[y : y + h, x : x + w].copy()

    # Predict Gender
    blob = cv2.dnn.blobFromImage(
        face_img,
        1.0,
        (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False,
    )
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = "Male" if gender_preds[0][0] > 0.5 else "Female"

    # Predict Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_list = [
        "(0-2)",
        "(4-6)",
        "(8-12)",
        "(15-20)",
        "(20-25)",
        "(26-32)",
        "(38-43)",
        "(48-53)",
        "(60-100)",
    ]
    age = age_list[age_preds[0].argmax()]

    return gender, age


def detect_faces_and_predict(image, face_net, age_net, gender_net, padding=20):
    try:
        if image is None or image.size == 0:
            print("Error: Empty or invalid image")
            return None

        frame_copy = image.copy()

        if len(frame_copy.shape) == 2:
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
        elif frame_copy.shape[2] == 4:
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)

        # MODIFICATION: Increased top padding for more space
        top_padding = 100
        frame_with_border = cv2.copyMakeBorder(
            frame_copy, top_padding, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        frame_height, frame_width = frame_copy.shape[:2]
        bboxes = get_face_box(face_net, frame_copy)

        if not bboxes:
            print("No faces detected in the image")
            return frame_with_border

        print(f"Detected {len(bboxes)} face(s) in the image")

        labels = []
        for bbox in bboxes:
            try:
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]

                if w <= 0 or h <= 0:
                    continue

                gender, age = predict_age_gender(
                    face_net,
                    age_net,
                    gender_net,
                    frame_copy,
                    (x, y, w, h),
                )

                labels.append(f"{gender}, {age}")

                # Draw the rectangle on the face, offsetting for the top padding
                rect_y1 = y + top_padding
                rect_y2 = y + h + top_padding
                cv2.rectangle(
                    frame_with_border, (x, rect_y1), (x + w, rect_y2), (0, 255, 0), 2
                )

            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue

        # MODIFICATION: New logic to reliably draw text
        if labels:
            full_label_text = " | ".join(labels)

            font_scale = 1.0
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                full_label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Reduce font size if the text is wider than the image
            if text_width > frame_with_border.shape[1] - 20:
                font_scale = 0.7
                (text_width, text_height), baseline = cv2.getTextSize(
                    full_label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    font_thickness,
                )

            # Center the text horizontally
            text_x = (frame_with_border.shape[1] - text_width) // 2

            # Use a fixed, safe vertical position for the text baseline
            text_y = 60

            cv2.putText(
                frame_with_border,
                full_label_text,
                (max(text_x, 10), text_y),  # Use max() to prevent negative x-coordinate
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text color
                font_thickness,
                cv2.LINE_AA,
            )

        return frame_with_border

    except Exception as e:
        print(f"Error in detect_faces_and_predict: {str(e)}")
        return None


def process_image(face_net, age_net, gender_net):
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("All supported types", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*"),
        ],
    )

    if not file_path:
        return

    try:
        image = cv2.imread(file_path)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read the image file: {file_path}")

        print(f"Image loaded successfully. Dimensions: {image.shape}")

        result = detect_faces_and_predict(image, face_net, age_net, gender_net)

        if result is None or result.size == 0:
            raise ValueError("No result returned from face detection")

        window_name = "Detection Result - Press any key to close"
        cv2.imshow(window_name, result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}\n\nFile path: {file_path}"
        messagebox.showerror("Error", error_msg)


def start_webcam(face_net, age_net, gender_net):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_faces_and_predict(frame, face_net, age_net, gender_net)

        cv2.imshow("Age and Gender Detection (Press 'q' to quit)", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def create_gui(face_net, age_net, gender_net):
    root = tk.Tk()
    root.title("Age and Gender Detection")
    root.geometry("400x300")

    try:
        root.iconbitmap(default="")
    except tk.TclError:
        pass

    root.configure(bg="#f0f0f0")
    button_style = {
        "font": ("Arial", 12),
        "bg": "#4CAF50",
        "fg": "white",
        "activebackground": "#45a049",
        "padx": 20,
        "pady": 10,
        "borderwidth": 0,
        "relief": "flat",
        "cursor": "hand2",
    }

    title_label = tk.Label(
        root,
        text="Age and Gender Detection",
        font=("Arial", 18, "bold"),
        bg="#f0f0f0",
        pady=10,
    )
    title_label.pack(pady=(10, 0))

    button_frame = tk.Frame(root, bg="#f0f0f0")
    button_frame.pack(expand=True, fill="y", pady=20)

    webcam_btn = tk.Button(
        button_frame,
        text="Start Webcam",
        command=lambda: start_webcam(face_net, age_net, gender_net),
        **button_style,
    )
    webcam_btn.pack(pady=10, ipadx=20)

    image_btn = tk.Button(
        button_frame,
        text="Select Image",
        command=lambda: process_image(face_net, age_net, gender_net),
        **button_style,
    )
    image_btn.pack(pady=10, ipadx=20)

    exit_btn = tk.Button(
        button_frame,
        text="Exit",
        command=root.quit,
        bg="#f44336",
        activebackground="#d32f2f",
        **{
            k: v for k, v in button_style.items() if k not in ["bg", "activebackground"]
        },
    )
    exit_btn.pack(pady=10, ipadx=20)

    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()


def main():
    try:
        face_net, age_net, gender_net = load_models()
        create_gui(face_net, age_net, gender_net)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load models: {str(e)}")
        return


if __name__ == "__main__":
    main()
