import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import cv2
from ultralytics import YOLO
import os

class PersonDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Person Detector App")
        self.master.geometry("800x600")

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")  # bisa ganti ke yolov8s.pt untuk lebih akurat
        self.cap = None
        self.running = False

        # Video frame label
        self.label = tk.Label(master)
        self.label.pack()

        # Tombol Pilih Video
        self.btn_open = tk.Button(master, text="Pilih Video", command=self.open_video)
        self.btn_open.pack(pady=5)

        # Tombol Start
        self.btn_start = tk.Button(master, text="Start", command=self.start_video, state=tk.DISABLED)
        self.btn_start.pack(pady=5)

        # Tombol Stop
        self.btn_stop = tk.Button(master, text="Stop", command=self.stop_video, state=tk.DISABLED)
        self.btn_stop.pack(pady=5)

        # Label Nama File
        self.filename_var = tk.StringVar()
        self.filename_var.set("Video: -")
        self.label_filename = tk.Label(master, textvariable=self.filename_var, font=("Arial", 12, "bold"))
        self.label_filename.pack(pady=5)

        # Label Jumlah Orang
        self.people_count_var = tk.StringVar()
        self.people_count_var.set("Jumlah orang: 0")
        self.label_count = tk.Label(master, textvariable=self.people_count_var, font=("Arial", 14))
        self.label_count.pack(pady=10)

        # Variabel tracking
        self.crossed_people = set()
        self.last_centroids = []
        self.next_id = 0

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if path:
            self.video_path = path
            filename = os.path.basename(path)
            self.filename_var.set(f"Video: {filename}")
            self.btn_start.config(state=tk.NORMAL)

    def start_video(self):
        if not hasattr(self, 'video_path'):
            messagebox.showwarning("Peringatan", "Silakan pilih video dulu.")
            return
        self.cap = cv2.VideoCapture(self.video_path)
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.crossed_people = set()
        self.last_centroids = []
        self.next_id = 0
        threading.Thread(target=self.process_video, daemon=True).start()

    def stop_video(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def process_video(self):
        line_y = None
        max_dist = 50  # Jarak maksimum untuk matching centroid

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if line_y is None:
                height, width = frame.shape[:2]
                line_y = int(height * 0.75)

            results = self.model(frame)[0]
            current_centroids = []

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    current_centroids.append((cx, cy, x1, y1, x2, y2))

            updated_centroids = []
            matched_ids = set()

            for cx, cy, x1, y1, x2, y2 in current_centroids:
                matched = False
                for pid, pcx, pcy, crossed in self.last_centroids:
                    if abs(cx - pcx) < max_dist and abs(cy - pcy) < max_dist and pid not in matched_ids:
                        matched_ids.add(pid)
                        if not crossed and cy >= line_y and pcy < line_y:
                            crossed = True
                            self.crossed_people.add(pid)
                        updated_centroids.append((pid, cx, cy, crossed))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'ID:{pid}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        matched = True
                        break

                if not matched:
                    pid = self.next_id
                    self.next_id += 1
                    crossed = cy >= line_y
                    if crossed:
                        self.crossed_people.add(pid)
                    updated_centroids.append((pid, cx, cy, crossed))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'ID:{pid}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            self.last_centroids = updated_centroids

            # Garis deteksi
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
            self.people_count_var.set(f"Jumlah orang: {len(self.crossed_people)}")

            # Tampilkan frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((720, 480))
            imgtk = ImageTk.PhotoImage(image=img)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if self.cap:
            self.cap.release()
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

# Jalankan aplikasi
if __name__ == "__main__":
    root = tk.Tk()
    app = PersonDetectorApp(root)
    root.mainloop()
