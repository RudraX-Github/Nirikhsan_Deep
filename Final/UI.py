import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import cv2
import os
import time
import threading
import face_recognition
import numpy as np
import glob
from collections import deque

from Utils import CONFIG, logger, get_sound_path, play_siren_sound, save_capture_snapshot
from CameraManager import CameraManager
from TrackerEngine import TrackerEngine

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class PoseApp:
    def __init__(self, window_title=" निराक्षण - Niraakshan"):
        self.root = ctk.CTk()
        self.root.title(window_title)
        self.root.geometry("1800x1000")
        
        self.current_language = "Hindi"
        self.translations = self.load_translations()
        self.current_trans = self.translations["Hindi"]
        
        # Initialize Modules
        self.camera_manager = CameraManager()
        self.tracker_engine = TrackerEngine()
        
        self.is_running = False
        self.is_logging = False
        self.is_pro_mode = False
        self.is_alert_mode = False
        self.is_stillness_alert = False
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.target_map = {}
        self.selected_target_names = []
        
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        self.setup_ui()
        self.load_targets()
        
        self.root.protocol("WM_DELETE_WINDOW", self.graceful_exit)

    def setup_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        
        # Main Container
        self.main_container = ctk.CTkFrame(self.root, fg_color="#1a1a1a")
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Camera Controls
        self.camera_controls_panel = ctk.CTkFrame(self.main_container, fg_color="#2b2b2b", height=50, corner_radius=0)
        self.camera_controls_panel.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        
        controls_frame = ctk.CTkFrame(self.camera_controls_panel, fg_color="transparent")
        controls_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.btn_camera_toggle = ctk.CTkButton(controls_frame, text=self.current_trans.get("camera_on", "▶ Camera ON/OFF"), 
                                               command=self.toggle_camera, height=28, fg_color="#27ae60")
        self.btn_camera_toggle.pack(side="left", fill="both", expand=True, padx=2)
        
        self.btn_snap = ctk.CTkButton(controls_frame, text=self.current_trans.get("snap", "📸 Snap"), 
                                     command=self.snap_photo, height=28, fg_color="#f39c12")
        self.btn_snap.pack(side="left", fill="both", expand=True, padx=2)
        
        self.btn_pro_toggle = ctk.CTkButton(controls_frame, text=self.current_trans.get("pro_mode", "⚡ PRO Mode"), 
                                           command=self.toggle_pro_mode, height=28, fg_color="#8e44ad")
        self.btn_pro_toggle.pack(side="left", fill="both", expand=True, padx=2)
        
        # Video Display
        self.video_container = ctk.CTkFrame(self.main_container, fg_color="#000000")
        self.video_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.video_label = ctk.CTkLabel(self.video_container, text="Camera Offline", font=("Segoe UI", 20))
        self.video_label.pack(fill="both", expand=True)
        
        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self.root, fg_color="#2b2b2b", width=320)
        self.sidebar_frame.grid(row=0, column=1, sticky="nsew")
        
        # Sidebar Content
        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color="transparent")
        self.sidebar_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Guard Management
        self.grp_guard = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e")
        self.grp_guard.pack(fill="x", pady=5)
        ctk.CTkLabel(self.grp_guard, text="GUARDS", text_color="#16a085").pack(anchor="w", padx=5)
        
        ctk.CTkButton(self.grp_guard, text="Add Guard", command=self.add_guard_dialog, fg_color="#16a085").pack(fill="x", padx=5, pady=2)
        ctk.CTkButton(self.grp_guard, text="Remove Guard", command=self.remove_guard_dialog, fg_color="#c0392b").pack(fill="x", padx=5, pady=2)
        ctk.CTkButton(self.grp_guard, text="Select Guard", command=self.open_guard_selection_dialog, fg_color="#2980b9").pack(fill="x", padx=5, pady=2)
        
        # Alerts
        self.grp_alert = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e")
        self.grp_alert.pack(fill="x", pady=5)
        ctk.CTkLabel(self.grp_alert, text="ALERTS", text_color="#e67e22").pack(anchor="w", padx=5)
        
        self.btn_alert_toggle = ctk.CTkButton(self.grp_alert, text="Alert ON/OFF", command=self.toggle_alert_mode, fg_color="#7f8c8d")
        self.btn_alert_toggle.pack(fill="x", padx=5, pady=2)
        
        self.btn_stillness_alert = ctk.CTkButton(self.grp_alert, text="Stillness ON/OFF", command=self.toggle_stillness_alert, fg_color="#7f8c8d")
        self.btn_stillness_alert.pack(fill="x", padx=5, pady=2)
        
        # Performance
        self.perf_label = ctk.CTkLabel(self.sidebar_scroll, text="FPS: 0")
        self.perf_label.pack(pady=10)
        
        # Exit
        ctk.CTkButton(self.sidebar_scroll, text="Exit", command=self.graceful_exit, fg_color="#c0392b").pack(side="bottom", pady=20)

    def load_translations(self):
        # Simplified for brevity, can be expanded
        return {"Hindi": {}, "English": {}}

    def toggle_camera(self):
        if self.is_running:
            self.is_running = False
            self.camera_manager.stop()
            self.video_label.configure(image=None, text="Camera Offline")
            self.btn_camera_toggle.configure(fg_color="#27ae60")
        else:
            if self.camera_manager.start():
                self.is_running = True
                self.btn_camera_toggle.configure(fg_color="#c0392b")
                threading.Thread(target=self.process_video_loop, daemon=True).start()
            else:
                messagebox.showerror("Error", "Failed to start camera")

    def process_video_loop(self):
        while self.is_running:
            frame = self.camera_manager.get_frame()
            if frame is not None:
                self.frame_counter += 1
                
                # Process frame with TrackerEngine
                # We need to pass known faces. 
                # Optimization: Only update known faces when selection changes
                
                processed_frame = self.tracker_engine.process_frame(
                    frame, 
                    self.known_face_encodings, 
                    self.known_face_names,
                    process_this_frame=(self.frame_counter % 2 == 0) # Skip every other frame for performance
                )
                
                # Update UI
                # Convert to ImageTk
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                # Resize to fit video container
                w = self.video_container.winfo_width()
                h = self.video_container.winfo_height()
                if w > 0 and h > 0:
                    img = img.resize((w, h), Image.Resampling.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=imgtk, text="")
                self.video_label.image = imgtk # Keep reference
                
                # FPS Calculation
                if self.frame_counter % 30 == 0:
                    current_time = time.time()
                    self.current_fps = 30 / (current_time - self.last_fps_time)
                    self.last_fps_time = current_time
                    self.perf_label.configure(text=f"FPS: {self.current_fps:.1f}")
            
            else:
                time.sleep(0.01)

    def load_targets(self):
        self.target_map = {}
        guard_profiles_dir = CONFIG["storage"]["guard_profiles_dir"]
        if not os.path.exists(guard_profiles_dir):
            os.makedirs(guard_profiles_dir)
            
        target_files = glob.glob(os.path.join(guard_profiles_dir, "target_*.jpg"))
        for f in target_files:
            try:
                base_name = os.path.basename(f).replace(".jpg", "")
                parts = base_name.split('_')
                if len(parts) >= 3 and parts[-1] == "face":
                    display_name = " ".join(parts[1:-1])
                    self.target_map[display_name] = f
            except Exception as e:
                logger.error(f"Error parsing {f}: {e}")
        
        # Update known faces based on selection
        self.update_known_faces()

    def update_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
        for name in self.selected_target_names:
            if name in self.target_map:
                filepath = self.target_map[name]
                try:
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                except Exception as e:
                    logger.error(f"Error loading face for {name}: {e}")

    def open_guard_selection_dialog(self):
        # Simplified dialog
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Select Guards")
        dialog.geometry("400x500")
        
        scroll = ctk.CTkScrollableFrame(dialog)
        scroll.pack(fill="both", expand=True)
        
        vars = {}
        for name in self.target_map.keys():
            var = tk.BooleanVar(value=name in self.selected_target_names)
            vars[name] = var
            ctk.CTkCheckBox(scroll, text=name, variable=var).pack(anchor="w", padx=10, pady=5)
            
        def on_done():
            self.selected_target_names = [name for name, var in vars.items() if var.get()]
            self.update_known_faces()
            dialog.destroy()
            
        ctk.CTkButton(dialog, text="Done", command=on_done).pack(pady=10)

    def add_guard_dialog(self):
        filepath = filedialog.askopenfilename(title="Select Guard Image")
        if filepath:
            name = simpledialog.askstring("Guard Name", "Enter guard name:")
            if name:
                safe_name = name.strip().replace(" ", "_")
                target_path = os.path.join(CONFIG["storage"]["guard_profiles_dir"], f"target_{safe_name}_face.jpg")
                
                import shutil
                shutil.copy(filepath, target_path)
                self.load_targets()

    def remove_guard_dialog(self):
        # Implementation similar to original
        pass

    def toggle_pro_mode(self):
        self.is_pro_mode = not self.is_pro_mode
        self.btn_pro_toggle.configure(fg_color="#8e44ad" if not self.is_pro_mode else "#2ecc71")

    def toggle_alert_mode(self):
        self.is_alert_mode = not self.is_alert_mode
        self.btn_alert_toggle.configure(fg_color="#7f8c8d" if not self.is_alert_mode else "#e74c3c")

    def toggle_stillness_alert(self):
        self.is_stillness_alert = not self.is_stillness_alert
        self.tracker_engine.is_stillness_alert = self.is_stillness_alert # Pass to engine
        self.btn_stillness_alert.configure(fg_color="#7f8c8d" if not self.is_stillness_alert else "#e74c3c")

    def snap_photo(self):
        if self.is_running:
            frame = self.camera_manager.get_frame()
            if frame is not None:
                save_capture_snapshot(frame, "User_Snap")
                messagebox.showinfo("Snap", "Snapshot saved!")

    def graceful_exit(self):
        self.is_running = False
        self.camera_manager.stop()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    app = PoseApp()
    app.root.mainloop()
