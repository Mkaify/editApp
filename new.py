import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QFileDialog, QSlider, QProgressBar,
    QMenuBar, QAction, QColorDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sleek Image Processor")
        self.setFixedSize(1200, 700)
        self.setStyleSheet(self.get_light_theme())

        # Variables
        self.original = None
        self.processed = None
        self.history = []
        self.redo_stack = []
        self.drawing_enabled = False
        self.drawing_active = False
        self.last_point = None
        self.pen_color = Qt.red
        self.pen_size = 3
        self.drawing_pixmap = None
        self.is_live = False
        self.cap = None

        self.initUI()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        main_layout = QHBoxLayout()

        # Control Panel
        control_panel = QVBoxLayout()

        # Menu Bar
        menu_bar = QMenuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open", self)
        save_action = QAction("Save", self)
        open_action.triggered.connect(self.open_image)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

        theme_menu = menu_bar.addMenu("Theme")
        dark_action = QAction("Dark Mode", self)
        light_action = QAction("Light Mode", self)
        dark_action.triggered.connect(lambda: self.change_theme("dark"))
        light_action.triggered.connect(lambda: self.change_theme("light"))
        theme_menu.addAction(dark_action)
        theme_menu.addAction(light_action)
        self.setMenuBar(menu_bar)

        # Buttons
        self.open_btn = QPushButton("Open Image")
        self.open_btn.clicked.connect(self.open_image)
        self.grayscale_btn = QPushButton("Grayscale")
        self.grayscale_btn.clicked.connect(self.apply_grayscale)
        self.blur_btn = QPushButton("Gaussian Blur")
        self.blur_btn.clicked.connect(self.apply_gaussian_blur)
        self.edge_btn = QPushButton("Edge Detection")
        self.edge_btn.clicked.connect(self.apply_canny_edge)
        self.sharpen_btn = QPushButton("Sharpen")
        self.sharpen_btn.clicked.connect(self.apply_sharpen)
        self.reset_btn = QPushButton("Reset Image")
        self.reset_btn.clicked.connect(self.reset_image)
        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        self.start_live_btn = QPushButton("Start Live Preview")
        self.start_live_btn.clicked.connect(self.start_live_preview)
        self.stop_live_btn = QPushButton("Stop Live Preview")
        self.stop_live_btn.clicked.connect(self.stop_live_preview)
        self.drawing_btn = QPushButton("Enable Drawing Mode")
        self.drawing_btn.clicked.connect(self.toggle_drawing_mode)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_action)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo_action)

        # Sliders
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setMinimum(1)
        self.blur_slider.setMaximum(21)
        self.blur_slider.setValue(5)
        self.blur_slider.valueChanged.connect(self.apply_gaussian_blur)

        self.canny_slider1 = QSlider(Qt.Horizontal)
        self.canny_slider1.setMinimum(0)
        self.canny_slider1.setMaximum(255)
        self.canny_slider1.setValue(100)
        self.canny_slider1.valueChanged.connect(self.apply_canny_edge)

        self.canny_slider2 = QSlider(Qt.Horizontal)
        self.canny_slider2.setMinimum(0)
        self.canny_slider2.setMaximum(255)
        self.canny_slider2.setValue(200)
        self.canny_slider2.valueChanged.connect(self.apply_canny_edge)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(0)
        self.contrast_slider.setMaximum(200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Add to layout
        control_panel.addWidget(self.open_btn)
        control_panel.addWidget(self.grayscale_btn)
        control_panel.addWidget(self.blur_btn)
        control_panel.addWidget(self.edge_btn)
        control_panel.addWidget(self.sharpen_btn)
        control_panel.addWidget(self.reset_btn)
        control_panel.addWidget(self.save_btn)
        control_panel.addWidget(QLabel("Blur Intensity"))
        control_panel.addWidget(self.blur_slider)
        control_panel.addWidget(QLabel("Canny Threshold 1"))
        control_panel.addWidget(self.canny_slider1)
        control_panel.addWidget(QLabel("Canny Threshold 2"))
        control_panel.addWidget(self.canny_slider2)
        control_panel.addWidget(QLabel("Brightness"))
        control_panel.addWidget(self.brightness_slider)
        control_panel.addWidget(QLabel("Contrast"))
        control_panel.addWidget(self.contrast_slider)
        control_panel.addWidget(self.drawing_btn)
        control_panel.addWidget(self.start_live_btn)
        control_panel.addWidget(self.stop_live_btn)
        control_panel.addWidget(self.undo_btn)
        control_panel.addWidget(self.redo_btn)
        control_panel.addWidget(self.progress_bar)

        # Image Display
        self.image_label = QLabel("Drop Image Here or Use 'Open'")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; padding: 20px;")
        self.image_label.setFixedSize(800, 600)
        self.image_label.setScaledContents(True)
        self.image_label.setAcceptDrops(True)
        self.image_label.dragEnterEvent = self.dragEnterEvent
        self.image_label.dropEvent = self.dropEvent
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event

        # Assemble Layout
        main_layout.addLayout(control_panel, 1)
        main_layout.addWidget(self.image_label, 3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def convert_cv_to_qimage(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def update_image(self, img=None):
        if img is not None:
            self.processed = img.copy()
            self.history.append(img.copy())
            self.redo_stack.clear()

            # Always create drawing layer when new image is loaded
            qimg = self.convert_cv_to_qimage(self.processed)
            self.drawing_pixmap = QPixmap(qimg.size())
            self.drawing_pixmap.fill(Qt.transparent)

        if self.processed is not None:
            qimg = self.convert_cv_to_qimage(self.processed)
            base_pixmap = QPixmap.fromImage(qimg)

            final_pixmap = base_pixmap.copy()
            painter = QPainter(final_pixmap)
            if self.drawing_pixmap:
                painter.drawPixmap(0, 0, self.drawing_pixmap)
            painter.end()

            self.image_label.setPixmap(final_pixmap)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if path:
            self.progress_bar.setValue(0)
            self.timer_progress = QTimer()
            self.timer_progress.timeout.connect(lambda: self.load_with_progress(path))
            self.timer_progress.start(50)

    def load_with_progress(self, path):
        current = self.progress_bar.value()
        if current >= 100:
            self.timer_progress.stop()
            self.original = cv2.imread(path)
            self.processed = self.original.copy()
            self.history = [self.original.copy()]
            self.update_image()
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setValue(current + 5)

    def save_image(self):
        if self.processed is not None:
            self.progress_bar.setValue(0)
            self.timer_progress = QTimer()
            self.timer_progress.timeout.connect(self.save_with_progress)
            self.timer_progress.start(50)

    def save_with_progress(self):
        current = self.progress_bar.value()
        if current >= 100:
            self.timer_progress.stop()
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg)")
            if path:
                # Get merged image from label
                merged_pixmap = self.image_label.pixmap()
                if merged_pixmap:
                    merged_image = merged_pixmap.toImage()

                    # Convert QImage to OpenCV format (numpy array)
                    width = merged_image.width()
                    height = merged_image.height()
                    ptr = merged_image.bits()
                    ptr.setsize(height * width * 4)  # Ensure correct size
                    arr = np.frombuffer(ptr.asstring(), dtype=np.uint8).reshape((height, width, 4))  # RGBA

                    # Convert RGBA -> RGB -> BGR for OpenCV
                    rgb_image = arr[..., :3]
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                    # Save image
                    cv2.imwrite(path, bgr_image)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setValue(current + 5)

    def apply_grayscale(self):
        if self.processed is not None:
            gray = cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)
            self.update_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    def apply_gaussian_blur(self):
        if self.processed is not None:
            ksize = self.blur_slider.value()
            if ksize % 2 == 0:
                ksize += 1
            blurred = cv2.GaussianBlur(self.processed, (ksize, ksize), 0)
            self.update_image(blurred)

    def apply_canny_edge(self):
        if self.processed is not None:
            gray = cv2.cvtColor(self.processed, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.canny_slider1.value(), self.canny_slider2.value())
            self.update_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    def apply_sharpen(self):
        if self.processed is not None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(self.processed, -1, kernel)
            self.update_image(sharpened)

    def reset_image(self):
        if self.original is not None:
            self.processed = self.original.copy()
            self.drawing_pixmap = QPixmap(self.convert_cv_to_qimage(self.processed).size())
            self.drawing_pixmap.fill(Qt.transparent)
            self.update_image()

    def adjust_brightness(self):
        if self.processed is not None:
            val = self.brightness_slider.value()
            adjusted = cv2.convertScaleAbs(self.processed, alpha=1, beta=val)
            self.update_image(adjusted)

    def adjust_contrast(self):
        if self.processed is not None:
            val = self.contrast_slider.value() / 100.0
            adjusted = cv2.convertScaleAbs(self.processed, alpha=val, beta=0)
            self.update_image(adjusted)

    def start_live_preview(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not access webcam.")
            return        
        self.is_live = True
        self.timer.start(30)
        

    def stop_live_preview(self):
        self.is_live = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.drawing_pixmap = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (800, 600))
            frame = cv2.flip(frame, 1)
            self.original = frame
            self.processed = frame.copy()
            if self.drawing_pixmap is None or self.drawing_pixmap.size() != self.convert_cv_to_qimage(frame).size():
                self.drawing_pixmap = QPixmap(self.convert_cv_to_qimage(frame).size())
                self.drawing_pixmap.fill(Qt.transparent)
            self.update_image()

    def toggle_drawing_mode(self):
        self.drawing_enabled = not self.drawing_enabled
        self.drawing_btn.setText("Disable Drawing Mode" if self.drawing_enabled else "Enable Drawing Mode")

    def mouse_press_event(self, event):
        if self.drawing_enabled and event.button() == Qt.LeftButton:
            if self.drawing_pixmap is None or self.drawing_pixmap.isNull():
                return  # No drawing layer available

            label_pos = event.pos()
            pixmap_pos = self.map_label_to_pixmap(label_pos)
            self.drawing_active = True
            self.last_point = pixmap_pos

    def mouse_move_event(self, event):
        if self.drawing_enabled and self.drawing_active and self.last_point and self.drawing_pixmap:
            if self.drawing_pixmap.isNull():
                return

            label_pos = event.pos()
            pixmap_pos = self.map_label_to_pixmap(label_pos)

            painter = QPainter(self.drawing_pixmap)
            pen = QPen(self.pen_color, self.pen_size, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(self.last_point, pixmap_pos)
            painter.end()

            self.update_image()
            self.last_point = pixmap_pos

    def mouse_release_event(self, event):
        if self.drawing_enabled and event.button() == Qt.LeftButton:
            self.drawing_active = False
            self.last_point = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.original = cv2.imread(path)
            self.processed = self.original.copy()
            self.history = [self.original.copy()]
            self.update_image()

    def change_theme(self, mode):
        if mode == "dark":
            self.setStyleSheet(self.get_dark_theme())
        else:
            self.setStyleSheet(self.get_light_theme())

    def get_light_theme(self):
        return """
            background-color: #f0f0f0;
            color: black;
            QPushButton { background-color: #ddd; border: 1px solid #aaa; padding: 5px; margin: 2px; }
            QMenuBar { background-color: #ccc; color: black; }
        """

    def get_dark_theme(self):
        return """
            background-color: #2b2b2b;
            color: white;
            QPushButton { background-color: #444; border: 1px solid #666; padding: 5px; margin: 2px; }
            QMenuBar { background-color: #333; color: white; }
        """

    def map_label_to_pixmap(self, label_pos):
        if self.drawing_pixmap is None or self.drawing_pixmap.isNull():
            return QPoint(0, 0)

        label_rect = self.image_label.rect()
        pixmap_size = self.drawing_pixmap.size()

        scale_x = pixmap_size.width() / label_rect.width()
        scale_y = pixmap_size.height() / label_rect.height()

        x = int(label_pos.x() * scale_x)
        y = int(label_pos.y() * scale_y)
        return QPoint(x, y)

    def undo_action(self):
        if len(self.history) > 1:
            # Move current image to redo stack
            self.redo_stack.append(self.history.pop())
            self.processed = self.history[-1].copy()
            self.update_image()

    def redo_action(self):
        if self.redo_stack:
            restored = self.redo_stack.pop()
            self.history.append(restored)
            self.processed = restored.copy()
            self.update_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())