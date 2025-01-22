from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, 
                            QVBoxLayout, QWidget, QDialog, QHBoxLayout, QSlider, 
                            QDoubleSpinBox, QFileDialog, QSizePolicy, QCheckBox)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal, QTimer
import main
import numpy as np
import cv2
import sys
import view_synthesis

def cv2_to_pixmap(cv_img):
    # Convert from BGR to RGB
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width, channel = rgb_image.shape
    bytes_per_line = channel * width
    
    # Create QImage from numpy array
    q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    # Convert QImage to QPixmap
    return QPixmap.fromImage(q_image)


class Worker(QThread):
    finished = Signal(str)
    result_image = Signal(np.ndarray)


    def __init__(self, imgL, imgR, disparityLR, disparityRL, alpha=0.5):
        super().__init__()
        self.imgL = imgL
        self.imgR = imgR
        self.disparityLR = disparityLR
        self.disparityRL = disparityRL
        self.alpha = alpha
        self.is_running = True

    def run(self):
        self.is_running = True

        try:
            if self.is_running:
                imgI = view_synthesis.create_intermediate_view(self.imgL, self.imgR, self.disparityLR, self.disparityRL, self.alpha)[0]
                print("Create new VIEW!")
                if self.is_running:
                    self.result_image.emit(imgI)
                    # self.is_running = False
        except Exception as e:
            print("Error in worker thread")

    def stop(self):
        self.is_running = False

    def value(self):
        return self.alpha

class ScalableQLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.original_pixmap = None

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        scaled_pixmap = self.original_pixmap.scaled(self.size(), 
                                                  Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)
        super().setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(self.size(),
                                                      Qt.KeepAspectRatio,
                                                      Qt.SmoothTransformation)
            super().setPixmap(scaled_pixmap)

class IndividualWindow(QWidget):
    def __init__(self, name):
        super().__init__()
        self.model = name
        self.initUI()
        self.worker = None
        self.intermediate_images: dict[float, np.ndarray] = {}
        self.is_vertical = False

        self.imgL = None  
        self.imgR = None 
        self.disparityLR = None 
        self.disparityRL = None

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.create_intermediate_view)


    def initUI(self):
        # Main layout
        self.main_layout = QVBoxLayout()

        # title
        title_label = QLabel(self.model)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Content layout (for image and controls)
        self.content_layout = QVBoxLayout()

        # image label
        self.image_label = ScalableQLabel()
        self.image_label.setMinimumSize(300, 300)  # Set minimum size
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Create placeholder gray image
        placeholder = QPixmap(300, 300)
        placeholder.fill(Qt.lightGray)
        self.image_label.setPixmap(placeholder)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Slider layout (horizontal layout for slider and spinbox)
        self.controls_layout = QHBoxLayout()

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(50)  # 50 steps for 0 to 1 with 0.02 step
        self.slider.setValue(25)  # Default value 0.5

        # Spinbox for precise value
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setMinimum(0.0)
        self.spinbox.setMaximum(1.0)
        self.spinbox.setSingleStep(0.02)
        self.spinbox.setValue(0.5)

        # Connect slider and spinbox
        self.slider.valueChanged.connect(self.slider_changed)
        self.spinbox.valueChanged.connect(self.spinbox_changed)
        
        # Add widgets to slider layout
        self.controls_layout.addWidget(self.slider)
        self.controls_layout.addWidget(self.spinbox)

        self.content_layout.addWidget(self.image_label, stretch=1)
        self.content_layout.addLayout(self.controls_layout)
        
        self.main_layout.addWidget(title_label)
        self.main_layout.addLayout(self.content_layout)
        
        self.setLayout(self.main_layout)
        self.setWindowTitle("Intermediate View synthesis")

    
    def set_slider_orientation(self, is_vertical):
        self.is_vertical = is_vertical
        # Remove old content layout
        old_content = self.main_layout.takeAt(1)
        if old_content:
            old_content.layout().setParent(None)
            
        # Create new content layout based on orientation
        if is_vertical:
            self.content_layout = QHBoxLayout()
            self.controls_layout = QVBoxLayout()
            self.slider.setOrientation(Qt.Vertical)
        else:
            self.content_layout = QVBoxLayout()
            self.controls_layout = QHBoxLayout()
            self.slider.setOrientation(Qt.Horizontal)
            
        # Reconstruct layouts
        self.controls_layout.addWidget(self.slider)
        self.controls_layout.addWidget(self.spinbox)
        
        self.content_layout.addWidget(self.image_label)
        if is_vertical:
            self.content_layout.addLayout(self.controls_layout)
        else:
            self.content_layout.addLayout(self.controls_layout)
            
        self.main_layout.addLayout(self.content_layout)


    def slider_changed(self, value):
        spinbox_value = value / 50
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(spinbox_value)
        self.spinbox.blockSignals(False)
        self.update_timer.stop()
        self.update_timer.start(100)
        print("Value changed " + str(value))

    def spinbox_changed(self, value):
        # Convert spinbox value (0-1) to slider value (0-50)
        slider_value = int(value * 50)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)

    def setImages(self, imgL, imgR):
        if(imgL is None or imgR is None):
            self.imgL = None
            self.imgR = None
            self.intermediate_images.clear()
        self.intermediate_images.clear()

        self.imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        self.imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
        if(self.is_vertical):
            self.imgL = cv2.rotate(self.imgL, cv2.ROTATE_90_CLOCKWISE)
            self.imgR = cv2.rotate(self.imgR, cv2.ROTATE_90_CLOCKWISE)

        self.disparityLR, self.disparityRL = main.calculate_disparity(self.imgL, self.imgR, self.model)

        if(self.disparityLR is not None and self.disparityRL is not None):
            self.create_intermediate_view()

    def create_intermediate_view(self):
        if(self.imgL is None or self.imgR is None or self.disparityLR is None or self.disparityRL is None):
            return
        result = self.intermediate_images.get(self.spinbox.value())
        if result is not None:
            self.handle_image_result(result, False)
            return
        
        if self.worker and self.worker.value() == self.spinbox.value():
            return
        
        if self.worker and self.worker.isRunning():
            # print("Stopping worker")
            return
            # self.worker.wait()

        self.worker = Worker(self.imgL, self.imgR, self.disparityLR, self.disparityRL, self.spinbox.value())
        self.worker.result_image.connect(self.handle_image_result)
        self.worker.start()

    def handle_image_result(self, cv_img, new_img = True):
        if(self.is_vertical and new_img):
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.intermediate_images[self.spinbox.value()] = cv_img.copy()
        pixmap = cv2_to_pixmap(cv_img)
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        self.intermediate_images.clear()
        return super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.IgevWindow = IndividualWindow('IGEV')
        self.IgevWindow.show()
        self.RaftWindow = IndividualWindow('RAFT')
        self.RaftWindow.show()
        self.initUI()
        self.is_vertical = False

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create horizontal layout for image sections
        images_layout = QHBoxLayout()
        
        # Left image section
        left_section = QVBoxLayout()
        self.left_button = QPushButton("Left Image")
        self.left_button.clicked.connect(self.load_left_image)
        self.left_image = QLabel()
        self.left_image.setMinimumSize(300, 300)
        self.left_image.setAlignment(Qt.AlignCenter)
        
        # Right image section
        right_section = QVBoxLayout()
        self.right_button = QPushButton("Right Image")
        self.right_button.clicked.connect(self.load_right_image)
        self.right_image = QLabel()
        self.right_image.setMinimumSize(300, 300)
        self.right_image.setAlignment(Qt.AlignCenter)
        
        # Set placeholder images
        placeholder = QPixmap(300, 300)
        placeholder.fill(Qt.lightGray)
        self.left_image.setPixmap(placeholder)
        self.right_image.setPixmap(placeholder)
        
        # Add widgets to sections
        left_section.addWidget(self.left_button)
        left_section.addWidget(self.left_image)
        right_section.addWidget(self.right_button)
        right_section.addWidget(self.right_image)
        
        # Add sections to images layout
        images_layout.addLayout(left_section)
        images_layout.addLayout(right_section)
        
        # Control Section
        control_layout = QHBoxLayout()

        self.orientation_checkbox = QCheckBox('Top and Bottom Images')
        self.orientation_checkbox.stateChanged.connect(self.toggle_orientation)
        
        self.publish_button = QPushButton('Publish')
        self.publish_button.clicked.connect(self.publishImage)
        
        control_layout.addWidget(self.orientation_checkbox)
        control_layout.addWidget(self.publish_button)
        
        # Add everything to main layout
        main_layout.addLayout(images_layout)
        main_layout.addLayout(control_layout)
        
        self.setWindowTitle("Main Window")

    def load_left_image(self):
        self.imgL = self.openImage()
        if (self.imgL is not None):
            pixmap = cv2_to_pixmap(self.imgL)
            scaled_pixmap = pixmap.scaled(self.left_image.size(),
                                        Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
            self.left_image.setPixmap(scaled_pixmap)

    def load_right_image(self):
        self.imgR = self.openImage()
        if (self.imgR is not None):
            pixmap = cv2_to_pixmap(self.imgR)
            scaled_pixmap = pixmap.scaled(self.right_image.size(),
                                        Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
            self.right_image.setPixmap(scaled_pixmap)

    def openImage(self):
        file_name, _ = QFileDialog.getOpenFileName(
        self,
        "Open Image",
        "",
        "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if file_name:
            img = cv2.imread(file_name)
            img = cv2.resize(img, None, fx=1/3, fy=1/3)
            return img
        return None
    
    def publishImage(self):
        if (self.imgL is not None and self.imgR is not None):
            self.IgevWindow.setImages(self.imgL, self.imgR)
            self.RaftWindow.setImages(self.imgL, self.imgR)

    def toggle_orientation(self, state):
        print("Hello")
        # Update button text
        if state == Qt.CheckState.Checked.value:
            self.left_button.setText("Bottom Camera")
            self.right_button.setText("Top Camera")
        else:
            self.left_button.setText("Left Image")
            self.right_button.setText("Right Image")
            
        # Update slider orientation in individual windows if images have been published
        self.IgevWindow.set_slider_orientation(state == Qt.CheckState.Checked.value)
        self.RaftWindow.set_slider_orientation(state == Qt.CheckState.Checked.value)

    def closeEvent(self, event):
        self.IgevWindow.close()
        self.RaftWindow.close()
        return super().closeEvent(event)
    

if __name__ == "__main__":
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()
    app.exec()