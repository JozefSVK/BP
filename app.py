import sys
from PySide6.QtWidgets import *
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Application")
        self.resize(1000, 600)

        # Create central widget and main vertical layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top section - Stereo matching algorithm GroupBox
        algo_group = QGroupBox("Stereo matching algorithm")
        algo_layout = QHBoxLayout(algo_group)
        algo_layout.setAlignment(Qt.AlignCenter)
        
        combo1 = QComboBox()
        combo2 = QComboBox()
        combo1.addItems(["Select Option 1", "Option 1A", "Option 1B", "Option 1C"])
        combo2.addItems(["Select Option 2", "Option 2A", "Option 2B", "Option 2C"])
        
        # Set minimum width for comboboxes
        combo1.setMinimumWidth(200)
        combo2.setMinimumWidth(200)
        
        algo_layout.addWidget(combo1)
        algo_layout.addSpacing(20)
        algo_layout.addWidget(combo2)

        # Middle section - Left and right images GroupBox
        images_group = QGroupBox("Left and right images")
        images_layout = QHBoxLayout(images_group)
        images_layout.setAlignment(Qt.AlignCenter)

        # Create two identical image groups
        for _ in range(2):
            group_widget = QWidget()
            group_layout = QVBoxLayout(group_widget)
            
            # Image label with scaling
            image_label = QLabel()
            image_label.setMinimumSize(400, 300)
            image_label.setScaledContents(True)
            
            # Create placeholder image
            pixmap = QPixmap(400, 300)
            pixmap.fill(Qt.lightGray)
            image_label.setPixmap(pixmap)
            
            # Add size policy for proper scaling
            size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            image_label.setSizePolicy(size_policy)
            
            # Buttons under image
            button_widget = QWidget()
            button_layout = QHBoxLayout(button_widget)
            button_layout.setAlignment(Qt.AlignCenter)
            
            open_button = QPushButton("Open")
            swap_button = QPushButton("Swap")
            
            button_layout.addWidget(open_button)
            button_layout.addWidget(swap_button)
            
            # Add to group layout
            group_layout.addWidget(image_label)
            group_layout.addWidget(button_widget)
            
            # Add group to middle layout
            images_layout.addWidget(group_widget)
        
        images_layout.setSpacing(40)  # Space between image groups

        # Bottom section - Navigation buttons
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setAlignment(Qt.AlignRight)
        
        back_button = QPushButton("Back")
        next_button = QPushButton("Next")
        
        bottom_layout.addWidget(back_button)
        bottom_layout.addWidget(next_button)

        # Add all sections to main layout
        main_layout.addWidget(algo_group)
        main_layout.addWidget(images_group, 1)  # Give middle section stretch priority
        main_layout.addWidget(bottom_widget)
        
        # Set spacing between main sections
        main_layout.setSpacing(20)

        # Set content margins
        main_layout.setContentsMargins(20, 20, 20, 20)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())