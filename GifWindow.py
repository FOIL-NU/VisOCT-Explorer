from PySide6.QtCore import (Qt)
from PySide6.QtGui import (QMovie)
from PySide6.QtWidgets import (QLabel, QMainWindow)

class GifWindow(QMainWindow):
    def __init__(self, gif_path):
        super().__init__()
        self.setFixedSize(500, 700)
        self.setWindowTitle("GIF Viewer")
        self.setGeometry(100, 100, 100, 100)

        # Create a QLabel for displaying the GIF
        gif_label = QLabel(self)
        gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(gif_label)

        # Load and display the GIF using QMovie
        movie = QMovie(gif_path)
        gif_label.setMovie(movie)
        movie.start()