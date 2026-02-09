import sys,os
from PySide6.QtCore import (Qt,QFile, QTextStream, Signal, QDir,QThread,QEvent,QPoint,QCoreApplication)
from PySide6.QtGui import (QColor, QFont,QPixmap, QIntValidator)
from PySide6.QtWidgets import (QApplication,QSlider, QLineEdit, QRadioButton, QDialog, QMessageBox, QLabel, QCheckBox, QHBoxLayout, QVBoxLayout, QWidget,QPushButton, QFileDialog, QComboBox,QTextEdit)


class OpenFileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        cur_dir = QDir.currentPath()
        
        layout = QVBoxLayout()
        buttons = QHBoxLayout()
        detector = QHBoxLayout()
        bit_depth = QHBoxLayout()
        scan = QHBoxLayout()
        res_axis_layout = QHBoxLayout()
        aline_per_b = QHBoxLayout()
        bscan_per_v = QHBoxLayout()
        bscan_rep = QHBoxLayout()
        volume_rep = QHBoxLayout()
        scan_range_x = QHBoxLayout()
        scan_range_y = QHBoxLayout()
        self.resize(330, 450)
        self.setWindowTitle("Open file")
        
        validator = QIntValidator(1, 100000)
        # Create the dropdown menu

        # File selection
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("Select a file...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.open_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.path_edit)
        file_layout.addWidget(browse_btn)

        # Detector type
        detector_text = QLabel("Detector:")
        single_rb = QRadioButton("Single")
        self.balanced_rb = QRadioButton("Balanced")
        self.balanced_rb.setChecked(True)
        detector.addWidget(detector_text)
        detector.addWidget(single_rb)
        detector.addWidget(self.balanced_rb)

        #pixel map
        pix_map_label = QLabel("Pixel Map:")
        self.path_edit_pixmap = QLineEdit(self)
        pix_browse_btn = QPushButton("Browse")
        pix_browse_btn.clicked.connect(self.open_pixmap_file)
        pixmap_layout = QHBoxLayout()
        pixmap_layout.addWidget(pix_map_label)
        pixmap_layout.addWidget(self.path_edit_pixmap)
        pixmap_layout.addWidget(pix_browse_btn)
        self.pixmap_widgets = [
            pix_map_label,
            self.path_edit_pixmap,
            pix_browse_btn
        ]
        self.balanced_rb.toggled.connect(self.update_pixmap_enabled)
        self.update_pixmap_enabled(self.balanced_rb.isChecked())

        pixmap_txt = os.path.join(cur_dir, "pixelmap.txt")

        if os.path.exists(pixmap_txt):
            with open(pixmap_txt, "r") as f:
                self.path_edit_pixmap.setText(f.read().strip())
        else:
            self.path_edit_pixmap.setPlaceholderText("Select a pixel map...")

        # Wavelength Map
        wave_label = QLabel("Wavelength:")
        self.wave_path_edit = QLineEdit(self)
        self.wave_path_edit.setPlaceholderText("Select a wavelength file...")
        wave_browse_btn = QPushButton("Browse")
        wave_browse_btn.clicked.connect(self.open_wave_file)
        wavefile_layout = QHBoxLayout()
        wavefile_layout.addWidget(wave_label)
        wavefile_layout.addWidget(self.wave_path_edit)
        wavefile_layout.addWidget(wave_browse_btn)

        # Bit depth selection
        bit_text = QLabel("Bit Depth:")
        self.bit_combo = QComboBox()
        self.bit_combo.addItems(["8","12","16"])
        self.bit_combo.setCurrentIndex(1)
        bit_depth.addWidget(bit_text)
        bit_depth.addWidget(self.bit_combo)

        #scan pattern
        scan_text = QLabel("Scan pattern:")
        uni_rb = QRadioButton("Uni")
        self.bi_rb = QRadioButton("Bi")
        self.bi_rb.setChecked(True)
        scan.addWidget(scan_text)
        scan.addWidget(uni_rb)
        scan.addWidget(self.bi_rb)

        

        #res axis
        axis_text = QLabel("Samples per A-scan")
        self.axis_input = QLineEdit("2048")
        self.axis_input.setValidator(validator)
        res_axis_layout.addWidget(axis_text)
        res_axis_layout.addWidget(self.axis_input)

        #aline
        aline_text = QLabel("A-scans per B-scan")
        self.aline_input = QLineEdit("512")
        self.aline_input.setValidator(validator)
        aline_per_b.addWidget(aline_text)
        aline_per_b.addWidget(self.aline_input)

        #bscan
        bscan_text = QLabel("B-scan per volume")
        self.bscan_input = QLineEdit("512")
        self.bscan_input.setValidator(validator)
        bscan_per_v.addWidget(bscan_text)
        bscan_per_v.addWidget(self.bscan_input)

        #bscan
        bscan_rep_text = QLabel("Repeated B-scan")
        self.bscan_rep_input = QLineEdit("1")
        self.bscan_rep_input.setValidator(validator)
        bscan_rep.addWidget(bscan_rep_text)
        bscan_rep.addWidget(self.bscan_rep_input)
        
        #volume
        volume_text = QLabel("Repeated Volume")
        self.volume_input = QLineEdit("1")
        self.volume_input.setValidator(validator)
        volume_rep.addWidget(volume_text)
        volume_rep.addWidget(self.volume_input)

        #scan_range_x
        scan_range_x_text = QLabel("Scan range X[mm]")
        self.scan_range_x_input = QLineEdit("2")
        self.scan_range_x_input.setValidator(validator)
        scan_range_x.addWidget(scan_range_x_text)
        scan_range_x.addWidget(self.scan_range_x_input)

        #scan_range_y
        scan_range_y_text = QLabel("Scan range Y[mm]")
        self.scan_range_y_input = QLineEdit("2")
        self.scan_range_y_input.setValidator(validator)
        scan_range_y.addWidget(scan_range_y_text)
        scan_range_y.addWidget(self.scan_range_y_input)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(file_layout)
        layout.addLayout(detector)
        layout.addLayout(pixmap_layout)
        layout.addLayout(wavefile_layout)
        layout.addLayout(bit_depth)
        layout.addLayout(scan)
        layout.addLayout(res_axis_layout)
        layout.addLayout(aline_per_b)
        layout.addLayout(bscan_per_v)
        layout.addLayout(bscan_rep)
        layout.addLayout(volume_rep)
        layout.addLayout(scan_range_x)
        layout.addLayout(scan_range_y)
        layout.addLayout(buttons)

        self.setLayout(layout)

    def accept(self):
        # 🔹 YOUR custom behavior
        #print("OK clicked — validating / saving / emitting data")

        # example: validation
        if not self.path_edit.text():
            QMessageBox.warning(self, "Error", "Path cannot be empty")
            return   # ❌ prevent dialog from closing

        # example: save state
        self.selected_path = self.path_edit.text()

        # ✅ VERY IMPORTANT: call super()
        super().accept()

    def update_pixmap_enabled(self, checked):
        for w in self.pixmap_widgets:
            w.setEnabled(checked)


    def get_selected_option(self):
        return self.dropdown.currentText()
    
    def open_file(self):        
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setNameFilter("*.raw")

        # if  self.frame_3D_linear.any() or self.frame_3D_20log.any() or self.frame_3D_resh.any():
        #     print("one of them not none")

        filenames = QComboBox()
        if dlg.exec():
            filenames = dlg.selectedFiles()
			
            self.path_edit.setText(filenames[0])

    def open_pixmap_file(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)

        filenames = QComboBox()
        if dlg.exec():
            filenames = dlg.selectedFiles()
            self.path_edit_pixmap.setText(filenames[0])

    def open_wave_file(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)

        filenames = QComboBox()
        if dlg.exec():
            filenames = dlg.selectedFiles()
            self.wave_path_edit.setText(filenames[0])
            


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = OpenFileDialog()
    window.show()

    sys.exit(app.exec())