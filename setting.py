import sys,os
from PySide6.QtCore import (Qt,QFile, QTextStream, Signal, QDir,QThread,QEvent,QPoint,QCoreApplication)
from PySide6.QtGui import (QColor, QFont,QPixmap, QMovie)
from PySide6.QtWidgets import (QApplication,QSlider, QTableWidget, QCheckBox, QDialog, QMessageBox, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget,QPushButton, QFileDialog, QComboBox,QTextEdit)
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
from PySide6 import QtCore,QtGui,QtWidgets
from qtrangeslider import QDoubleRangeSlider
from sidebar_ui import NumberOnlyTextEdit

class System_setting(object):
    def __init__(self, parent=None):
        #self.app = QtWidgets.QApplication(sys.argv)
        file_name = "saved_setting.txt"
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, file_name)
        font = QtGui.QFont('Arial', 10)
        self.Edit_aline = NumberOnlyTextEdit()
        self.Edit_aline.setFixedSize(QtCore.QSize(80, 29))
        self.Edit_aline.setObjectName("Edit_aline")

        self.Edit_depth = NumberOnlyTextEdit()
        self.Edit_depth.setFixedSize(QtCore.QSize(80, 29))
        self.Edit_depth.setObjectName("Edit_depth")

        self.Edit_upsample = NumberOnlyTextEdit()
        self.Edit_upsample.setFixedSize(QtCore.QSize(80, 29))
        self.Edit_upsample.setObjectName("Edit_upsample")

        self.Edit_num_chunk = NumberOnlyTextEdit()
        self.Edit_num_chunk.setFixedSize(QtCore.QSize(80, 29))
        self.Edit_num_chunk.setObjectName("Edit_num_chunk")

        aline_label = QtWidgets.QLabel('A-lines to GPU:')
        aline_label.setFont(font)
        depth_label = QtWidgets.QLabel('Pixel depth:')
        depth_label.setFont(font)
        upsample_label = QtWidgets.QLabel('Upsample factor:')
        upsample_label.setFont(font)
        chunks_label = QtWidgets.QLabel('Number of chunks for balancing:')
        chunks_label.setFont(font)
        keep_in_ram_label = QLabel("Keep result in RAM:")
        keep_in_ram_label.setFont(font)
        self.check_in_ram = QCheckBox()

        envelope = QHBoxLayout()
        #Envelope
        envelope_text = QLabel("Envelope extration:")
        self.enable_extraction = QCheckBox("Enable")
        self.enable_extraction.setChecked(False)
        envelope.addWidget(envelope_text)
        envelope.addWidget(self.enable_extraction)
        
        try:
            # Attempt to open the file for reading
            with open(file_path, 'r') as file:
                content = file.read()
                self.Edit_aline.setText(content.split(",")[0])
                self.Edit_depth.setText(content.split(",")[1])
                self.Edit_upsample.setText(content.split(",")[2])
                self.Edit_num_chunk.setText(content.split(",")[3])
                self.enable_extraction.setChecked(True)

        except FileNotFoundError:
            self.Edit_aline.setText("2048")
            self.Edit_depth.setText("1024")
            self.Edit_upsample.setText("6")
            self.Edit_num_chunk.setText("32")

        self.update_btn = QtWidgets.QPushButton("Update Settings")
        self.update_btn.setStyleSheet("color: white")
        self.update_btn.setCheckable(False)
        self.update_btn.setObjectName("update_btn")
        self.update_btn.clicked.connect(self.update_settings)

        spacerItem0 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        spacerItem1 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        spacerItem2 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        
        self.horizontalLayout0 = QtWidgets.QHBoxLayout()
        self.horizontalLayout0.addWidget(aline_label)
        self.horizontalLayout0.addWidget(self.Edit_aline)
        self.horizontalLayout0.addItem(spacerItem0)
        #spacerItem6_circ = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        #self.horizontalLayout.addItem(spacerItem6_circ)
        self.horizontalLayout1 = QtWidgets.QHBoxLayout()
        self.horizontalLayout1.addWidget(depth_label)
        self.horizontalLayout1.addWidget(self.Edit_depth)
        self.horizontalLayout1.addItem(spacerItem1)
        self.horizontalLayout2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout2.addWidget(upsample_label)
        self.horizontalLayout2.addWidget(self.Edit_upsample)
        self.horizontalLayout2.addItem(spacerItem2)
        self.horizontalLayout5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout5.addWidget(chunks_label)
        self.horizontalLayout5.addWidget(self.Edit_num_chunk)
        self.horizontalLayout5.addItem(spacerItem2)
        self.horizontalLayout3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout4.addWidget(keep_in_ram_label)
        self.horizontalLayout4.addWidget(self.check_in_ram)
        self.horizontalLayout4.addItem(spacerItem2)
        self.horizontalLayout6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout6.addWidget(self.update_btn)

        self.z_depth = QtWidgets.QLabel("Set Z range")
        self.z_depth.setObjectName("label")
        self.z_depth.setFont(font)
        #self.slider.valueChanged.connect(self.update_z_range)
        
        self.layout = QtWidgets.QVBoxLayout()  # Create a layout
        
        self.layout.addLayout(self.horizontalLayout0)
        self.layout.addLayout(self.horizontalLayout1)
        self.layout.addLayout(self.horizontalLayout2)
        self.layout.addLayout(self.horizontalLayout5)
        self.layout.addLayout(self.horizontalLayout3)
        self.layout.addLayout(self.horizontalLayout4)
        self.layout.addLayout(envelope)
        self.layout.addLayout(self.horizontalLayout6)
        #self.layout.addWidget(self.slider)  # Add slider to layout
        self.widget = QtWidgets.QWidget()  # Create a widget
        self.widget.setLayout(self.layout)  # Set layout for the widget
        self.widget.setWindowTitle('Processing Settings')
        self.widget.setGeometry(110, 110, 300, 300)
        self.widget.setStyleSheet("background-color: rgb(50, 50, 50);color: rgb(255,255,255);")
        #self.widget.show()  # Show the widget

    def get_selected_option(self):
        return self.dropdown.currentText()
    
    def update_settings(self):
        self.a_line_to_gpu = int(self.Edit_aline.toPlainText())
        self.depth = int(self.Edit_depth.toPlainText())
        self.upsample_factor = int(self.Edit_upsample.toPlainText())
        self.num_check = int(self.Edit_num_chunk.toPlainText())
        
        self.b_scan_preview_average = int(self.Edit_b_scan_avg_preview.toPlainText())
        file_name = "saved_setting.txt"
        data = str(self.a_line_to_gpu) + "," + str(self.depth) + "," + str(self.upsample_factor)+","+str(self.num_check)+","+str(self.preview_mode)+","+str(self.dispersion_coef)+","+str(self.b_scan_preview_average)
        with open(file_name, 'w') as file:
            file.write(data)
        #self.upsample_factor = int(self.Edit_upsample.toPlainText())
        return
    