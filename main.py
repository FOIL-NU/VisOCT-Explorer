import sys,os
from PySide6.QtCore import (Qt,QFile, QTextStream, Signal, QDir,QThread,QEvent,QPoint,QCoreApplication)
from PySide6.QtGui import (QColor, QFont,QPixmap, QMovie)
from PySide6.QtWidgets import (QApplication,QSlider, QTableWidget, QTableWidgetItem, QDialog, QMessageBox, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget,QPushButton, QFileDialog, QComboBox,QTextEdit)

import psutil
import openpyxl
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

from Saving_thread import SavingThread
from dm_tab import Distance_Measurement_Tab
from balancefringe import fringes
from processParams import processParams
from resample import linear_k
from resampleThread import ResampleThread
from skimage import exposure
from processingThread import ProcessingThread
from TSAThread import TSAThread
from dispersion import *
from GifWindow import *
from volumerender import *
from enface_3D import *
from fibergram import *
from setting import *
from open_file import OpenFileDialog
from glass_flatten import *
from UpdateVolume import *

import numpy as np


from PIL import Image,ImageTk
from octFuncs import *
from random import randint
from sidebar_ui import Ui_MainWindow


class MyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        buttons = QHBoxLayout()
        self.resize(300, 150)
        self.setWindowTitle("Save to file")
        # Create the dropdown menu
        self.dropdown = QComboBox(self)
        if parent.processParameters.newSRFlag or parent.processParameters.res_fast == 8192:
            self.dropdown.addItem("Save speckle reduced images")
            layout.addWidget(self.dropdown)
        else:
            self.dropdown.addItem("Average 32 frames and save")
            self.dropdown.addItem("Average 16 frames and save")
            self.dropdown.addItem("Average 8 frames and save")
            self.dropdown.addItem("Average 4 frames and save")
            self.dropdown.addItem("Save as tiff stack")
            self.dropdown.addItem("Save as dicom")
            layout.addWidget(self.dropdown)

        
        ok_button = QPushButton("OK", self)
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        layout.addLayout(buttons)
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)

        self.setLayout(layout)

    def get_selected_option(self):
        return self.dropdown.currentText()

    


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.cur_img_slow_no = 0
        self.cur_img_fast_no = 0
        self.progressValue = 0
        self.setWindowTitle("VisOCT Metro")
        self.image_ready = False
        self.flip_bscan = False
        self.start_point_marked = False
        self.status = "Waiting for user to choose raw file"
        self._translate = QCoreApplication.translate

        self.frame_3D_linear = np.array([])
        self.frame_3D_20log = np.array([])
        self.frame_3D_resh = np.array([])
        self.system_setting = System_setting(self)
        self.ui.stackedWidget.setCurrentIndex(5)
        self.ui.home_btn.clicked.connect(self.go_home)
        self.ui.flip_btn.clicked.connect(self.flip)
        self.ui.set_note_btn.clicked.connect(self.set_note_toggle)
        self.ui.actionOpen_Raw.triggered.connect(self.open_file)
        self.ui.actionSystem_Setting.triggered.connect(self.open_system_setting)
        self.ui.open_file_btn.clicked.connect(self.open_file)

        self.ui.pixel_map_btn.clicked.connect(self.set_pixel_map)
        #self.ui.contrast_btn_2.clicked.connect(self.set_pixel_map)

        #self.ui.adaptive_btn_1.clicked.connect(self.adaptive_balance)
        #self.ui.adaptive_btn_2.clicked.connect(self.adaptive_balance)

        self.ui.octa_btn.clicked.connect(self.octa_window)
        self.ui.octa_btn.setEnabled(True)
        self.ui.batch_btn.clicked.connect(self.batch_processing)
        self.ui.fibergram_btn.clicked.connect(self.fibergram_window)
        self.ui.circ_btn.clicked.connect(self.circ_window)
        self.ui.enface_otherdim.clicked.connect(self.openEnface_3D)
        self.ui.pushButton_4.clicked.connect(self.openRendererWindow)
        self.enface_divider = 4
        
        self.plot_grid = False
        self.circ_grid_plotted = False
        self.dm_tab = None
        self.resample_process = ResampleThread(0,0,0,0,0)
        self.process = ProcessingThread()
        self.update_volume_process = UpdateVolumeThread()
        self.process.progress.connect(self.progress_callback)
        self.process.updateProgressBar.connect(self.updateProgress)
        self.process.image_ready.connect(self.image_ready_callback)
        self.ui.circ_resample_btn.clicked.connect(self.resample)
        self.ui.circ_save_raw_btn.setDisabled(True)
        self.resample_process.progress.connect(self.circ_progress_callback)
        self.resample_process.resample_ready.connect(self.resample_ready_callback)

        self.ui.tsa_btn.clicked.connect(self.TSA_processing)
        self.dialog = None
        self.open_file_dialog = OpenFileDialog()
        file_name = "saved_setting.txt"
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, file_name)
        try:
            # Attempt to open the file for reading
            with open(file_path, 'r') as file:
                content = file.read()
                
                self.preview_mode = False

        except FileNotFoundError:
            self.preview_mode = False

        self.update_volume_process = UpdateVolumeThread()
        self.update_volume_process.progress.connect(self.update_volume_progress_callback)
        self.update_volume_process.volume_ready.connect(self.update_volume_ready_callback)

        self.process = ProcessingThread()
        self.process.progress.connect(self.progress_callback)
        self.process.updateProgressBar.connect(self.updateProgress)
        self.process.image_ready.connect(self.image_ready_callback)


    ## Change QPushButton Checkable status when stackedWidget index changed
    def on_stackedWidget_currentChanged(self, index):
        btn_list = self.ui.icon_only_widget.findChildren(QPushButton) \
                    + self.ui.full_menu_widget.findChildren(QPushButton)
        
        for btn in btn_list:
            if index in [5, 6]:
                btn.setAutoExclusive(False)
                btn.setChecked(False)
            else:
                btn.setAutoExclusive(True)

    def open_system_setting(self):
        self.system_setting.widget.show()

    def set_rindex(self):
        index = float(self.ui.Refractive_index.toPlainText())
        self.ui.Slow_Axis.set_refractive_index(index)
        self.ui.textEdit_2.append("Refractive index has been set to:"+index)

    def run_flatten(self):
        self.ui.fibergram_seg_btn.setEnabled(False)
        self.frame_3D_resh_origin_linear = np.load(self.processParameters.fname.split('.RAW')[0]+'/frame_3D.npy')
        self.glass_flatten_process = glass_flatten(self.frame_3D_resh_origin_linear)
        self.glass_flatten_process.start()
        #self.glass_flatten_process.progress.connect(self.flatten_process_callback)
        self.glass_flatten_process.flatten_ready.connect(self.flatten_ready_callback)

    def flatten_ready_callback(self,flatten_volume):
        self.frame_3D_resh = np.float32(20*np.log10(flatten_volume))
        self.updateImg(self.ui.Enface,self.ui.Slow_Axis,self.oct_lowVal,self.oct_highVal,False)

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() == Qt.WindowMaximized:
                self.ui.Slow_Axis.onWindowSizeChange(self.ui.Slow_Axis.geometry())
            else:
                self.ui.Slow_Axis.onWindowSizeChange(self.ui.Slow_Axis.geometry())

    def TSA_processing(self):
        self.TSA = TSAThread(self.processParameters,self.frame_3D_resh_origin,self.frame_OCTA)
        self.TSA.volume_ready.connect(self.TSA_ready_callback)
        self.TSA.start()

    def batch_processing(self):
        cur_dir = QDir.currentPath()
        #with open(cur_dir+"/pixelmap.txt", 'r') as file:
                #match_Path = str(file.read().rstrip())
        
        self.ui.label.setText(self._translate("MainWindow", "Status: VisOCT Explorer(Selecting File...)"))
        print(cur_dir)
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setNameFilter("*.raw")

        filenames = QComboBox()
        if dlg.exec():
            
            filenames = dlg.selectedFiles()
            
            directory = os.path.dirname(filenames[0])
            self.f = filenames[0]
            print("filename:"+self.f)
            datasets = []
            extension = "1.RAW"
            for root, dirs, files in os.walk(directory):
                for file in files:
                    print(file)
                    if file.endswith(extension):
                        datasets.append(os.path.join(root, file))

            #print("matched_files:",datasets)
			
            self.ui.textEdit_2.setText("File "+self.f+" selected")
            if "_1.RAW" in self.f or "_2.RAW" in self.f:
                try:
                    with open(cur_dir+"/pixelmap.txt", 'r') as file:
                        self.match_Path = str(file.read().rstrip())
                            
                except (IOError,FileNotFoundError):
                    msg_box = QMessageBox()
                    msg_box.setIcon(QMessageBox.Icon.Warning)
                    msg_box.setWindowTitle("Warning")
                    msg_box.setText('Pixel map not selected yet.\nYou can select an existing pixel map to begin with.\nOr generate one for the current dataset?')
                    
                    select_button = QPushButton("Select existing pixel map")

                    select_button.clicked.connect(self.select_pixmap)

                    msg_box.addButton(select_button, QMessageBox.AcceptRole)
                    # Connect custom slots to button clicks
                    #select_button.clicked.connect(lambda: custom_button_handler(custom_button1))
                    #generate_button.clicked.connect(lambda: custom_button_handler(custom_button2))

                    msg_box.exec()
                    return
            else:
                self.match_Path = "None"
            #for dataset in datasets:
            #self.processParameters = processParams(32, datasets, self.match_Path)
            self.process = ProcessingThread(datasets,True,self.match_Path)
            self.process.finished.connect(self.process.deleteLater)
            self.process.progress.connect(self.progress_callback)
            self.process.updateProgressBar.connect(self.updateProgress)
            self.process.image_ready.connect(self.image_ready_callback)
            self.process.start()
    
    def pixelMap_ready(self,x,pixMap_name):
        self.ui.textEdit_2.verticalScrollBar().setValue(self.ui.textEdit_2.verticalScrollBar().maximum())
        self.pixelMap_ready = True
        pix_array = np.linspace(0,2048,2048)
        pixMap = np.polyval(x,pix_array)
        pixMap[pixMap<0] = 0
        pixMap[pixMap>2048] = 2048
        file_path = os.path.join(QDir.currentPath(), 'Pixel Maps', pixMap_name)
        with open(file_path,'wb') as file:
            pixMap.tofile(file)
        print(pixMap)

    def open_video(self):
        self.ui.label.setText(self._translate("MainWindow", "Status: VisOCT Explorer (B-scan Fly Through)"))
        self.gif_window = GifWindow(self.directory+'./Bscan_Flythru.gif')
        #self.gif_window = GifWindow(self.directory+'./Bscan_Flythru.gif')
        self.gif_window.show()

    def octa_window(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def fibergram_window(self):
        self.ui.fibergram_seg_btn.clicked.connect(self.run_fibergram_seg)
        self.threshold_Val = int(self.ui.Threshold_fibergram.toPlainText())
        self.ui.stackedWidget.setCurrentIndex(3)
    
    def run_fibergram_seg(self):
        self.ui.fibergram_seg_btn.setEnabled(False)
        self.frame_3D_resh_origin_linear = np.load(self.processParameters.fname.split('.RAW')[0]+'/frame_3D.npy')
        self.frame_3D_resh_origin_20log = 20*np.log10(self.frame_3D_resh_origin_linear)
        self.fibergram_process = FibergramThread(self.frame_3D_resh_origin_20log,self.frame_3D_resh_origin_linear,self.oct_lowVal,self.oct_highVal,self.threshold_Val)
        self.fibergram_process.start()
        self.fibergram_process.progress.connect(self.fibergram_progress_callback)
        self.fibergram_process.fibergram_ready.connect(self.fibergram_ready_callback)

    def fibergram_progress_callback(self,progress_percent):
        self.ui.progressBar_fibergram.setValue(progress_percent)

    def fibergram_ready_callback(self,fibergram_enface,fitted_top_boundary):
        self.fibergram_enface = fibergram_enface
        self.fitted_top_boundary = fitted_top_boundary
        self.fibergram_enface = imcontrast_adjust(self.fibergram_enface)
        #print(np.shape(self.fibergram))
        enface = pg.ImageItem(np.fliplr(np.flipud(self.fibergram_enface)))
        enface.setRect(0,0,1024,1024)
        fitted_slow_axis = pg.ImageItem(np.rot90(((np.clip((self.frame_3D_resh[:,:,int(self.frame_3D_resh.shape[2]/2)] - self.lowVal) / (self.highVal - self.lowVal), 0, 1)) * 255).astype(np.uint8)))
        fitted_slow_axis.setRect(0,0,1024,1024)
        x = np.arange(np.shape(self.frame_3D_resh)[1])*2
        y = np.flip(self.fitted_top_boundary[:,int(np.shape(self.frame_3D_resh_origin_linear)[2]/2)-10])
        self.ui.bscan_number_label_fibergram.setText(self._translate("MainWindow", "B-scan #: "+str(int(self.frame_3D_resh.shape[2]/2)*4)))
        self.ui.bscan_dim_label_fibergram.setText(self._translate("MainWindow", "B-scan dimension: (H)1024 x (W)"+str(self.processParameters.res_fast)))
        self.ui.actual_dim_label_fibergram.setText(self._translate("MainWindow", "Actual dimension: (H)1150 um x (W)"+str(self.processParameters.xrng)+" um"))
        self.ui.Fibergram_Enface.addItem(enface,clear=True)
        self.ui.Fibergram_Slow_Axis.addItem(fitted_slow_axis,clear=True)
        scatter = pg.ScatterPlotItem(x=x,y=y, size=2, pen='r')
        self.ui.Fibergram_Slow_Axis.addItem(scatter)
        del self.frame_3D_resh_origin_linear
        del self.frame_3D_resh_origin_20log
        self.ui.fibergram_seg_btn.setEnabled(True)

    #to be implemented: when to delete the frame_3D_resh_origin_20log
    def openRendererWindow(self):
        self.frame_3D_resh_origin_20log = 20*np.log10(np.load(self.processParameters.fname.split('.RAW')[0]+'/frame_3D.npy'))
        self.rendererWindow = Visualizer(self.frame_3D_resh_origin_20log,self.oct_lowVal,self.oct_highVal)
        del self.frame_3D_resh_origin_20log
        #self.rendererWindow.start()
        #self.rendererWindow.show()

    def openEnface_3D(self):
        self.frame_3D_resh_origin_20log = (np.load(self.processParameters.fname.split('.RAW')[0]+'/frame_3D.npy'))
        self.rendererWindow = enface_3D(self.frame_3D_resh_origin_20log,self.enface_pixmap)
        del self.frame_3D_resh_origin_20log

    def circ_window(self):
        self.ui.stackedWidget.setCurrentIndex(1)
        self.ui.horizontalSlider_01.valueChanged.connect(self.radius_changed)
        self.ui.Edit_circ_x.setText(str(int(self.x_center/1024*self.processParameters.xrng)))
        self.ui.Edit_circ_y.setText(str(int(self.y_center/1024*self.processParameters.yrng)))
        self.ui.Edit_circ_x.textChanged.connect(self.on_center_changed)
        self.ui.Edit_circ_y.textChanged.connect(self.on_center_changed)
        self.ui.Edit_radius.textChanged.connect(self.on_radius_text_changed)
        self.prev_x = self.ui.Edit_circ_x.toPlainText()
        self.prev_y = self.ui.Edit_circ_y.toPlainText()
        self.prev_r = self.ui.Edit_radius.toPlainText()
        if self.plot_grid:
            
            self.angle_grid([self.start_coord[0],1023-self.start_coord[1]],[self.center[0],1023-self.center[1]],self.r,self.processParameters.eye)

    def radius_changed(self):
        self.radius_given = self.ui.horizontalSlider_01.value()
        self.ui.Edit_radius.textChanged.disconnect(self.on_radius_text_changed)
        self.ui.Edit_radius.setText(f"{str(int(self.radius_given/1024*self.processParameters.xrng))}")
        self.resample_circ.setSize(2*self.radius_given)
        self.ui.Edit_radius.textChanged.connect(self.on_radius_text_changed)

    def on_radius_text_changed(self):
        new_r = self.ui.Edit_radius.toPlainText()
        self.ui.horizontalSlider_01.valueChanged.disconnect(self.radius_changed)
        if new_r == self.prev_r:
            return
        else:
            self.prev_r = new_r
            if new_r == '':
                self.radius_given = 0
                self.resample_circ.setSize(2*self.radius_given)
                self.ui.horizontalSlider_01.setValue(0)
            else:
                self.radius_given = int(float(new_r) * (1024/self.processParameters.xrng))
                self.resample_circ.setSize(2*self.radius_given)
                self.ui.horizontalSlider_01.setValue(self.radius_given)
        self.ui.horizontalSlider_01.valueChanged.connect(self.radius_changed)
        
    def on_center_changed(self):
        new_x = self.ui.Edit_circ_x.toPlainText()
        new_y = self.ui.Edit_circ_y.toPlainText()
        if new_x == self.prev_x and new_y == self.prev_y:
            return
        else:
            self.prev_x = new_x
            self.prev_y = new_y
            if new_x == '':
                self.x_center = 0
            else:
                self.x_center = int(float(new_x)*(1024/self.processParameters.xrng))
            
            if new_y == '':
                self.y_center = 0
            else:
                self.y_center = int((float(new_y))*(1024/self.processParameters.yrng))
            
            print("y_center",self.y_center)
            self.resample_circ.setData(x=[self.x_center], y=[self.y_center])
            self.resample_center.setData(x=[self.x_center], y=[self.y_center])
            
            if self.start_point_marked or self.circ_grid_plotted:
                self.start_point_marked = False
                self.circ_grid_plotted = False
                self.ui.Enface_circ.clear()
                self.ui.Enface_circ.addItem(self.circ_enface,clear = True)
                self.ui.Enface_circ.addItem(self.resample_circ)
                self.ui.Enface_circ.addItem(self.resample_center)

    def resample(self):
        self.ui.circ_resample_btn.setEnabled(False)
        center = np.array((self.x_center,1024-self.y_center))
        width = 12
        self.frame_3D_resh_origin = 20*np.log10(np.load(self.processParameters.fname.split('.RAW')[0]+'/frame_3D.npy'))
        print("frame_3D shape",np.shape(self.frame_3D_resh))
        if self.processParameters.res_fast == 2 * self.processParameters.res_slow:
            self.frame_3D = np.zeros((1024,self.processParameters.res_slow,self.processParameters.res_slow))
            for i in range(0,self.processParameters.res_slow):
                self.frame_3D[:,i,:] = np.mean(self.frame_3D_resh_origin[:,2*i:2*(i+1),:],axis=1)
        if self.processParameters.res_slow == 2 * self.processParameters.res_fast:
            self.frame_3D = np.zeros((1024,self.processParameters.res_fast,self.processParameters.res_fast))
            for i in range(0,self.processParameters.res_fast):
                self.frame_3D[:,:,i] = np.mean(self.frame_3D_resh_origin[:,:,2*i:2*(i+1)],axis=1)
        if self.processParameters.res_slow == self.processParameters.res_fast:
            self.frame_3D = self.frame_3D_resh_origin
        self.resample_process.update_resample_content(self.processParameters,self.frame_3D,self.radius_given,center,width)
        self.resample_process.start()

    def circ_progress_callback(self,prog_val):
        print(prog_val)
        self.ui.progressBar_circ.setValue(prog_val)
        
    def resample_ready_callback(self,frame_3D,start_coord,end_coord,center):
        self.ui.circ_Slow_Axis.clear()
        print(start_coord)
        self.circ_step = 0.1
        if frame_3D.shape[1] % 6 != 0:
        # Remove extra columns to make the number of columns divisible by 4
            frame_3D = frame_3D[:, :(frame_3D.shape[1] // 6) * 6]

        # Average every 4 columns
        self.frame_3D = np.mean(frame_3D.reshape(frame_3D.shape[0], -1, 6), axis=2)
        self.circ_lowVal = np.mean(self.frame_3D)+1
        self.circ_highVal = self.circ_lowVal + 2.25 * np.std(self.frame_3D)
        circ_img = ((np.clip((self.frame_3D - self.circ_lowVal) / (self.circ_highVal - self.circ_lowVal), 0, 1)) * 255).astype(np.uint8)
        
        if self.processParameters.eye == 0 or self.processParameters.eye == 3:
            circ_img = pg.ImageItem(np.rot90(circ_img))
        else:
            circ_img = pg.ImageItem(np.flipud(np.rot90(np.roll(circ_img,int(np.shape(circ_img)[1]/2),axis=1))))
        self.r = start_coord[0]-center[0]
        center[1] = 1023-center[1]
        start_coord[1] = 1023-start_coord[1]
        self.ui.Enface_circ.clear()
        self.ui.Enface_circ.addItem(self.circ_enface,clear = True)
        self.ui.Enface_circ.addItem(self.resample_circ)
        self.plot_grid = True
        self.start_coord = start_coord
        self.center = center
        self.angle_grid(start_coord,center,self.r,self.processParameters.eye)

        
        center[1] = 1023-center[1]
        start_coord[1] = 1023-start_coord[1]

        self.start_point_marked = True
        circ_img.setRect(0,0,int(np.shape(self.frame_3D)[1]*0.8),330)
        cmap = plt.get_cmap('gray')

        # Convert the colormap to a lookup table (LUT)
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        circ_img.setLookupTable(lut)
        self.ui.circ_Slow_Axis.setLimits(xMin=0, xMax=int(np.shape(self.frame_3D)[1]*0.8), yMin=0, yMax=330)
        self.ui.circ_Slow_Axis.addItem(circ_img, clear=True)
        self.ui.circ_Slow_Axis.setBscan_number(0)
        self.ui.circ_Slow_Axis.setImageSize(self.processParameters,self.ui.distance_circ,self.ui.circ_Slow_Axis.geometry(),np.shape(frame_3D)[1],self.r)

        self.ui.circ_Slow_Axis.setMeasurement_result_table(self.dm_tab)
        self.ui.reset_measure_circ.clicked.connect(self.reset_circ_measurement)
        self.reset_circ_measurement()
        self.ui.oct_contrast_Slider_circ.setEnabled(True)
        self.ui.oct_contrast_Slider_circ.valueChanged.connect(self.circ_low)
        
        self.ui.circ_resample_btn.setEnabled(True)
        self.ui.save_marked_plot_circ.clicked.connect(self.save_marked_circ_measurement_plot)
        #self.ui.circ_horizontalSlider_2.setEnabled(True)
        #self.ui.circ_horizontalSlider_2.valueChanged.connect(self.circ_high)
        self.ui.circ_save_raw_btn.setEnabled(True)
        self.ui.circ_save_raw_btn.clicked.connect(self.save_raw_circ_resample)
    
    def circ_low(self,value):
        self.circ_lowVal = (value-75)*self.circ_step+self.lowVal                           #lower lowbound and higher upperbound for bscan
        self.circ_highVal = (self.ui.oct_contrast_Slider_circ.value()-75)*self.circ_step+self.highVal
        self.update_Circ_Img()

    def angle_grid(self,start_coord,center,r,eye):
        if eye ==0 or eye ==3:
            self.circ_enface_add_angle_grid_OD(start_coord,center,r)
        elif eye==1:
            self.circ_enface_add_angle_grid_OS(start_coord,center,r)

    def circ_high(self,value):
        self.circ_lowVal = (self.ui.circ_horizontalSlider.value()-75)*self.circ_step+self.lowVal                           #lower lowbound and higher upperbound for bscan
        self.circ_highVal = (value-75)*self.circ_step+self.highVal
        self.update_Circ_Img()
    
    def update_Circ_Img(self):
        self.ui.circ_Slow_Axis.clear()
        circ_img = ((np.clip((self.frame_3D - self.circ_lowVal) / (self.circ_highVal - self.circ_lowVal), 0, 1)) * 255).astype(np.uint8)
        circ_img = pg.ImageItem(np.flipud(np.rot90(circ_img)))
        circ_img.setRect(0,0,int(np.shape(self.frame_3D)[1]),330)
        self.ui.circ_Slow_Axis.addItem(circ_img, clear=True)
        #self.ui.circ_Slow_Axis.setImageSize(self.processParameters,self.ui.distance,self.ui.circ_Slow_Axis.geometry())

    def go_home(self):
        if self.ui.stackedWidget.currentIndex() != 0:
            self.ui.stackedWidget.setCurrentIndex(0)
            self.ui.label.setText(self._translate("MainWindow", "Status: VisOCT Explorer "+self.status))
            self.ui.horizontalSlider_01.valueChanged.disconnect()
            self.ui.Edit_circ_x.textChanged.disconnect()
            self.ui.Edit_circ_y.textChanged.disconnect()
            self.ui.Edit_radius.textChanged.disconnect()

    def reset_circ_measurement(self):
        self.ui.circ_Slow_Axis.clear()
        print("reset")
        circ_img = ((np.clip((self.frame_3D - self.circ_lowVal) / (self.circ_highVal - self.circ_lowVal), 0, 1)) * 255).astype(np.uint8)
        
        if self.processParameters.eye == 0 or self.processParameters.eye == 3:
            circ_img = pg.ImageItem(np.rot90(circ_img))
        else:
            circ_img = pg.ImageItem(np.flipud(np.rot90(np.roll(circ_img,int(np.shape(circ_img)[1]/2),axis=1))))

        cmap = plt.get_cmap('gray')

        # Convert the colormap to a lookup table (LUT)
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        
        circ_img.setLookupTable(lut)

        circ_img.setRect(0,0,int(np.shape(self.frame_3D)[1]*0.8),330)
        self.ui.circ_Slow_Axis.addItem(circ_img, clear=True)

        self.circ_resample_add_angle_grid()
        #grid = GridItem(int(np.shape(self.frame_3D)[1]*0.8))
        #self.ui.Slow_Axis.showGrid(x=True, y=True)
        #self.ui.circ_Slow_Axis.addItem(grid)
        self.ui.circ_Slow_Axis.reset_measure()
        self.ui.distance_circ.setText(" ")

    def circ_enface_add_angle_grid_OD(self,start_coord,center,r):
        self.circ_grid_plotted = True
        self.ui.Enface_circ.plot([start_coord[0],center[0]],[start_coord[1],center[1]],pen=pg.mkPen('r',width=3))
        measurement_number = pg.TextItem(text=f"{0}",color=(255,0,0))
        measurement_number.setFont(QFont("Arial",10))
        measurement_number.setPos(start_coord[0]-35,start_coord[1]+15)
        self.ui.Enface_circ.addItem(measurement_number)

        self.ui.Enface_circ.plot([int(center[0]-r * np.cos(np.radians(30))),center[0]],[int(start_coord[1]+r * np.sin(np.radians(30))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number2 = pg.TextItem(text=f"{210}",color=(255,0,0))
        measurement_number2.setFont(QFont("Arial",10))
        measurement_number2.setPos(int(center[0]-r * np.cos(np.radians(30)))-20,int(start_coord[1]+r * np.sin(np.radians(30)))-15)
        self.ui.Enface_circ.addItem(measurement_number2)
        
        self.ui.Enface_circ.plot([int(center[0]-r * np.cos(np.radians(60))),center[0]],[int(start_coord[1]+r * np.sin(np.radians(60))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number11 = pg.TextItem(text=f"{240}",color=(255,0,0))
        measurement_number11.setFont(QFont("Arial",10))
        measurement_number11.setPos(int(center[0]-r * np.cos(np.radians(60)))-20,int(start_coord[1]+r * np.sin(np.radians(60)))-15)
        self.ui.Enface_circ.addItem(measurement_number11)

        self.ui.Enface_circ.plot([center[0],center[0]],[start_coord[1]+r,center[1]],pen=pg.mkPen('r',width=3))
        measurement_number3 = pg.TextItem(text=f"{270}",color=(255,0,0))
        measurement_number3.setFont(QFont("Arial",10))
        measurement_number3.setPos(center[0]-10,start_coord[1]+r-10)
        self.ui.Enface_circ.addItem(measurement_number3)

        self.ui.Enface_circ.plot([int(center[0]+r * np.cos(np.radians(30))),center[0]],[int(start_coord[1]+r * np.sin(np.radians(30))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number4 = pg.TextItem(text=f"{330}",color=(255,0,0))
        measurement_number4.setFont(QFont("Arial",10))
        measurement_number4.setPos(int(center[0]+r * np.cos(np.radians(30)))-20,int(start_coord[1]+r * np.sin(np.radians(30)))-10)
        self.ui.Enface_circ.addItem(measurement_number4)
        
        self.ui.Enface_circ.plot([int(center[0]+r * np.cos(np.radians(60))),center[0]],[int(start_coord[1]+r * np.sin(np.radians(60))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number12 = pg.TextItem(text=f"{300}",color=(255,0,0))
        measurement_number12.setFont(QFont("Arial",10))
        measurement_number12.setPos(int(center[0]+r * np.cos(np.radians(60)))-20,int(start_coord[1]+r * np.sin(np.radians(60)))-10)
        self.ui.Enface_circ.addItem(measurement_number12)

        self.ui.Enface_circ.plot([center[0]-r,center[0]],[start_coord[1],center[1]],pen=pg.mkPen('r',width=3))
        measurement_number5 = pg.TextItem(text=f"{180}",color=(255,0,0))
        measurement_number5.setFont(QFont("Arial",10))
        measurement_number5.setPos(center[0]-r+20,start_coord[1]-10)
        self.ui.Enface_circ.addItem(measurement_number5)

        self.ui.Enface_circ.plot([center[0],center[0]],[start_coord[1]-r,center[1]],pen=pg.mkPen('r',width=3))
        measurement_number6 = pg.TextItem(text=f"{90}",color=(255,0,0))
        measurement_number6.setFont(QFont("Arial",10))
        measurement_number6.setPos(center[0],start_coord[1]-r+45)
        self.ui.Enface_circ.addItem(measurement_number6)

        self.ui.Enface_circ.plot([int(center[0]+r * np.cos(np.radians(30))),center[0]],[int(start_coord[1]-r * np.sin(np.radians(30))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number7 = pg.TextItem(text=f"{30}",color=(255,0,0))
        measurement_number7.setFont(QFont("Arial",10))
        measurement_number7.setPos(int(center[0]+r * np.cos(np.radians(30)))-50,int(start_coord[1]-r * np.sin(np.radians(30)))+50)
        self.ui.Enface_circ.addItem(measurement_number7)

        self.ui.Enface_circ.plot([int(center[0]+r * np.cos(np.radians(60))),center[0]],[int(start_coord[1]-r * np.sin(np.radians(60))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number9 = pg.TextItem(text=f"{60}",color=(255,0,0))
        measurement_number9.setFont(QFont("Arial",10))
        measurement_number9.setPos(int(center[0]+r * np.cos(np.radians(60)))-50,int(start_coord[1]-r * np.sin(np.radians(60)))+50)
        self.ui.Enface_circ.addItem(measurement_number9)

        self.ui.Enface_circ.plot([int(center[0] - r * np.cos(np.radians(60))),center[0]],[int(start_coord[1]-r * np.sin(np.radians(60))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number10 = pg.TextItem(text=f"{120}",color=(255,0,0))
        measurement_number10.setFont(QFont("Arial",10))
        measurement_number10.setPos(int(center[0] - r * np.cos(np.radians(60)))+5,int(start_coord[1]-r * np.sin(np.radians(60)))+20)
        self.ui.Enface_circ.addItem(measurement_number10)

        self.ui.Enface_circ.plot([int(center[0] - r * np.cos(np.radians(30))),center[0]],[int(start_coord[1]-r * np.sin(np.radians(30))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number8 = pg.TextItem(text=f"{150}",color=(255,0,0))
        measurement_number8.setFont(QFont("Arial",10))
        measurement_number8.setPos(int(center[0] - r * np.cos(np.radians(30)))+5,int(start_coord[1]-r * np.sin(np.radians(30)))+20)
        self.ui.Enface_circ.addItem(measurement_number8)

    def circ_enface_add_angle_grid_OS(self,start_coord,center,r):
        self.circ_grid_plotted = True
        self.ui.Enface_circ.plot([start_coord[0],center[0]],[start_coord[1],center[1]],pen=pg.mkPen('r',width=3))
        measurement_number = pg.TextItem(text=f"{180}",color=(255,0,0))
        measurement_number.setFont(QFont("Arial",10))
        measurement_number.setPos(start_coord[0]-35,start_coord[1]+15)
        self.ui.Enface_circ.addItem(measurement_number)

        self.ui.Enface_circ.plot([int(center[0]-r * np.cos(np.radians(30))),center[0]],[int(start_coord[1]+r * np.sin(np.radians(30))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number2 = pg.TextItem(text=f"{330}",color=(255,0,0))
        measurement_number2.setFont(QFont("Arial",10))
        measurement_number2.setPos(int(center[0]-r * np.cos(np.radians(30)))-20,int(start_coord[1]+r * np.sin(np.radians(30)))-15)
        self.ui.Enface_circ.addItem(measurement_number2)
        
        self.ui.Enface_circ.plot([int(center[0]-r * np.cos(np.radians(60))),center[0]],[int(start_coord[1]+r * np.sin(np.radians(60))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number11 = pg.TextItem(text=f"{300}",color=(255,0,0))
        measurement_number11.setFont(QFont("Arial",10))
        measurement_number11.setPos(int(center[0]-r * np.cos(np.radians(60)))-20,int(start_coord[1]+r * np.sin(np.radians(60)))-15)
        self.ui.Enface_circ.addItem(measurement_number11)

        self.ui.Enface_circ.plot([center[0],center[0]],[start_coord[1]+r,center[1]],pen=pg.mkPen('r',width=3))
        measurement_number3 = pg.TextItem(text=f"{270}",color=(255,0,0))
        measurement_number3.setFont(QFont("Arial",10))
        measurement_number3.setPos(center[0]-10,start_coord[1]+r-10)
        self.ui.Enface_circ.addItem(measurement_number3)

        self.ui.Enface_circ.plot([int(center[0]+r * np.cos(np.radians(30))),center[0]],[int(start_coord[1]+r * np.sin(np.radians(30))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number4 = pg.TextItem(text=f"{210}",color=(255,0,0))
        measurement_number4.setFont(QFont("Arial",10))
        measurement_number4.setPos(int(center[0]+r * np.cos(np.radians(30)))-20,int(start_coord[1]+r * np.sin(np.radians(30)))-10)
        self.ui.Enface_circ.addItem(measurement_number4)
        
        self.ui.Enface_circ.plot([int(center[0]+r * np.cos(np.radians(60))),center[0]],[int(start_coord[1]+r * np.sin(np.radians(60))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number12 = pg.TextItem(text=f"{240}",color=(255,0,0))
        measurement_number12.setFont(QFont("Arial",10))
        measurement_number12.setPos(int(center[0]+r * np.cos(np.radians(60)))-20,int(start_coord[1]+r * np.sin(np.radians(60)))-10)
        self.ui.Enface_circ.addItem(measurement_number12)

        self.ui.Enface_circ.plot([center[0]-r,center[0]],[start_coord[1],center[1]],pen=pg.mkPen('r',width=3))
        measurement_number5 = pg.TextItem(text=f"{0}",color=(255,0,0))
        measurement_number5.setFont(QFont("Arial",10))
        measurement_number5.setPos(center[0]-r+20,start_coord[1]-10)
        self.ui.Enface_circ.addItem(measurement_number5)

        self.ui.Enface_circ.plot([center[0],center[0]],[start_coord[1]-r,center[1]],pen=pg.mkPen('r',width=3))
        measurement_number6 = pg.TextItem(text=f"{90}",color=(255,0,0))
        measurement_number6.setFont(QFont("Arial",10))
        measurement_number6.setPos(center[0],start_coord[1]-r+45)
        self.ui.Enface_circ.addItem(measurement_number6)

        self.ui.Enface_circ.plot([int(center[0]+r * np.cos(np.radians(30))),center[0]],[int(start_coord[1]-r * np.sin(np.radians(30))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number7 = pg.TextItem(text=f"{150}",color=(255,0,0))
        measurement_number7.setFont(QFont("Arial",10))
        measurement_number7.setPos(int(center[0]+r * np.cos(np.radians(30)))-50,int(start_coord[1]-r * np.sin(np.radians(30)))+50)
        self.ui.Enface_circ.addItem(measurement_number7)

        self.ui.Enface_circ.plot([int(center[0]+r * np.cos(np.radians(60))),center[0]],[int(start_coord[1]-r * np.sin(np.radians(60))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number9 = pg.TextItem(text=f"{120}",color=(255,0,0))
        measurement_number9.setFont(QFont("Arial",10))
        measurement_number9.setPos(int(center[0]+r * np.cos(np.radians(60)))-50,int(start_coord[1]-r * np.sin(np.radians(60)))+50)
        self.ui.Enface_circ.addItem(measurement_number9)

        self.ui.Enface_circ.plot([int(center[0] - r * np.cos(np.radians(60))),center[0]],[int(start_coord[1]-r * np.sin(np.radians(60))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number10 = pg.TextItem(text=f"{60}",color=(255,0,0))
        measurement_number10.setFont(QFont("Arial",10))
        measurement_number10.setPos(int(center[0] - r * np.cos(np.radians(60)))+5,int(start_coord[1]-r * np.sin(np.radians(60)))+20)
        self.ui.Enface_circ.addItem(measurement_number10)

        self.ui.Enface_circ.plot([int(center[0] - r * np.cos(np.radians(30))),center[0]],[int(start_coord[1]-r * np.sin(np.radians(30))),center[1]],pen=pg.mkPen('r',width=3))
        measurement_number8 = pg.TextItem(text=f"{30}",color=(255,0,0))
        measurement_number8.setFont(QFont("Arial",10))
        measurement_number8.setPos(int(center[0] - r * np.cos(np.radians(30)))+5,int(start_coord[1]-r * np.sin(np.radians(30)))+20)
        self.ui.Enface_circ.addItem(measurement_number8)
        

    def circ_resample_add_angle_grid(self):
        unit_width = int(np.shape(self.frame_3D)[1]*0.8/12)
        self.line_11 = self.ui.circ_Slow_Axis.plot([unit_width,unit_width],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_1 = pg.TextItem(text=f"{30}",color=(255,0,0))
        self.angle_1.setFont(QFont("Arial",10))
        self.angle_1.setPos(unit_width-15,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_1)
        self.line_1 = self.ui.circ_Slow_Axis.plot([unit_width*2,unit_width*2],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_2 = pg.TextItem(text=f"{60}",color=(255,0,0))
        self.angle_2.setFont(QFont("Arial",10))
        self.angle_2.setPos(unit_width*2-15,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_2)
        self.line_2 = self.ui.circ_Slow_Axis.plot([unit_width*3,unit_width*3],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_3 = pg.TextItem(text=f"{90}",color=(255,0,0))
        self.angle_3.setFont(QFont("Arial",10))
        self.angle_3.setPos(unit_width*3-25,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_3)
        self.line_3 = self.ui.circ_Slow_Axis.plot([unit_width*4,unit_width*4],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_4 = pg.TextItem(text=f"{120}",color=(255,0,0))
        self.angle_4.setFont(QFont("Arial",10))
        self.angle_4.setPos(unit_width*4-25,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_4)
        self.line_4 = self.ui.circ_Slow_Axis.plot([unit_width*5,unit_width*5],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_5 = pg.TextItem(text=f"{150}",color=(255,0,0))
        self.angle_5.setFont(QFont("Arial",10))
        self.angle_5.setPos(unit_width*5-25,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_5)
        self.line_5 = self.ui.circ_Slow_Axis.plot([unit_width*6,unit_width*6],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_6 = pg.TextItem(text=f"{180}",color=(255,0,0))
        self.angle_6.setFont(QFont("Arial",10))
        self.angle_6.setPos(unit_width*6-25,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_6)
        self.line_6 = self.ui.circ_Slow_Axis.plot([unit_width*7,unit_width*7],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_7 = pg.TextItem(text=f"{210}",color=(255,0,0))
        self.angle_7.setFont(QFont("Arial",10))
        self.angle_7.setPos(unit_width*7-25,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_7)
        self.line_7 = self.ui.circ_Slow_Axis.plot([unit_width*8,unit_width*8],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_8 = pg.TextItem(text=f"{240}",color=(255,0,0))
        self.angle_8.setFont(QFont("Arial",10))
        self.angle_8.setPos(unit_width*8-25,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_8)
        self.line_8 = self.ui.circ_Slow_Axis.plot([unit_width*9,unit_width*9],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_9 = pg.TextItem(text=f"{270}",color=(255,0,0))
        self.angle_9.setFont(QFont("Arial",10))
        self.angle_9.setPos(unit_width*9-25,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_9)
        self.line_9 = self.ui.circ_Slow_Axis.plot([unit_width*10,unit_width*10],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_10 = pg.TextItem(text=f"{300}",color=(255,0,0))
        self.angle_10.setFont(QFont("Arial",10))
        self.angle_10.setPos(unit_width*10-25,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_10)
        self.line_10 = self.ui.circ_Slow_Axis.plot([unit_width*11,unit_width*11],[1024,0],pen=pg.mkPen('r',width=1))
        self.angle_11 = pg.TextItem(text=f"{330}",color=(255,0,0))
        self.angle_11.setFont(QFont("Arial",10))
        self.angle_11.setPos(unit_width*11-25,15)
        self.ui.circ_Slow_Axis.addItem(self.angle_11)

    def circ_resample_remove_angle_grid(self):
        self.ui.circ_Slow_Axis.removeItem(self.angle_1)
        self.ui.circ_Slow_Axis.removeItem(self.angle_2)
        self.ui.circ_Slow_Axis.removeItem(self.angle_3)
        self.ui.circ_Slow_Axis.removeItem(self.angle_4)
        self.ui.circ_Slow_Axis.removeItem(self.angle_5)
        self.ui.circ_Slow_Axis.removeItem(self.angle_6)
        self.ui.circ_Slow_Axis.removeItem(self.angle_7)
        self.ui.circ_Slow_Axis.removeItem(self.angle_8)
        self.ui.circ_Slow_Axis.removeItem(self.angle_9)
        self.ui.circ_Slow_Axis.removeItem(self.angle_10)
        self.ui.circ_Slow_Axis.removeItem(self.angle_11)
        self.ui.circ_Slow_Axis.removeItem(self.line_1)
        self.ui.circ_Slow_Axis.removeItem(self.line_2)
        self.ui.circ_Slow_Axis.removeItem(self.line_3)
        self.ui.circ_Slow_Axis.removeItem(self.line_4)
        self.ui.circ_Slow_Axis.removeItem(self.line_5)
        self.ui.circ_Slow_Axis.removeItem(self.line_6)
        self.ui.circ_Slow_Axis.removeItem(self.line_7)
        self.ui.circ_Slow_Axis.removeItem(self.line_8)
        self.ui.circ_Slow_Axis.removeItem(self.line_9)
        self.ui.circ_Slow_Axis.removeItem(self.line_10)
        self.ui.circ_Slow_Axis.removeItem(self.line_11)

    def reset_measurement(self):
        slow_axis_img_np = cv2.resize(((np.clip((self.frame_3D_resh[:,:,self.cur_img_slow_no] - self.oct_lowVal) / (self.oct_highVal - self.oct_lowVal), 0, 1)) * 255).astype(np.uint8),(0,0),fx=4,fy=4)
        self.ui.Slow_Axis.clear()
        slow_img = pg.ImageItem(np.flipud(np.rot90(slow_axis_img_np)))
        slow_img.setRect(0,0,1024,1024)
        self.ui.Slow_Axis.addItem(slow_img, clear=True)
        self.ui.Slow_Axis.reset_measure()
        self.ui.distance.setText(" ")

    def TSA_ready_callback(self,frame_3D,frame_OCTA):
        lowVal = self.lowVal
        highVal = self.highVal
        self.slow_axis_img_np = ((np.clip((frame_3D[:,:,int(frame_3D.shape[2]/2)] - lowVal) / (highVal - lowVal), 0, 1)) * 255).astype(np.uint8)
        self.slow_axis_img = pg.ImageItem(np.rot90(self.slow_axis_img_np))
        self.slow_axis_img.setRect(0,0,1024,1024)
        self.ui.Slow_Axis.clear()
        self.ui.Slow_Axis.addItem(self.slow_axis_img, clear=True)
        self.ui.Slow_Axis.setImageSize(self.processParameters,self.ui.distance,self.ui.Slow_Axis.geometry(),1024)
        self.cur_img_slow_no = int(self.frame_3D_resh.shape[2]/2)

        fast_axis_img = ((np.clip((frame_3D[:,int(frame_3D.shape[1]/2),:] - lowVal) / (highVal - lowVal), 0, 1)) * 255).astype(np.uint8)
        self.ui.Fast_Axis.clear()
        self.ui.Fast_Axis.addItem(pg.ImageItem(np.rot90(fast_axis_img)), clear=True)
        self.cur_img_fast_no = int(self.frame_3D_resh_origin.shape[1]/2)
        
        
    def circ_on_mouse_drag(self,event):
        if event.button() == Qt.LeftButton:
            # Update the position of the draggable point
            pos = self.ui.Enface_circ.getViewBox().mapToView(event.scenePos())
            print("x:",pos.x())
            print("y:",pos.x())
            self.resample_center.setData(x=[pos.x()], y=[pos.y()])
            circ_x = str(int(pos.x()*self.processParameters.xrng/1024))
            circ_y = str(int(pos.y()*self.processParameters.yrng/1024))
            self.ui.Edit_circ_x.setText(circ_x)
            self.ui.Edit_circ_y.setText(circ_y)

    def update_volume(self):
        self.frame_3D_resh_origin_linear = np.load(self.processParameters.fname.split('.RAW')[0]+'/frame_3D.npy')
        self.enface_divider = int(self.ui.averaging_dropdown.toPlainText())
        
        self.ui.bscan_width.setText(self._translate("MainWindow", "B-scan width: "+str(self.processParameters.xrng*int(self.ui.averaging_dropdown.toPlainText())/int(self.processParameters.res_slow))+" um"))
        self.update_volume_process.set_attrib(self.frame_3D_resh_origin_linear,self.processParameters,self.enface_divider)
        self.update_volume_process.start()
        del self.frame_3D_resh_origin_linear

    def update_volume_progress_callback(self,progress_percent):
        self.ui.progressBar.setValue(progress_percent)

    def update_volume_ready_callback(self,frame_3D_linear):
        self.frame_3D_linear = frame_3D_linear
        self.frame_3D_20log = 20*np.log10(frame_3D_linear)
        self.frame_3D_resh = self.frame_3D_20log
        self.cur_img_slow_no = int(self.frame_3D_resh.shape[2]/2)
        self.slow_axis_img_np = cv2.resize(((np.clip((self.frame_3D_resh[:,:,self.cur_img_slow_no] - self.oct_lowVal) / (self.oct_highVal - self.oct_lowVal), 0, 1)) * 255).astype(np.uint8),(0,0),fx=4,fy=4)
        self.ui.Slow_Axis.clear()
        slow_img = pg.ImageItem(np.flipud(np.rot90(self.slow_axis_img_np)))
        slow_img.setRect(0,0,1024,1024)
        self.ui.Slow_Axis.addItem(slow_img, clear=True)
        self.ui.Slow_Axis.reset_measure()
        self.ui.distance.setText(" ")
        self.red_line.setValue(512)

    def oct_enface_contrast(self,value):
        img_pil = Image.open(self.directory+"/enface.tiff").convert("RGB")
        #enhancer = ImageEnhance.Contrast(img_pil)
        gamma = (value+1)/33
        arr = np.array(img_pil, dtype=np.float32) / 255.0
        arr_gamma = np.power(arr, gamma)

        # Convert back to uint8 and rotate
        arr_gamma_uint8 = (arr_gamma * 255).astype(np.uint8)
        enface_pixmap = np.rot90(arr_gamma_uint8)
        self.enface.setImage(enface_pixmap)

    def image_ready_callback(self,frame_3D_resh,directory,lowVal,highVal,QI,frame_OCTA=None,octa_lowVal=None,octa_highVal=None):
        self.frame_3D_20log = np.float32(20*np.log10(frame_3D_resh))
        self.frame_3D_resh = self.frame_3D_20log
        if self.dialog is None:
            self.dialog = MyDialog(self)
        
        self.linear_scale = False
        #for q in QI:
        #    print("QI is:",q)
        #hist,bins = np.histogram(self.frame_3D_20log,bins=100)
        #hist = hist/(np.max(hist) * 0.1)

        #self.ui.tlogt.setChecked(True)
        #self.ui.linear.setChecked(False)
        #self.ui.tlogt.clicked.connect(self.set_frame_3D_log)
        #self.frame_3D_resh
        self.enface_divider = 4

        self.frame_OCTA = frame_OCTA
        self.oct_lowVal = lowVal
        self.oct_highVal = highVal
        self.lowVal = lowVal
        self.highVal = highVal
        self.image_ready = True
        self.oct_step = float(70/99)
        self.ui.reset_measure.clicked.connect(self.reset_measurement)

        self.directory = directory
        
        self.ui.Enface.clear()

        if self.preview_mode!=True:
            enface_pixmap = np.array(Image.open(directory+"/enface.tiff"))
            self.enface_pixmap = enface_pixmap
            self.enface_height = enface_pixmap.shape[1]
            self.enface_width = enface_pixmap.shape[0]

            self.enface = pg.ImageItem(np.fliplr(enface_pixmap))
            self.circ_enface = pg.ImageItem(np.fliplr(enface_pixmap))
            self.enface.setRect(0,0,1024,1024)
            self.circ_enface.setRect(0,0,1024,1024)
            self.enface_lowVal = np.min(enface_pixmap)
            self.enface_highVal = np.max(enface_pixmap)
            self.oct_enface_step = float((self.enface_highVal-self.enface_lowVal)/220)

        #self.ui.contrast_histogram_widget.clear()
        #contrast_histo = CustomBarGraphItem(x = bins[:-1], height = hist, width = 1, brush ='g') 
        
        log_histo = QPixmap(directory+"\log_scale_contrast.tiff")
        log_histo = log_histo.scaledToWidth(405)
        self.ui.contrast_histogram_widget.setPixmap(log_histo)
        #self.ui.contrast_histogram.setXRange(-100, 10, padding=0)
        #self.ui.contrast_histogram.setYRange(0, 10, padding=0)
        if self.preview_mode != True:
            self.ui.Enface.addItem(self.enface,clear = True)
        red_line = pg.InfiniteLine(pos=512,angle=0,movable=True,pen=pg.mkPen('blue',width=5))
        red_line.setBounds([0,1024])
        self.ui.Enface.addItem(red_line)
        red_line.sigDragged.connect(self.Enface_red_dragged)
        self.red_line = red_line
        
        self.ui.bscan_dim_label.setText(self._translate("MainWindow", "B-scan dimension: (H)1024 x (W)"+str(self.processParameters.res_fast)))
        self.ui.actual_dim_label.setText(self._translate("MainWindow", "Actual dimension: (H)1150 um x (W)"+str(self.processParameters.xrng)+" um"))
        self.ui.bscan_width.setText(self._translate("MainWindow", "B-scan width: "+str(self.processParameters.xrng*int(self.ui.averaging_dropdown.toPlainText())/int(self.processParameters.res_slow))+" um"))
        self.slow_axis_img_np = cv2.resize(((np.clip((self.frame_3D_resh[:,:,self.cur_img_slow_no] - self.oct_lowVal) / (self.oct_highVal - self.oct_lowVal), 0, 1)) * 255).astype(np.uint8),(0,0),fx=4,fy=4)
        self.cur_img_slow_no = int(self.frame_3D_resh.shape[2]/2)
        self.ui.bscan_number_label.setText(self._translate("MainWindow", "B-scan #: "+str(self.cur_img_slow_no*4)))
        self.ui.Slow_Axis.setImageSize(self.processParameters,self.ui.distance,1024)
        slow_axis_img = pg.ImageItem(np.fliplr(np.rot90(self.slow_axis_img_np)))
        slow_axis_img.setRect(0,0,1024,1024)
        self.ui.Slow_Axis.clear()
        self.ui.Slow_Axis.addItem(slow_axis_img, clear=True)
        
        self.ui.Slow_Axis.setBscan_number(int(self.frame_3D_resh.shape[2]*2))
        self.ui.save_marked_plot.clicked.connect(self.save_marked_measurement_plot)

        #fast_axis_img = ((np.clip((self.frame_3D_resh_origin[:,int(self.frame_3D_resh_origin.shape[1]/2),:] - lowVal) / (highVal - lowVal), 0, 1)) * 255).astype(np.uint8)
        #fast_axis_img = pg.ImageItem(np.flipud(np.rot90(fast_axis_img)))
        #fast_axis_img.setRect(0,0,1024,1024)
        #self.ui.Fast_Axis.clear()
        #self.ui.Fast_Axis.addItem(fast_axis_img, clear=True)
        #self.ui.Fast_Axis.setImageSize(self.processParameters,self.ui.distance,self.ui.Fast_Axis.geometry(),1024)
        self.reset_measurement()

        if self.flip_bscan:
            self.ui.Slow_Axis.invertY(True)
            #self.ui.Fast_Axis.invertY(True)

        
        self.ui.Enface_circ.clear()
        self.ui.circ_Slow_Axis.clear()
        self.ui.Enface_circ.addItem(self.circ_enface,clear = True)

        self.resample_circ = pg.ScatterPlotItem(size=0, pos=[(512,512)], brush=None, symbolBrush='r', pen=pg.mkPen('r',width=3), symbol='o')
        self.resample_circ.setPxMode(False)
        self.resample_center = pg.ScatterPlotItem(size=8, pos=[(512,512)], brush=None, symbolBrush='r', pen=pg.mkPen('r',width=3), symbol='o')
        self.resample_center.setPxMode(False)
        self.ui.Enface_circ.getViewBox().scene().sigMouseClicked.connect(self.circ_on_mouse_drag)
        #self.resample_center.sigDragged.connect(self.Enface_red_dragged)
        #self.new_x = 512
        #self.new_y = 512
        self.x_center = 512
        self.y_center = 512
        self.ui.Enface_circ.addItem(self.resample_circ)
        self.ui.Enface_circ.addItem(self.resample_center)

        self.ui.save_btn.clicked.connect(self.save_img)
        self.ui.next_frame_btn.clicked.connect(self.go_prev)
        self.ui.prev_frame_btn.clicked.connect(self.go_next)
        self.ui.Enface_contrast.setEnabled(True)
        self.ui.Enface_contrast.valueChanged.connect(self.oct_enface_contrast)

        self.radius_given = self.ui.horizontalSlider_01.value()
        self.ui.Edit_radius.setText(f"{str(int(self.radius_given/1024*self.processParameters.xrng))}")
        self.ui.circ_resample_btn.clicked.connect(self.resample)
        self.status = " (View Image)"
        self.ui.label.setText(self._translate("MainWindow", "Status: VisOCT Explorer"+self.status))
        #self.ui.oct_label.setText(self._translate("MainWindow", "Min value: "+str(round(self.lowVal,3))))
        #self.ui.oct_label2.setText(self._translate("MainWindow", "Max value: "+str(round(self.highVal,3))))
        self.ui.oct_contrast_Slider.setEnabled(True)
        #slider_low = 
        self.ui.oct_contrast_Slider.setValue((int(self.lowVal),int(self.highVal)))
        self.ui.Edit_min.setText(str(round(int(self.lowVal),3)))
        self.ui.Edit_max.setText(str(round(int(self.highVal),3)))
        self.ui.oct_contrast_Slider.valueChanged.connect(self.contrast)
        self.ui.pushButton_4.setEnabled(True)
        self.ui.fibergram_btn.setEnabled(True)
        self.ui.circ_btn.setEnabled(True)
        self.ui.octa_btn.setEnabled(True)
        self.ui.averaging_btn.setEnabled(True)
        self.ui.enface_otherdim.setEnabled(True)
        self.ui.flatten_btn.setEnabled(True)
        self.ui.tsa_btn.setEnabled(True)
        self.ui.averaging_btn.clicked.connect(self.update_volume)

        if self.processParameters.octaFlag:
            self.octa_enface_pixmap = np.array(Image.open(directory+"/octa_enface.tiff"))
            self.ui.OCTA_Enface.clear()
            octa_enface = pg.ImageItem(np.fliplr(np.flipud(self.octa_enface_pixmap)))
            #if self.octa_enface_pixmap.shape[1] == 512:
            octa_enface.setRect(0,0,1024,1024)
            #octa_enface.setRect(0,0,512,512)
            self.octa_enface_lowVal = 0
            self.octa_enface_highVal = 255
            self.octa_step = float(octa_highVal-octa_lowVal)/100
            self.octa_lowVal_base = octa_lowVal#-10*self.octa_step
            self.octa_lowVal = octa_lowVal#-10*self.octa_step
            self.octa_highVal_base = octa_highVal
            self.octa_highVal = octa_highVal
            octa_log_histo = QPixmap(directory+"\log_scale_octa_contrast.tiff")
            octa_log_histo = octa_log_histo.scaledToWidth(402)
            self.ui.contrast_histogram_widget_octa.setPixmap(octa_log_histo)
            
            self.ui.OCTA_Enface.addItem(octa_enface,clear = True)
            self.ui.octa_btn.setEnabled(True)
            #self.ui.tsa_btn_1.setEnabled(True)
            #self.ui.tsa_btn_2.setEnabled(True)
            
            octa_red_line = pg.InfiniteLine(pos=512,angle=0,movable=True,pen=pg.mkPen('b',width=5))
            octa_red_line.setBounds([0,1024])
            self.ui.OCTA_Enface.addItem(octa_red_line)
            octa_red_line.sigDragged.connect(self.OCTA_Enface_red_dragged)
            self.octa_red_line = octa_red_line
            self.octa_slow_axis_img_np = ((np.clip((frame_OCTA[:,:,int(self.frame_OCTA.shape[2]/2)] - self.octa_lowVal_base) / (self.octa_highVal_base - self.octa_lowVal_base), 0, 1)) * 255).astype(np.uint8)
            print(self.octa_slow_axis_img_np)
            self.octa_slow_axis_img = pg.ImageItem(np.flipud(np.rot90(self.octa_slow_axis_img_np)))
            self.octa_slow_axis_img.setRect(0,0,1024,1024)
            print("slow axis geometry is:",self.ui.OCTA_Slow_Axis.geometry().width())
            self.ui.OCTA_Slow_Axis.clear()
            self.ui.OCTA_Slow_Axis.addItem(self.octa_slow_axis_img, clear=True)
            self.ui.OCTA_Slow_Axis.setImageSize(self.processParameters,self.ui.distance,self.ui.OCTA_Slow_Axis.geometry(),1024)
            self.cur_octa_img_slow_no = int(self.frame_OCTA.shape[2]/2)
            self.ui.OCTA_Slow_Axis.setBscan_number(int(self.frame_OCTA.shape[2]/2))
            self.ui.bscan_number_label_octa.setText(self._translate("MainWindow", "B-scan #: "+str(self.cur_octa_img_slow_no)))
            self.ui.bscan_dim_label_octa.setText(self._translate("MainWindow","B-scan dimension: (H)1024 x (W)"+str(self.processParameters.res_fast)))
            #self.ui.octa_save_btn.clicked.connect(self.octa_save_img)
            #self.ui.octa_zoom_in_btn.clicked.connect(self.octa_go_prev)
            #self.ui.octa_zoom_out_btn.clicked.connect(self.octa_go_next)


    def save_img(self):
        if self.dialog.exec() == QDialog.Accepted:
            selected_option = self.dialog.get_selected_option()
            #QMessageBox.information(self,"Selection", f"Selected option: {selected_option}")
            if selected_option == "Average 32 frames and save":
                average_number = 32
            elif selected_option == "Average 16 frames and save":
                average_number = 16
            elif selected_option == "Average 8 frames and save":
                average_number = 8
            elif selected_option == "Average 4 frames and save":
                average_number = 4
            elif selected_option == "Save as dicom":
                average_number = 1
            elif selected_option == "Save speckle reduced images":
                average_number = 1
            elif selected_option == "Save as tiff stack":
                average_number = 1
            
            self.frame_3D_resh_origin = 20*np.log10(np.load(self.processParameters.fname.split('.RAW')[0]+'/frame_3D.npy'))
            self.ui.textEdit_2.append("Start to save processed image to file...")
            self.save_thread = SavingThread(self.processParameters,self.frame_3D_resh_origin,selected_option,self.directory,average_number,self.oct_lowVal,self.oct_highVal)
            self.save_thread.finished.connect(self.save_finished)
            self.save_thread.start()
        #return

    def save_finished(self):
        self.ui.textEdit_2.append("File saving completed")

    def multiple_measurement_circ_btn_toggle(self):
        self.ui.circ_Slow_Axis.toggle_multiple_measurement()

    def save_marked_measurement_plot(self):
        self.ui.Slow_Axis.autoRange()
        exporter = ImageExporter(self.ui.Slow_Axis)
        if not os.path.exists(self.directory+"/measurement_result"):
                os.makedirs(self.directory+"/measurement_result")
        exporter.export(self.directory+'/measurement_result/marked_bscan_'+str(self.cur_img_slow_no*self.enface_divider)+'.tif')

    def save_raw_circ_resample(self):
        print("save_raw circ")
        self.ui.Enface_circ.autoRange()
        exporter = ImageExporter(self.ui.Enface_circ)
        exporter.parameters()['width'] = 350
        if not os.path.exists(self.directory+"/measurement_result"):
            os.makedirs(self.directory+"/measurement_result")
        if not os.path.exists(self.directory+"/measurement_result/r="+self.ui.Edit_radius.toPlainText()):
            os.makedirs(self.directory+"/measurement_result/r="+self.ui.Edit_radius.toPlainText())
        enface_filename = self.directory+"/measurement_result/r="+self.ui.Edit_radius.toPlainText()+'/enface.tif'
        exporter.export(enface_filename)

        print("save_raw circ")
        circ_img = (np.flipud((np.clip((self.frame_3D - self.circ_lowVal) / (self.circ_highVal - self.circ_lowVal), 0, 1)) * 255)).astype(np.uint8)
        cv2_image = cv2.cvtColor(circ_img, cv2.COLOR_GRAY2RGB)
        export_filename = self.directory+"/measurement_result/r="+self.ui.Edit_radius.toPlainText()+'/resampled_raw_img.tiff'
        #while os.path.exists(export_filename):
        cv2.imwrite(export_filename, cv2_image)


    def save_marked_circ_measurement_plot(self):
        #self.ui.circ_Slow_Axis.autoRange()
        self.ui.Enface_circ.autoRange()
        exporter = ImageExporter(self.ui.Enface_circ)
        exporter.parameters()['width'] = 350
        if not os.path.exists(self.directory+"/measurement_result"):
            os.makedirs(self.directory+"/measurement_result")
        if not os.path.exists(self.directory+"/measurement_result/r="+self.ui.Edit_radius.toPlainText()):
            os.makedirs(self.directory+"/measurement_result/r="+self.ui.Edit_radius.toPlainText())
        i=1
        enface_filename = self.directory+"/measurement_result/r="+self.ui.Edit_radius.toPlainText()+'/enface.tif'
        exporter.export(enface_filename)
        
        #self.circ_resample_remove_angle_grid()
        exporter = ImageExporter(self.ui.circ_Slow_Axis)
        exporter.parameters()['width'] = np.shape(self.frame_3D)[1]
        export_filename = self.directory+"/measurement_result/r="+self.ui.Edit_radius.toPlainText()+'/marked_circ_bscan_r='+self.ui.Edit_radius.toPlainText()+'_center='+self.ui.Edit_circ_x.toPlainText()+','+self.ui.Edit_circ_y.toPlainText()+'_'+str(i)+'.tif'
        while os.path.exists(export_filename):
            i += 1
            export_filename = self.directory+"/measurement_result/r="+self.ui.Edit_radius.toPlainText()+'/marked_circ_bscan_r='+self.ui.Edit_radius.toPlainText()+'_center='+self.ui.Edit_circ_x.toPlainText()+','+self.ui.Edit_circ_y.toPlainText()+'_'+str(i)+'.tif'
        exporter.export(export_filename)
        #self.circ_resample_add_angle_grid()

    def set_note_toggle(self):
        measure_note = self.ui.measurement_note.toPlainText()
        self.ui.Slow_Axis.set_measurement_note(measure_note)
        
    def set_circ_note_toggle(self):
        measure_note = self.ui.measurement_note_circ.toPlainText()
        self.ui.circ_Slow_Axis.set_measurement_note(measure_note)

    def flip(self):
        self.flip_bscan = not self.flip_bscan
        if self.image_ready:
            if self.flip_bscan:
                self.ui.Slow_Axis.flipped = True
                self.ui.Slow_Axis.invertY(True)
            else:
                self.ui.Slow_Axis.flipped = False
                self.ui.Slow_Axis.invertY(False)
        memory_usage = sys.getsizeof(self.frame_3D_resh)/(1024*1024)
        print(f"Memory usage of frame_3D_resh: {memory_usage} MB")
        memory_usage = sys.getsizeof(self.frame_3D_20log)/(1024*1024)
        print(f"Memory usage of frame_3D_20log: {memory_usage} MB")
        memory_usage = sys.getsizeof(self.ui)/(1024*1024)
        print(f"Memory usage of ui: {memory_usage} MB")
        

    def go_prev(self):
        if self.cur_img_slow_no > 0:
            self.cur_img_slow_no -= 1
            self.slow_axis_img_np = cv2.resize(((np.clip((self.frame_3D_resh[:,:,self.cur_img_slow_no] - self.oct_lowVal) / (self.oct_highVal - self.oct_lowVal), 0, 1)) * 255).astype(np.uint8),(0,0),fx=4,fy=4)
            self.ui.Slow_Axis.clear()
            slow_img = pg.ImageItem(np.flipud(np.rot90(self.slow_axis_img_np)))
            slow_img.setRect(0,0,1024,1024)
            self.ui.Slow_Axis.addItem(slow_img, clear=True)
            self.ui.Slow_Axis.reset_measure()
            self.ui.distance.setText(" ")
            self.red_line.setValue((self.frame_3D_resh.shape[2]-1-self.cur_img_slow_no)*self.enface_divider*(1024/(self.frame_3D_resh.shape[2]*self.enface_divider)))

    
    def go_next(self):
        if self.cur_img_slow_no < self.frame_3D_resh.shape[2]-1:
            self.cur_img_slow_no += 1
            self.slow_axis_img_np = cv2.resize(((np.clip((self.frame_3D_resh[:,:,self.cur_img_slow_no] - self.oct_lowVal) / (self.oct_highVal - self.oct_lowVal), 0, 1)) * 255).astype(np.uint8),(0,0),fx=4,fy=4)
            self.ui.Slow_Axis.clear()
            slow_img = pg.ImageItem(np.flipud(np.rot90(self.slow_axis_img_np)))
            slow_img.setRect(0,0,1024,1024)
            self.ui.Slow_Axis.addItem(slow_img, clear=True)
            self.ui.Slow_Axis.reset_measure()
            self.ui.distance.setText(" ")
            self.red_line.setValue((self.frame_3D_resh.shape[2]-1-self.cur_img_slow_no)*self.enface_divider*(1024/(self.frame_3D_resh.shape[2]*self.enface_divider)))

    def octa_save_img(self):
        image = Image.fromarray(((np.clip((self.frame_OCTA[:,:,self.cur_octa_img_slow_no] - self.octa_lowVal) / (self.octa_highVal - self.octa_lowVal), 0, 1)) * 255).astype(np.uint8))
        image.save(self.directory+'/OCT_Reconstruct_'+str(self.cur_octa_img_slow_no)+'_octa.tiff')
        return

    def octa_go_prev(self):
        if self.cur_img_slow_no > 0:
            self.cur_octa_img_slow_no -= 1
            self.octa_slow_axis_img_np = ((np.clip((self.frame_OCTA[:,:,self.cur_octa_img_slow_no] - self.octa_lowVal) / (self.octa_highVal - self.octa_lowVal), 0, 1)) * 255).astype(np.uint8)
            self.ui.OCTA_Slow_Axis.clear()
            OCTA_slow_img = pg.ImageItem(np.rot90(self.octa_slow_axis_img_np))
            OCTA_slow_img.setRect(0,0,1024,1024)
            self.ui.OCTA_Slow_Axis.addItem(OCTA_slow_img, clear=True)
            self.ui.OCTA_Slow_Axis.reset_measure()
            self.ui.distance.setText(" ")
            self.octa_red_line.setValue(self.frame_OCTA.shape[2]-1-self.cur_octa_img_slow_no)

    def octa_go_next(self):
        if self.cur_img_slow_no < self.frame_OCTA.shape[2]-1:
            self.cur_octa_img_slow_no += 1
            self.octa_slow_axis_img_np = ((np.clip((self.frame_OCTA[:,:,self.cur_octa_img_slow_no] - self.octa_lowVal) / (self.octa_highVal - self.octa_lowVal), 0, 1)) * 255).astype(np.uint8)
            self.ui.OCTA_Slow_Axis.clear()
            OCTA_slow_img = pg.ImageItem(np.rot90(self.octa_slow_axis_img_np))
            OCTA_slow_img.setRect(0,0,1024,1024)
            self.ui.OCTA_Slow_Axis.addItem(OCTA_slow_img, clear=True)
            self.ui.OCTA_Slow_Axis.reset_measure()
            self.ui.distance.setText(" ")
            self.octa_red_line.setValue(self.frame_OCTA.shape[2]-1-self.cur_octa_img_slow_no)

    def Enface_red_dragged(self,obj):
        if(int(obj.value())> 0 and obj.value() <= 1024): 
            self.cur_img_slow_no = int(self.frame_3D_resh.shape[2]-(obj.value()/1024)*self.frame_3D_resh.shape[2])
            self.slow_axis_img_np = cv2.resize(((np.clip((self.frame_3D_resh[:,:,self.cur_img_slow_no] - self.oct_lowVal) / (self.oct_highVal - self.oct_lowVal), 0, 1)) * 255).astype(np.uint8),(0,0),fx=4,fy=4)
            self.ui.Slow_Axis.clear()
            slow_img = pg.ImageItem(np.flipud(np.rot90(self.slow_axis_img_np)))
            slow_img.setRect(0,0,1024,1024)
            self.ui.Slow_Axis.addItem(slow_img, clear=True)
            self.ui.Slow_Axis.reset_measure()
            self.ui.Slow_Axis.setBscan_number(int(self.cur_img_slow_no*4))
            self.ui.bscan_number_label.setText(self._translate("MainWindow", "B-scan #:"+str(self.cur_img_slow_no*4)))
            self.ui.distance.setText(" ")

    def OCTA_Enface_red_dragged(self,obj):
        if(obj.value()> 0 and obj.value() <= 1024): 
            self.cur_octa_img_slow_no = int(self.frame_OCTA.shape[2]-(obj.value()/1024)*self.frame_3D_resh.shape[2])
            print(self.cur_octa_img_slow_no)
            self.octa_slow_axis_img_np = ((np.clip((self.frame_OCTA[:,:,self.cur_octa_img_slow_no] - self.octa_lowVal) / (self.octa_highVal - self.octa_lowVal), 0, 1)) * 255).astype(np.uint8)
            self.ui.OCTA_Slow_Axis.clear()
            OCTA_slow_img = pg.ImageItem(np.flipud(np.rot90(self.octa_slow_axis_img_np)))
            OCTA_slow_img.setRect(0,0,1024,1024)
            self.ui.OCTA_Slow_Axis.addItem(OCTA_slow_img, clear=True)
            self.ui.OCTA_Slow_Axis.reset_measure()
            self.ui.distance.setText(" ")


    def updateImg(self,enface,slow_axis,lowVal,highVal,is_octa):
        if is_octa:
            slow_axis_img_np = ((np.clip((self.frame_OCTA[:,:,self.cur_octa_img_slow_no] - lowVal) / (highVal - lowVal), 0, 1)) * 255).astype(np.uint8)
            #self.ui.octa_label.setText(self._translate("MainWindow", "Min value: "+str(round(lowVal,3))))
            #self.ui.octa_label2.setText(self._translate("MainWindow", "Max value: "+str(round(highVal,3))))
        else:
            slow_axis_img_np = ((np.clip((self.frame_3D_resh[:,:,self.cur_img_slow_no] - lowVal) / (highVal - lowVal), 0, 1)) * 255).astype(np.uint8)
            #self.ui.oct_label.setText(self._translate("MainWindow", "Min value: "+str(round(lowVal,3))))
            #self.ui.oct_label2.setText(self._translate("MainWindow", "Max value: "+str(round(highVal,3))))
        
        OCTA_slow_img = pg.ImageItem(np.flipud(np.rot90(slow_axis_img_np)))
        OCTA_slow_img.setRect(0,0,1024,1024)
        slow_axis.clear()
        slow_axis.addItem(OCTA_slow_img, clear=True)
        slow_axis.reset_measure()
        #fast_axis.clear()
        

    def contrast(self,value):
        self.oct_lowVal = value[0]#(value[0])*self.oct_step-35#+self.lowVal                                                      #lower lowbound and higher upperbound for bscan
        self.oct_highVal = value[1]#(value[1])*self.oct_step-35#+self.highVal
        print("low:",self.oct_lowVal)
        print("high:",self.oct_highVal)
        self.ui.Slow_Axis.clear()
        self.ui.Edit_min.setText(str(round(value[0],3)))
        self.ui.Edit_max.setText(str(round(value[1],3)))
        self.updateImg(self.ui.Enface,self.ui.Slow_Axis,self.oct_lowVal,self.oct_highVal,False)

    def contrast_octa(self,value):
        self.octa_lowVal = (value[0])*self.octa_step+self.octa_lowVal_base                                                      #lower lowbound and higher upperbound for bscan
        self.octa_highVal = (value[1]-99)*self.octa_step+self.octa_highVal_base
        print("low:",self.octa_lowVal)
        print("high:",self.octa_highVal)
        self.ui.OCTA_Slow_Axis.clear()
        self.updateImg(self.ui.OCTA_Enface,self.ui.OCTA_Slow_Axis,self.octa_lowVal,self.octa_highVal,True)


    def gamma(self):
        return
    
    def select_pixmap(self):
        
        dlg2 = QFileDialog()
        cur_dir = QDir.currentPath()
        dlg2.setDirectory(cur_dir+r'\Pixel Maps')
        dlg2.setFileMode(QFileDialog.AnyFile)
        if dlg2.exec():
            filenames = dlg2.selectedFiles()
            f = filenames[0]
            pixelmap_file = open(cur_dir+"\pixelmap.txt", "w")
            pixelmap_file.write(f)
            pixelmap_file.close()
            #match_Path = filenames[0]

            with open(cur_dir+"/pixelmap.txt", 'r') as file:
                self.match_Path = str(file.read().rstrip())

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("You are ready to go")
            msg_box.setText('Pixel map has been set.')
            msg_box.exec()
        else:
            return
        #self.select_raw_file()
        
    
    def open_file(self):
        self.ui.label.setText(self._translate("MainWindow", "Status: VisOCT Explorer(Selecting File...)"))
        cur_dir = QDir.currentPath()

        if self.open_file_dialog.exec():
            filenames = self.open_file_dialog.path_edit.text()
            print(filenames)
            meta_data_dict = {}

            self.image_ready = False
            self.ui.open_file_btn.clicked.disconnect(self.open_file)
            self.ui.open_file_btn.clicked.connect(self.wait_warning)
            self.ui.stackedWidget.setCurrentIndex(0)
                
            self.ui.enface_otherdim.setEnabled(False)
            self.ui.fibergram_btn.setEnabled(False)
            self.ui.octa_btn.setEnabled(False)
            self.ui.circ_btn.setEnabled(False)
            self.ui.pushButton_4.setEnabled(False)

            
            meta_data_dict["envelope"] = str(self.system_setting.enable_extraction.isChecked())
            meta_data_dict["chunks"] = int(self.system_setting.Edit_num_chunk.value())
            meta_data_dict["pixel"] = str(self.open_file_dialog.path_edit_pixmap.text())
            
            if not self.open_file_dialog.wave_path_edit.text():
                meta_data_dict["wavelength"] = 'Wavelength Files/wavelength_blizz_06'
            else:
                meta_data_dict["wavelength"] = str(self.open_file_dialog.path_edit_pixmap.text())
                
            if "Bal" in filenames:
                from_fname = True
            else:
                from_fname = False
                meta_data_dict["balanced"] = self.open_file_dialog.balanced_rb.isChecked()
                meta_data_dict["bidirection"] = self.open_file_dialog.bi_rb.isChecked()
                meta_data_dict["res_axis"] = int(self.open_file_dialog.axis_input.text())
                meta_data_dict["res_fast"] = int(self.open_file_dialog.aline_input.text())
                meta_data_dict["res_slow"] = int(self.open_file_dialog.bscan_input.text())
                meta_data_dict["repNum"] = int(self.open_file_dialog.bscan_rep_input.text())
                meta_data_dict["volNum"] = int(self.open_file_dialog.volume_input.text())
                meta_data_dict["xrng"] = int(self.open_file_dialog.scan_range_x_input.text())
                meta_data_dict["yrng"] = int(self.open_file_dialog.scan_range_y_input.text())
                meta_data_dict["zrng"] = int(self.open_file_dialog.scan_range_z_input.text())

            #print("envelope: ",meta_data_dict["envelope"])

            pixelmap_file = open(cur_dir+"\pixelmap.txt", "w")
            pixelmap_file.write(meta_data_dict["pixel"])
            pixelmap_file.close()
            self.processParameters = processParams(meta_data_dict["chunks"], filenames, from_fname, meta_data_dict, meta_data_dict["pixel"])

            if self.processParameters.balFlag:
                self.ui.balancing_label.setText(self._translate("MainWindow", "Balanced: Yes"))
            else:
                self.ui.balancing_label.setText(self._translate("MainWindow", "Balanced: No"))
                
            self.ui.averaging_dropdown.setText("4")
            self.ui.filename_label.setText(self._translate("MainWindow", "Filename: "+os.path.basename(filenames)[:-4]))

            if self.processParameters.octaFlag:
                self.ui.scan_protocol_label.setText(self._translate("MainWindow", "Scan protocol: OCTA"))
            elif self.processParameters.SRFlag:
                self.ui.scan_protocol_label.setText(self._translate("MainWindow", "Scan protocol: SR raster"))
            else:
                self.ui.scan_protocol_label.setText(self._translate("MainWindow", "Scan protocol: Raster"))
            
            self.ui.textEdit_2.append("Image file load succeed. Processing...")
            self.progressValue = 0
            self.ui.progressBar.setValue(self.progressValue)
            self.status = " (Processing...)"
            self.ui.label.setText(self._translate("MainWindow", "Status: VisOCT Explorer"+self.status))
            
            file_name = "saved_setting.txt"
            current_directory = os.getcwd()
            file_path = os.path.join(current_directory, file_name)
                
            self.process.set_attrib(self.processParameters)
            self.process.start()
            self.dm_tab = Distance_Measurement_Tab(self.processParameters.excel_fname)
            self.ui.Slow_Axis.setMeasurement_result_table(self.dm_tab)
            #self.ui.Fast_Axis.setMeasurement_result_table(self.dm_tab)
            self.ui.dm_table.clicked.connect(self.show_dm_tab)
            

    def show_dm_tab(self):
        self.dm_tab.show()
        self.dm_tab.raise_()

    def wait_warning(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Warning")
        msg_box.setText("Please wait until current processing to be finished!")
        msg_box.exec()

    def set_pixel_map(self):
        dlg2 = QFileDialog()
        cur_dir = QDir.currentPath()
        dlg2.setDirectory(cur_dir+r'\Pixel Maps')
        dlg2.setFileMode(QFileDialog.AnyFile)
        if dlg2.exec():
            filenames = dlg2.selectedFiles()
            f = filenames[0]
            pixelmap_file = open(cur_dir+"\pixelmap.txt", "w")
            pixelmap_file.write(f)
            pixelmap_file.close()

    def updateProgress(self,progress_num):
        if progress_num == 0:
            self.progressValue = 0
            self.ui.progressBar.setValue(0)
        if progress_num == 1:
            self.progressValue += 1
            self.ui.progressBar.setValue(int(self.progressValue))
        if progress_num == 2:
            self.progressValue += 2.5
            self.ui.progressBar.setValue(int(self.progressValue))
        if progress_num == 3:
            self.progressValue += 0.4
            self.ui.progressBar.setValue(int(self.progressValue))
        if progress_num == 4:
            self.progressValue += 6
            self.ui.progressBar.setValue(int(self.progressValue))
        if progress_num == 5:
            self.progressValue += 16./self.processParameters.res_slow
            self.ui.progressBar.setValue(int(self.progressValue))
        if progress_num == 6:
            self.progressValue += 5
            self.ui.progressBar.setValue(int(self.progressValue))
        if progress_num == 7:
            self.ui.progressBar.setValue(100)

    def progress_callback(self,progress, time=0):
        scroll_bar = self.ui.textEdit_2.verticalScrollBar()
        if progress == 0:
            self.ui.textEdit_2.append("Balancing fringes...")
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 1:
            self.ui.textEdit_2.append("Fringes balancing completed. Time used: "+str(time))
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 2:
            self.ui.textEdit_2.append("Resampled to linear in k")
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 3:
            self.ui.textEdit_2.append("Starting dispersion compensation...")
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 4:
            self.ui.textEdit_2.append("Dispersion compensated. Time used: "+str(time))
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 5:
            self.ui.textEdit_2.append("Starting OCT reconstruction...")
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 6:
            self.ui.textEdit_2.append("OCT Reconstruction completed. Time used: "+str(time))
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 7:
            self.ui.textEdit_2.append("Starting image registration...")
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 8:
            self.ui.textEdit_2.append("Image Registration completed. Time used: "+str(time))
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 17:
            self.ui.textEdit_2.append("Adjusting contrast...")
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 18:
            self.ui.textEdit_2.append("Min/max pixel value calculated.")
            scroll_bar.setValue(scroll_bar.maximum())
        if progress == 9:
            #self.ui.textEdit_2.append("Start processing B-scan fly through video...")
            scroll_bar.setValue(scroll_bar.maximum())
            
            if self.processParameters.octaFlag:
                self.ui.octa_contrast_Slider.setEnabled(True)
                self.ui.octa_contrast_Slider.valueChanged.connect(self.contrast_octa)
            #self.ui.horizontalSlider_2.setEnabled(True)
            #self.ui.horizontalSlider_2.valueChanged.connect(self.brightness)
        if progress == 10:
            self.ui.textEdit_2.append("Reconstruction completed and ready to be viewed")
            scroll_bar.setValue(scroll_bar.maximum())
            self.ui.open_file_btn.clicked.disconnect(self.wait_warning)
            self.ui.open_file_btn.clicked.connect(self.open_file)

            #if self.processParameters.octaFlag:
                #self.ui.tsa_btn_1.setEnabled(True)
                #self.ui.tsa_btn_2.setEnabled(True)
            
            self.ui.circ_btn.setEnabled(True)
            self.ui.pixel_map_btn.setEnabled(True)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    ## loading style file
    # with open("style.qss", "r") as style_file:
    #     style_str = style_file.read()
    # app.setStyleSheet(style_str)

    #style_file = QFile("style.qss")
    #style_file.open(QFile.ReadOnly | QFile.Text)
    #style_stream = QTextStream(style_file)
    #app.setStyleSheet(style_stream.readAll())


    window = MainWindow()
    window.show()

    sys.exit(app.exec())