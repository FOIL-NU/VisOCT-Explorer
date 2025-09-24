
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
from PySide6 import QtCore,QtGui,QtWidgets
from qtrangeslider import QDoubleRangeSlider
from sidebar_ui import NumberOnlyTextEdit

class Visualizer(object):
    def __init__(self,OCT_volume,low_Val,high_Val):
        #self.app = QtWidgets.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.z_range = [0,512]
        self.w.opts['distance'] = 1500
        self.w.setFixedSize(QtCore.QSize(800, 800))
        
        font = QtGui.QFont('Arial', 10)
        self.Edit_z_min = NumberOnlyTextEdit()
        self.Edit_z_min.setFixedSize(QtCore.QSize(100, 27))
        self.Edit_z_min.setObjectName("Edit_circ_x")
        self.Edit_z_min.setText("0")
        self.Edit_z_max = NumberOnlyTextEdit()
        self.Edit_z_max.setFixedSize(QtCore.QSize(100, 27))
        self.Edit_z_max.setObjectName("Edit_circ_x")
        self.Edit_z_max.setText("512")
        min_label = QtWidgets.QLabel('Depth range start from:')
        min_label.setFont(font)
        max_label = QtWidgets.QLabel('to:')
        max_label.setFont(font)
        self.update_btn = QtWidgets.QPushButton("Update Volume")
        self.update_btn.setStyleSheet("color: black")
        self.update_btn.setCheckable(False)
        self.update_btn.setObjectName("update_btn")
        self.update_btn.clicked.connect(self.update_z_range)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.addWidget(min_label)
        self.horizontalLayout.addWidget(self.Edit_z_min)
        #spacerItem6_circ = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        #self.horizontalLayout.addItem(spacerItem6_circ)
        self.horizontalLayout.addWidget(max_label)
        self.horizontalLayout.addWidget(self.Edit_z_max)
        self.horizontalLayout.addWidget(self.update_btn)

        self.pix_value_label = QtWidgets.QLabel("Min pixel value: "+str(round(low_Val,3))+"                Max pixel value: "+str(round(high_Val,3)))
        self.contrast_Slider_circ = QDoubleRangeSlider()
        self.contrast_Slider_circ.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.contrast_Slider_circ.setSingleStep(0.01)
        self.contrast_Slider_circ.setValue((low_Val,high_Val))
        self.contrast_Slider_circ.valueChanged.connect(self.update_pix_val)

        self.z_depth = QtWidgets.QLabel("Set Z range")
        self.z_depth.setObjectName("label")
        self.z_depth.setFont(font)
        #self.slider.valueChanged.connect(self.update_z_range)
        
        self.layout = QtWidgets.QVBoxLayout()  # Create a layout
        self.layout.addWidget(self.w)  # Add GLViewWidget to layout
        self.layout.addLayout(self.horizontalLayout)
        self.layout.addWidget(self.pix_value_label)
        self.layout.addWidget(self.contrast_Slider_circ)
        #self.layout.addWidget(self.slider)  # Add slider to layout
        self.widget = QtWidgets.QWidget()  # Create a widget
        self.widget.setLayout(self.layout)  # Set layout for the widget
        self.widget.setWindowTitle('3D Visualization')
        self.widget.setGeometry(0, 110, 800, 900)
        self.widget.show()  # Show the widget
        #max_along_z = np.max(((np.clip((OCT_volume[:,:,:] - low_Val) / (high_Val - low_Val), 0, 1)) * 255),axis=(1, 2))
        i_min = 0
        i_max = 512
        # Create a random 3D numpy array as an example
        self.low_Val = low_Val
        self.high_Val = high_Val
        self.OCT_volume = OCT_volume
        volume_data = np.ones((i_max-i_min,np.shape(OCT_volume)[1],np.shape(OCT_volume)[2],4))
        for z_index in range(i_min,i_max):
            volume_data[z_index-i_min,:,:,0] = (np.clip((OCT_volume[z_index,:,:] - low_Val) / (high_Val - low_Val), 0, 1) * 255).astype(np.ubyte)
            
        volume_data[:,:,:,1] = volume_data[:,:,:,0]
        volume_data[:,:,:,2] = volume_data[:,:,:,0]


        # Create GLVolumeItem
        self.volume_item = gl.GLVolumeItem(volume_data)
        self.volume_item.translate(-(i_max-i_min)/2 , -512/2 , -512/2 )  # Center the volume
        self.w.addItem(self.volume_item)
        del volume_data

    def update_z_range(self):
        i_min = int(self.Edit_z_min.toPlainText())
        i_max = int(self.Edit_z_max.toPlainText())
        self.w.clear()
        volume_data = np.ones((i_max-i_min,np.shape(self.OCT_volume)[1],np.shape(self.OCT_volume)[2],4))
        for z_index in range(i_min,i_max):
            volume_data[z_index-i_min,:,:,0] = (np.clip((self.OCT_volume[z_index,:,:] - self.low_Val) / (self.high_Val - self.low_Val), 0, 1) * 255).astype(np.ubyte)
            
        volume_data[:,:,:,1] = volume_data[:,:,:,0]
        volume_data[:,:,:,2] = volume_data[:,:,:,0]
        self.volume_item = gl.GLVolumeItem(volume_data)
        self.volume_item.translate(-(i_max-i_min)/2 , -512/2 , -512/2 )  # Center the volume
        self.w.addItem(self.volume_item)

    def update_pix_val(self,value):
        self.pix_value_label.setText("Min pixel value: "+str(round(value[0],3))+"                Max pixel value: "+str(round(value[1],3)))
        self.low_Val = value[0]
        self.high_Val = value[1]

        
if __name__ == '__main__':
    vis = Visualizer()
    vis.start()
    
'''
class Visualizer(object):
    def __init__(self):
        self.traces = dict()
        self.app = QtWidgets.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 40
        self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
        self.w.setGeometry(0, 110, 1920, 1080)
        self.w.show()

        # create the background grids
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 0)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, 0)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, -10)
        self.w.addItem(gz)

        self.n = 50
        self.m = 1000
        self.y = np.linspace(-10, 10, self.n)
        self.x = np.linspace(-10, 10, self.m)
        self.phase = 0

        for i in range(self.n):
            yi = np.array([self.y[i]] * self.m)
            d = np.sqrt(self.x ** 2 + yi ** 2)
            z = 10 * np.cos(d + self.phase) / (d + 1)
            pts = np.vstack([self.x, yi, z]).transpose()
            self.traces[i] = gl.GLLinePlotItem(pos=pts, color=pg.glColor(
                (i, self.n * 1.3)), width=(i + 1) / 10, antialias=True)
            self.w.addItem(self.traces[i])

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

    def set_plotdata(self, name, points, color, width):
        self.traces[name].setData(pos=points, color=color, width=width)

    def update(self):
        
        for i in range(self.n):
            yi = np.array([self.y[i]] * self.m)
            d = np.sqrt(self.x ** 2 + yi ** 2)
            z = 10 * np.cos(d + self.phase) / (d + 1)
            pts = np.vstack([self.x, yi, z]).transpose()
            self.set_plotdata(
                name=i, points=pts,
                color=pg.glColor((i, self.n * 1.3)),
                width=(i + 1) / 10
            )
            self.phase -= .003

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    v = Visualizer()
    v.animation()'''