
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys,cv2
from skimage import exposure
from PySide6 import QtCore,QtGui,QtWidgets
from qtrangeslider import QDoubleRangeSlider
from sidebar_ui import NumberOnlyTextEdit
from pyqtgraph import PlotWidget,GraphicsLayoutWidget,PlotItem,widgets,GraphicsLayout,GraphicsView

class enface_3D(QtWidgets.QWidget):
    def __init__(self,oct_volume = None,cur_enface = None):
        #self.app = QtWidgets.QApplication(sys.argv)
        super().__init__()
        self.setWindowTitle('Enface 3D view')
        self.setGeometry(150, 150, 532, 610)
        self.setFixedSize(532,610)
        # Set up layout and a simple label
        self.layout = QtWidgets.QVBoxLayout()
        #label = QtWidgets.QLabel('This is a new window!', self)
        #layout.addWidget(label)

        
        self.Enface_widget = GraphicsLayoutWidget()
        self.Enface_widget.setGeometry(QtCore.QRect(20, 20, 512, 512))
        self.Enface_widget.ci.setContentsMargins(0, 0, 0, 0)
        self.Enface_widget.setFixedSize(512,512)
        self.Enface = self.Enface_widget.addPlot()
        #placeholder = np.array(Image.open("./icon/Logo.png"))
        #self.Enface.addItem(pg.ImageItem(placeholder),clear=True)
        self.Enface.hideAxis('left')
        self.Enface.hideAxis('bottom')
        self.Enface.setLimits(xMin=0, xMax=1024, yMin=0, yMax=1024)
        self.Enface.setAspectLocked(True)
        self.layout.addWidget(self.Enface_widget)

        if oct_volume.any():
            self.volume = oct_volume
            enface = pg.ImageItem(np.fliplr(cur_enface))
            enface.setRect(0,0,1024,1024)
            self.Enface.addItem(enface,clear = True)

        #self.setFixedSize(QtCore.QSize(800, 800))
        
        font = QtGui.QFont('Arial', 10)
        self.Edit_z_min = NumberOnlyTextEdit()
        self.Edit_z_min.setFixedSize(QtCore.QSize(60, 27))
        self.Edit_z_min.setObjectName("Edit_circ_x")
        self.Edit_z_min.setText("0")
        self.Edit_z_max = NumberOnlyTextEdit()
        self.Edit_z_max.setFixedSize(QtCore.QSize(60, 27))
        self.Edit_z_max.setObjectName("Edit_circ_x")
        self.Edit_z_max.setText("1024")

        projection_type = QtWidgets.QLabel('Projection Type:')
        self.projection_dropdown = QtWidgets.QComboBox()
        self.projection_dropdown.addItem("Mean intensity projection")
        self.projection_dropdown.addItem("Max intensity projection")
        self.projection_dropdown.setCurrentIndex(0)
        self.projection_dropdown.setStyleSheet("QComboBox { min-height: 27px;max-height: 27px; max-width: 180px; }")
        self.projection_dropdown.setFont(font)

        min_label = QtWidgets.QLabel('Range:')
        min_label.setFont(font)
        max_label = QtWidgets.QLabel('to:')
        max_label.setFont(font)
        along_axis = QtWidgets.QLabel('Along:')
        self.axis_dropdown = QtWidgets.QComboBox()
        self.axis_dropdown.addItem("Depth")
        self.axis_dropdown.addItem("Slow Axis")
        self.axis_dropdown.addItem("Fast Axis")
        self.axis_dropdown.setCurrentIndex(0)
        self.axis_dropdown.setStyleSheet("QComboBox { min-height: 27px;max-height: 27px; max-width: 100px; }")
        self.axis_dropdown.setFont(font)
        self.axis_dropdown.currentIndexChanged.connect(self.index_changed)

        self.update_btn = QtWidgets.QPushButton("Update Enface")
        self.update_btn.setStyleSheet("color: black")
        self.update_btn.setCheckable(False)
        self.update_btn.setObjectName("update_btn")
        self.update_btn.clicked.connect(self.update_enface)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.addWidget(min_label)
        self.horizontalLayout.addWidget(self.Edit_z_min)
        #spacerItem6_circ = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        #self.horizontalLayout.addItem(spacerItem6_circ)
        self.horizontalLayout.addWidget(max_label)
        self.horizontalLayout.addWidget(self.Edit_z_max)
        spacerItem0 = QtWidgets.QSpacerItem(120, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem0)
        self.horizontalLayout.addWidget(self.update_btn)
        
        self.layout.addLayout(self.horizontalLayout)


        self.horizontalLayout2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout2.addWidget(along_axis)
        self.horizontalLayout2.addWidget(self.axis_dropdown)
        self.horizontalLayout2.addWidget(projection_type)
        self.horizontalLayout2.addWidget(self.projection_dropdown)
        spacerItem1 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout2.addItem(spacerItem1)
        self.layout.addLayout(self.horizontalLayout2)
        self.setLayout(self.layout)

        self.z_depth = QtWidgets.QLabel("Set Z range")
        self.z_depth.setObjectName("label")
        self.z_depth.setFont(font)
        #self.slider.valueChanged.connect(self.update_z_range)
        
        #self.layout = QtWidgets.QVBoxLayout()  # Create a layout
        #self.layout.addLayout(self.horizontalLayout)
        #self.layout.addWidget(self.contrast_Slider_circ)
        #self.layout.addWidget(self.slider)  # Add slider to layout
        #self.widget = QtWidgets.QWidget()  # Create a widget
        #self.widget.setLayout(self.layout)  # Set layout for the widget
        #self.widget.setWindowTitle('3D Visualization')
        #self.widget.setGeometry(0, 110, 800, 900)
        self.show()  # Show the widget
        #max_along_z = np.max(((np.clip((OCT_volume[:,:,:] - low_Val) / (high_Val - low_Val), 0, 1)) * 255),axis=(1, 2))
        i_min = 0
        i_max = 512


    def update_enface(self):
        if self.projection_dropdown.currentIndex() == 0:
            mean_proj = True
        else:
            mean_proj = False
        enface_axis = self.axis_dropdown.currentIndex()
        if mean_proj:
            enface = np.squeeze(np.mean(self.volume,axis=enface_axis))
            enface = enface/np.mean(enface,axis=0)
            enface = enface/np.max(enface)
            enface = exposure.equalize_adapthist(enface,clip_limit=0.003)
        else:
            enface = np.squeeze(np.max(self.volume,axis=enface_axis))
            enface = enface/np.mean(enface,axis=0)
            enface = enface/np.max(enface)
            enface = exposure.equalize_adapthist(enface,clip_limit=0.03)
            
        if enface_axis != 0:
            enface = np.rot90(cv2.resize(enface,(512,512),interpolation=cv2.INTER_LINEAR))
        else:
            enface = cv2.resize(enface,(512,512),interpolation=cv2.INTER_LINEAR)
        enface = pg.ImageItem(np.fliplr(enface))
        enface.setRect(0,0,1024,1024)
        self.Enface.clear()
        self.Enface.addItem(enface,clear = True)
        

    def index_changed(self,index):
        if index!=0:
            self.Edit_z_max.setText("512")
        else:
            self.Edit_z_max.setText("1024")

        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    vis = enface_3D()
    vis.show()
    sys.exit(app.exec())
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