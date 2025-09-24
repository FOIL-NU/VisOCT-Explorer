from PySide6.QtCore import (Qt,QFile, QTextStream, Signal, QDir,QThread,QEvent,QPoint,QCoreApplication)
from PySide6.QtGui import (QColor, QFont, QMovie)
from PySide6.QtWidgets import (QApplication,QSlider, QTableWidget, QTableWidgetItem, QDialog, QMessageBox, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget,QPushButton, QFileDialog, QComboBox,QTextEdit)
import sys
import psutil
import csv


class Distance_Measurement_Tab(QMainWindow):
    def __init__(self,excel_fname):
        super().__init__()
        self.init_ui(excel_fname)

    def init_ui(self,excel_fname):
        self.setWindowTitle('Distance Measurement Table')
        self.setGeometry(100, 100, 400, 400)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout(self.central_widget)
        self.save_btn = QPushButton(parent=self.central_widget)
        self.save_btn.setText("Save results")
        self.table_widget = QTableWidget()
        
        layout.addWidget(self.save_btn)
        layout.addWidget(self.table_widget)
        self.fpath = excel_fname+'.xlsx'
        self.save_btn.clicked.connect(self.save_to_file)
        self.load_excel_data()  # Replace with your Excel file path


    def load_excel_data(self):
        row = ["Vertical Distance(um)", "Horizontal Distance(um)", "Hypotenuse Distance(um)", "Bscan #","Note"]
        print("loading...")
        self.table_widget.setColumnCount(len(row))
        self.table_widget.insertRow(0)
        for col_index, cell in enumerate(row):
            item = QTableWidgetItem(str(cell))
            self.table_widget.setItem(0, col_index, item)

    def get_cur_row_num(self):
        return self.table_widget.rowCount()
    
    def write_to_table(self,row):
        #worksheet = self.workbook.active
        row_num = self.table_widget.rowCount()
        #self.table_widget.setColumnCount(len(row))
        self.table_widget.insertRow(row_num)
        for col_index, cell in enumerate(row):
            item = QTableWidgetItem(str(cell))
            self.table_widget.setItem(row_num, col_index, item)
        
        #self.workbook.save(self.fpath)

    
    def save_to_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Table to CSV", "", "CSV Files (*.csv);;All Files (*)")

        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for row in range(self.table_widget.rowCount()):
                    row_data = []
                    for column in range(self.table_widget.columnCount()):
                        item = self.table_widget.item(row, column)
                        if item is not None:
                            row_data.append(item.text())
                        else:
                            row_data.append("")
                    writer.writerow(row_data)


if __name__ == "__main__":
    app = QApplication(sys.argv)