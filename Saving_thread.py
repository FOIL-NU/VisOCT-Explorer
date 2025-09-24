import cupy as cp
import sys
from PySide6.QtCore import (Qt,QFile, QTextStream, Signal, QDir,QThread,QEvent,QPoint,QCoreApplication)
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid
from pydicom import DataElement, Dataset
from pydicom.tag import Tag
from datetime import datetime
import numpy as np
import cv2,os
from matplotlib import pyplot as plt
import registration
from PIL import Image

class SavingThread(QThread):
    finished = Signal()

    def __init__(self, processParameters,frame_3D_resh,selected_option,directory,average_number,lowVal,highVal):
        self.processParameters = processParameters
        self.frame_3D_resh = frame_3D_resh
        self.selected_option = selected_option
        self.average_number = average_number
        self.directory = directory
        self.lowVal = lowVal
        self.highVal = highVal
        super().__init__()

    def dicom_add_custom_elements(self, ds):
        # Define private tags for custom elements
        bscan_tag = Tag(0x0011, 0x1010)
        frame_tag = Tag(0x0011, 0x1020)
        oct_type_tag = Tag(0x0011, 0x1030)
        scanning_pattern_tag = Tag(0x0011, 0x1040)
        xrange_tag = Tag(0x0011, 0x1050)
        yrange_tag = Tag(0x0011, 0x1060)

        # Add custom elements to the DICOM dataset
        ds.add(DataElement(bscan_tag, "IS", int(self.processParameters.res_fast)))
        ds[bscan_tag].description = 'B-scan Number'
        ds.add(DataElement(frame_tag, "IS", int(self.processParameters.res_slow)))
        ds[frame_tag].description = 'A-line Number'
        if self.processParameters.octaFlag:
            ds.add(DataElement(oct_type_tag, "LO", 'OCTA'))
        else:
            ds.add(DataElement(oct_type_tag, "LO", 'OCT'))
        ds[oct_type_tag].description = 'OCT Type'
        ds.add(DataElement(scanning_pattern_tag, "LO", 'raster'))
        ds[scanning_pattern_tag].description = 'Scanning Pattern'
        ds.add(DataElement(xrange_tag, "LO", str(self.processParameters.xrng)))
        ds[xrange_tag].description = 'X Range'
        ds.add(DataElement(yrange_tag, "LO", str(self.processParameters.yrng)))
        ds[yrange_tag].description = 'Y Range'


    def run(self):
        xpw = self.processParameters.xrng        
        zpw = self.processParameters.zrng/(2)
        bar_f = xpw/zpw
        average_number = self.average_number
        #average_number = 10
        directory = self.directory
        dcm_file_path = directory+'/OCT_Reconstruct_'+self.processParameters.fname.split('/')[1].split('.')[0]+'.dcm'
        ds = FileDataset(dcm_file_path, {}, file_meta=None, preamble=b"\0" * 128)

        if self.selected_option == "Save as dicom":
            save_mat = cp.asarray(self.frame_3D_resh)
            for i in range(int(self.processParameters.res_slow)):
                self.frame_3D_resh[:,:,i] = image = cp.asnumpy((cp.flipud((cp.clip((save_mat[:,:,i] - self.lowVal) / (self.highVal - self.lowVal), 0, 1)) * 255)).astype(cp.uint8))

            print("reach 1")
            self.dicom_add_custom_elements(ds)
            ds.PatientName = "David"
            ds.Sex = "male"
            input_date_str = "Aug_25_2022"
            ds.PatientID = "001"

            print("reach 2")
            input_date = datetime.strptime(input_date_str, "%b_%d_%Y")
            dicom_date_str = input_date.strftime("%Y%m%d")
            ds.StudyDate = dicom_date_str
            study_time_str = f"{self.processParameters.hr}:{self.processParameters.min}:{self.processParameters.sec}"
            ds.StudyTime = datetime.strptime(study_time_str, "%H:%M:%S").strftime("%H%M%S")
            ds.StudyInstanceUID = generate_uid()
            ds.SeriesInstanceUID = generate_uid()
            ds.SOPInstanceUID = generate_uid()

            print("reach 3")
            ds.Rows = self.frame_3D_resh.shape[1]
            ds.Columns = self.frame_3D_resh.shape[2]
            ds.NumberOfFrames = self.frame_3D_resh.shape[0]
            ds.BitsAllocated = 8
            print("reach 4")
            ds.PixelData = self.frame_3D_resh.astype(np.uint8).tobytes()
            print("reach 5")
            ds.save_as(dcm_file_path)
            del save_mat
            cp.get_default_memory_pool().free_all_blocks()
            return
        
        if self.selected_option == "Save as tiff stack":
            print("1")
            output_path = directory+'/OCT_Reconstruct_volume_tif_stack.tiff'
            #registration.shift_correction_bulk(save_mat,average_number)
            output_tiff = []
            for i in range(int(self.processParameters.res_slow)):
                print("2")
                image = np.squeeze(np.flipud(np.clip((self.frame_3D_resh[:,:,i]-self.lowVal)/(self.highVal-self.lowVal),0,1)*255).astype(np.uint8))
                #print(np.shape(image))
                output_tiff.append(Image.fromarray(image))
            
            output_tiff[0].save(output_path, save_all=True, append_images=output_tiff[1:], compression="tiff_deflate")
            return

        if self.selected_option != "Save speckle reduced images":
            save_mat = cp.asarray(self.frame_3D_resh)
            start_frame_num = self.processParameters.res_slow/(32*2)-1
            increment = self.processParameters.res_slow/32
            registration.shift_correction_bulk(save_mat,average_number)
            print("bar_f:",bar_f)
            
            if not os.path.exists(directory+"/"+str(average_number)+"_averaged"):
                os.makedirs(directory+"/"+str(average_number)+"_averaged")
            for i in range(32):
                averaged_img = cp.squeeze(cp.mean(save_mat[:,:,start_frame_num+i*increment-average_number/2:min(start_frame_num+i*increment+average_number/2,self.processParameters.res_slow-1)],axis=2))
                #plt.imshow(20*np.log10(averaged_img.get()), cmap = 'gray', origin = "lower", aspect = bar_f, vmin = self.lowVal, vmax = self.highVal)
                #plt.savefig(self.directory+'/OCT_Reconstruct_'+str(average_number)+'_averaged_'+str(i)+'.tiff')
                image = cp.asnumpy((cp.flipud((cp.clip((averaged_img - self.lowVal) / (self.highVal - self.lowVal), 0, 1)) * 255)).astype(cp.uint8))
                cv2_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                res = cv2.resize(cv2_image, dsize=(4*int(np.shape(image)[0]*bar_f), 4*np.shape(image)[0]))
                cv2.imwrite(directory+"/"+str(average_number)+'_averaged/OCT_Reconstruct_'+str(average_number)+'_averaged_'+str(i)+'.tiff', res)
                
        else:
            save_mat = cp.asarray(self.frame_3D_resh)
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i in range(16):
                averaged_img = cp.flipud(save_mat[:,:,i])
                image = cp.asnumpy(((cp.clip((averaged_img - self.lowVal) / (self.highVal - self.lowVal), 0, 1)) * 255).astype(cp.uint8))
                cv2_img = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                res = cv2.resize(cv2_img, dsize=(4*int(np.shape(image)[0]*bar_f), 4*np.shape(image)[0]))
                plt.imsave(directory+'/OCT_Reconstruct_'+str(average_number)+'_averaged_'+str(i)+'.tiff',res)
        del save_mat
        cp.get_default_memory_pool().free_all_blocks()