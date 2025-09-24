import os
import shutil

source_root = "Z:\\SC_Segmentation\\MLTrainingData\\0222 HD"  # Replace with the root directory path
destination_root = "Z:\\SC_auto_segmentation\\"  # Replace with your destination base path

for dirpath, dirnames, filenames in os.walk(source_root):

    if 'frame_struc.mat' in filenames:
        # Create a new directory with full path name as part of its name
        relative_path = os.path.relpath(dirpath, source_root)
        folder_name = relative_path.replace(os.sep, '_')
        new_dir_name = os.path.join(destination_root, folder_name)
        os.makedirs(new_dir_name, exist_ok=True)
        if os.path.isdir(new_dir_name):
            print("Directory exists")
            src_folder = os.path.join(dirpath, 'frame_struc.mat')
            #if os.path.isdir(src_folder):
            dst_folder = os.path.join(new_dir_name, 'frame_struc.mat')
                # if os.path.isdir(dst_folder):
                #     print('done')
                # else:
            shutil.copy(src_folder, dst_folder)
        else:
            print(0)
        