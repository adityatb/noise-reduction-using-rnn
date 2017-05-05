#Create Test Database by sampling n files and moving to Test file directory

import os,random
import shutil

move_no_files = 1000

work_path = os.getcwd()+"/Training/NoiseAdded/"
test_files_path = os.getcwd()+"/Wavs/"

src_files = (os.listdir(work_path))

def valid_path(dir_path, filename):
    full_path = os.path.join(dir_path, filename)
    return os.path.isfile(full_path)  

files = [os.path.join(work_path, f) for f in src_files if valid_path(work_path, f)]
choices = random.sample(files, move_no_files)
for files in choices:
    shutil.move(files, test_files_path)
    print "Moved: " + str(files)

print ('Finished!')