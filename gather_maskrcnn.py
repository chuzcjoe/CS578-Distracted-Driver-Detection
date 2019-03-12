from os import listdir
from os.path import isfile, join, isdir
from shutil import copyfile, copy2

BASE_DIR = "test_out"
OUT_DIR = "mask_rcnn_full"

for f in listdir(BASE_DIR):
    CURR_PATH = join(BASE_DIR, f)
    for img_f in listdir(CURR_PATH):
        FILE_PATH = join(BASE_DIR, f, img_f, "full.jpg")
        OUT_PATH = join(OUT_DIR, f, img_f + ".jpg")
        #print(OUT_PATH)
        copyfile(FILE_PATH, OUT_PATH)
        #print(FILE_PATH)
