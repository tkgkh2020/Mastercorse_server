"""
前処理
フォルダ内の画像のサイズを揃えてndarrayにして保存する
"""

import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

files = glob.glob("/opt/pfw/dragon_ball_img/disc11/story4/*.jpg")

image_h_size = 256
image_w_size = 320
num_color = 3

for f in tqdm(files):

    img = Image.open(f)
    img = img.resize((image_w_size, image_h_size))
    arr = np.array(img)
    arr = (arr - 127.5) / 127.5
    arr.resize((image_h_size, image_w_size, num_color))

    filename = os.path.basename(f)
    ftitle, fext = os.path.splitext(filename)
    
    #print(ftitle)
    np.save('/opt/pfw/dragon_ball_img/all_ndarray/story4/' + ftitle + ".npy", arr)
