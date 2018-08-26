"""
シーン分割を行うスクリプト
先にCutDetect.pyで分割フレーム点を記載したcsvファイルを作成する必要あり
"""
import pandas as pd
from glob import glob
import os
import shutil
from tqdm import tqdm


DiscName = "disc02"
 
csv_files = "/opt/pfw/dragon_ball_img/" + DiscName  + "/story*_scene.csv"
csv_list = glob(csv_files)

for c in tqdm(csv_list):
    scene_list = pd.read_csv(c)
    jpg_files = os.path.dirname(c) + "/" + os.path.basename(c).split("_")[0] + "/*jpg"
    jpg_list = sorted(glob(jpg_files))

    p_row = 0

    for index, row in scene_list.iterrows():
        imgDir = os.path.dirname(jpg_files) + "/scene%04d/" % index

        if os.path.exists(imgDir):
            shutil.rmtree(imgDir)

        if not os.path.exists(imgDir):
            os.makedirs(imgDir)

        for f in jpg_list[p_row:row[0]]:
            shutil.move(f, imgDir + ".")

        p_row = row[0]

