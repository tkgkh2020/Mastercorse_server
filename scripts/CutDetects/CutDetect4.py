"""
現在と２個前のフレームとの比較を行いシーン転換が起きているか判定するスクリプト
比較手法はヒストグラム法とピクセルの絶対値誤差の二種類
ヒストグラムはフェードアウトに弱く，ピクセルはカメラ移動に弱い
/opt/pfw/dragon_ball_img/disc*/以下に各storyのシーン転換点を記述したcsvファイルを作成する
"""
import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import csv


THRESH_MAE = 70  # MAEを使う際に．決め打ち
THRESH_HIST = 0.25  # histgramを使う際に．決め打ち．

DiscName = "disc05"

ESC_KEY = 27     # Escキー
INTERVAL = 1      # 待ち


def MAE(pic):  # mean absolute error
    return np.mean(np.abs(pic))


def resize(pic, picsize):
    resized = cv2.resize(pic, picsize, interpolation=cv2.INTER_LINEAR)
    return resized


def main():
    width = 160
    height = 128
    picsize = (width, height)
    l = "/opt/pfw/dragon_ball_img/" + DiscName + "/story*"
    stories = glob(l)

    for s in stories:
        
        files = sorted(glob(s + "/*jpg"))
        f = open(s+"_scene.csv", 'a')
        writer = csv.writer(f)

        frame_cnt = 0
        frame_1 = np.zeros((*picsize[::-1], 3))  # create empty image 1
        frame_2 = np.zeros((*picsize[::-1], 3))
        frame_3 = np.zeros((*picsize[::-1], 3))  # 3

        for frame in tqdm(files):
            frame = cv2.imread(frame)
            frame_3 = frame_2  # 2 to 3
            frame_2 = frame_1  # 1 to 2
            frame_1 = resize(frame, picsize)  # make 1

            hist_1 = cv2.calcHist([frame_1], [0], None, [256], [0, 256])
            if frame_cnt <= 1:
                hist_2 = hist_1
                hist_3 = hist_2
            else:
                hist_2 = cv2.calcHist([frame_2], [0], None, [256], [0, 256])
                hist_3 = cv2.calcHist([frame_3], [0], None, [256], [0, 256])

            key = cv2.waitKey(1)  # quit when esc-key pressed
            if key == ESC_KEY:
                break

            ret = cv2.compareHist(hist_1, hist_3, 0)  # 近い画像ほど1に近い値
            
            if ret <= THRESH_HIST:
                # print("{}".format(frame_cnt))
                a = []
                a.append(frame_cnt)
                writer.writerow(a)

            frame_cnt += 1
          
        f.close()


if __name__ == "__main__":
    main()
