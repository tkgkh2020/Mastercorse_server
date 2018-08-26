"""
前処理
ディレクトリ内の画像を中心でクロップしてndarrayにして保存する
multiprocessingを使うためプログレスバーの表示をしていない
"""

import os
import glob
from PIL import Image
import numpy as np
from multiprocessing import Pool


files = glob.glob("/opt/pfw/dragon_ball_img/disc06/story1/*jpg")
crop_size = 128 # 目的サイズ
num_color = 3


def crop(img, width, height):
    """
    画像の中心から目的サイズの正方形画像を切り取る
    """
    img_w, img_h = img.size
    return img.crop(((img_w - width) // 2,
                     (img_h - height) // 2,
                     (img_w + width) // 2,
                     (img_h + height) // 2))


def img2ndarray(path):
    """
    リサイズしてndarrayに変換する
    """
    img = Image.open(path)
    img = img.resize((160, 128))
    img = crop(img, crop_size, crop_size)
    arr = np.array(img)
    arr = (arr - 127.5) / 127.5
    arr.resize((crop_size, crop_size, num_color))
    file_name = os.path.basename(path)
    ftitle, fext = os.path.splitext(file_name)
    #img.save('./img/' + ftitle + '_resized.jpg')
    #np.save('/opt/pfw/dragon_ball_img/all_ndarray/story5_128/' + ftitle + '.npy', arr)
    np.save('/opt/pfw/dragon_ball_img/all_ndarray/disc06/story1/' + ftitle + '.npy', arr)
    #print(ftitle)


def main():
    pool = Pool()
    result = pool.map(img2ndarray, files)
    pool.close()


if __name__ == "__main__":
    main()
