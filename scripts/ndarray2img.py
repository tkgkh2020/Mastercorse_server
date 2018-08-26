"""
ndarrayを画像にして返す関数
(batch_size, image_h, image_w, color)の４階テンソルが引数
"""

import numpy as np
import random
import glob
from PIL import Image


def combine_images(generated_images, cols=3, rows=3):
    """
    9枚の画像を3*3に敷き詰めて一枚の画像にする
    """
    shape = generated_images.shape
    h = shape[1]
    w = shape[2]
    c = shape[3]
    image = np.zeros((rows * h,  cols * w, c))
    for index, img in enumerate(generated_images):
        if index >= cols * rows:
            break
        i = index // cols
        j = index % cols
        image[i*h:(i+1)*h, j*w:(j+1)*w, :] = img[:, :, :]
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8))
    return image


if __name__ == "__main__":
    batch_size = 9
    ndarray_file_list = glob.glob("/opt/pfw/dragon_ball_img/all_ndarray/disc11/story1/*.npy", recursive=True)
    files = random.sample(ndarray_file_list, batch_size) # pathのリストからランダムにチョイス
    batch_images = np.array([np.load(path) for path in files]) # ファイル読み込み
    print(batch_images.shape)
    image = combine_images(batch_images)
    image.save("./test.jpg")
    print("image saved.")
