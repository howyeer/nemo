import numpy
from utils import config
import csv
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from utils.tools import pdf, calculateMRV, savepara
from utils.mask import *


img_name = '311.bmp'
# img_name = ["311.bmp", "313.bmp", "315.bmp", "317.bmp"] # img_dir + "/" + img_sample
img_dir = 'E:/aaaacode/nemonemo/pythonProject/dataset'
img = Image.open(config.test311_image)  # 需要改成测试的图像文件名
bayes = config.bayes_root

# 获取mask
mask = get_mask(img_name, img_dir, sample_path=config.array_path,
                sample_img=config.tarin_image, mask_path=config.mask_path)
Mask = np.array(mask, dtype=np.float64)

rgb_reql = []
rgb_weql = []
rgb_rs = np.zeros((3, 3))
rgb_ws = np.zeros((3, 3))
with open(bayes) as f:
    reader = csv.reader(f)
    para = list(reader)
    print(para)
    gr_eql = np.array(para[0][1], dtype=numpy.float64)
    gr_s = np.array(para[0][2], dtype=numpy.float64)
    gw_eql = np.array(para[0][3], dtype=numpy.float64)
    gw_s = np.array(para[0][4], dtype=numpy.float64)
    pripro_r = np.array(para[2][1], dtype=numpy.float64)
    pripro_w = np.array(para[2][2], dtype=numpy.float64)
    for i in range(3):
        rgb_reql.append(para[4][i+1])
        rgb_weql.append(para[4][i+4])
    for i in range(3):
        for j in range(3):
            rgb_rs[i][j] = para[2*i+6][j+1]
            rgb_ws[i][j] = para[2*i+12][j+1]
    rgb_reql = np.array(rgb_reql)
    rgb_weql = np.array(rgb_weql)
    # para = np.array(para,dtype=numpy.float64)
img_rgb = np.array(img)
img_gary = np.array(img.convert('L'))
mask_rgb = np.array([Mask, Mask, Mask]).transpose(1, 2, 0)
fish_rgb = img_rgb * mask_rgb/255
fish_gary = img_gary * Mask/255
print(np.shape(rgb_weql))
print(rgb_ws)
print(rgb_rs)
# 灰度分类
gray_result = np.zeros_like(img_gary)
"""
for i in range(len(fish_gary)):
    for j in range(len(fish_gary[0])):
        if fish_gary[i][j] == 0:
            gray_result[i][j] = 0
        elif fish_gary[i][j]>para[0][1]:
            gray_result[i][j] = 255
        else:
            gray_result[i][j] = 100
"""
for i in range(len(fish_gary)):
    for j in range(len(fish_gary[0])):
        if fish_gary[i][j] == 0:
            gray_result[i][j] = 0
        elif pripro_r*pdf(fish_gary[i][j], gr_eql, gr_s) > pripro_r*pdf(fish_gary[i][j], gw_eql, gw_s):
            gray_result[i][j] = 255
        else:
            gray_result[i][j] = 100

pic_gary = plt.subplot(1, 2, 1)
pic_gary.set_title('fish_gary')
pic_gary.imshow(fish_gary, cmap='gray')
pic_gary = plt.subplot(1, 2, 2)
pic_gary.set_title('gray result')
pic_gary.imshow(gray_result, cmap='gray')
plt.show()
# rgb
rgb_result = np.zeros_like(fish_rgb)
for i in range(len(fish_rgb)):
    for j in range(len(fish_rgb[0])):
        if np.sum(fish_rgb[i][j]) == 0:
            continue
        elif pripro_r*multivariate_normal.pdf(fish_rgb[i][j], rgb_reql, rgb_rs) > pripro_w*multivariate_normal.pdf(fish_rgb[i][j], rgb_weql, rgb_ws):
            rgb_result[i][j] = [0.5, 0, 0.5]
        else:
            rgb_result[i][j] = [0, 1, 0]

pic_rgb = plt.subplot(1, 2, 1)
pic_rgb.set_title('RGB fish')
pic_rgb.imshow(fish_rgb)
pic_rgb = plt.subplot(1, 2, 2)
pic_rgb.set_title('RGB result')
pic_rgb.imshow(rgb_result)
plt.show()
