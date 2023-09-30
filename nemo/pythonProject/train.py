import numpy
from utils import config
import csv
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
#from utils.get_mask import get_mask
from utils.tools import pdf, calculateMRV, savepara, histogram, parzen

# -----------------加载309样本-----------------
array = config.array_root
img = Image.open(config.tarin_image)
with open(array) as f:
    reader = csv.reader(f)
    x = list(reader)
    Array = np.array(x, dtype=numpy.float64)

#-----------------加载mask-----------------
mask = config.mask_root
with open(mask) as f:
    reader = csv.reader(f)
    x = list(reader)
    Mask = np.array(x, dtype=numpy.float64)
"""
print(Array)
print(Mask)
"""

#-----------------获取小小鱼的色彩与灰度值-----------------
img_rgb = np.array(img)
img_gary = np.array(img.convert('L'))
mask_rgb = np.array([Mask, Mask, Mask]).transpose(1, 2, 0)
"""
print("rgb",np.shape(img_rgb))
print("Mask",np.shape(Mask))
print("gary",np.shape(img_gary))
print(img_rgb)
print(img_gary)
"""
fish_rgb = img_rgb * mask_rgb/255     #csv文件给的是归一化后的数据
fish_gary = img_gary * Mask/255       #所以我们也在此将数据归一化
# print(np.shape(fish_rgb))
# print("fish_rgb",fish_rgb)


#-----------------根据label将红色和白色的像素点分开-----------------
gray_r = []             #r代表类红色的像素点
gray_w = []             #w代表类白色的像素点
rgb_r = []
rgb_w = []
for i in range(len(Array)):
    if (Array[i][4]) == 1.:
        gray_r.append(Array[i][0])
        rgb_r.append(Array[i][1:4])
    else:
        gray_w.append(Array[i][0])
        rgb_w.append(Array[i][1:4])
rgb_r = np.array(rgb_r)
rgb_w = np.array(rgb_w)

#-----------------计算先验概率-----------------
pri_pre_r = len(gray_r)/(len(gray_r)+len(gray_w))
pri_pre_w = 1-pri_pre_r
# print(pri_pre_w)
# print(pri_pre_r)

"""
-----------------计算灰度值的参数-----------------
"""
#-----------------用正态分布估计类条件概率密度函数的分布-----------------
gr_eql = np.mean(gray_r)
gw_eql = np.mean(gray_w)
gr_s = np.std(gray_r)
gw_s = np.std(gray_w)


#-----------------贝叶斯分类-----------------
# gray_result = np.zeros_like(img_gary)
# x, y = np.shape(fish_gary)
# for i in range(x):
#     for j in range(y):
#         if (fish_gary[i][j] != 0):
#            # print(fish_gary[i][j])
#             pdf_r = pdf(fish_gary[i][j], gr_eql, gr_s)
#             pdf_w = pdf(fish_gary[i][j], gw_eql, gw_s)
#            # print("i",i,"j",j, pdf_r , pdf_w)
#             gray_result[i][j] = fish_gary[i][j]
#         if fish_gary[i][j] == 0:
#             gray_result[i][j] = 0
#         elif pri_pre_r*pdf_r > pri_pre_w*pdf_w:
#             gray_result[i][j] = 100
#        #     #print("hong")
#         else:
#             gray_result[i][j] = 255
#             # print("bai")

#-----------------无参数估计贝叶斯分类-----------------
gray_result = np.zeros_like(img_gary)
x, y = np.shape(fish_gary)
for i in range(x):
    for j in range(y):
        if (fish_gary[i][j] != 0):
           # print(fish_gary[i][j])
            pdf_r = parzen(fish_gary[i][j], gray_r, gr_s)
            pdf_w = parzen(fish_gary[i][j], gray_w, gw_s)
           # print("i",i,"j",j, pdf_r , pdf_w)
            gray_result[i][j] = fish_gary[i][j]
        if fish_gary[i][j] == 0:
            gray_result[i][j] = 0
        elif pri_pre_r*pdf_r > pri_pre_w*pdf_w:
            gray_result[i][j] = 100
       #     #print("hong")
        else:
            gray_result[i][j] = 255
            # print("bai")
#-----------------结果可视化-----------------
# 像素分割与原图对比
plt.figure(1)
pic_gary = plt.subplot(1, 1, 1)
pic_gary.set_title('fish_gary')
pic_gary.imshow(fish_gary, cmap='gray')
plt.figure(2)
pic_garyresult = plt.subplot(1, 1, 1)
pic_garyresult.set_title('gray result')
pic_garyresult.imshow(gray_result, cmap='gray')
#类条件概率密度函数图像
plt.figure(3)
pdf_rsample = []
pdf_wsample = []
# for i in range(1000):
#     pdf_rsample.append(pdf(i/1000, gr_eql, gr_s))
#     pdf_wsample.append(pdf(i/1000, gw_eql, gw_s))
# for i in range(1000):
#     pdf_rsample.append(histogram(i/1000, gray_r))
#     pdf_wsample.append(histogram(i/1000, gray_w))
for i in range(1000):
    pdf_rsample.append(parzen(i/1000, gray_r, gr_s))
    pdf_wsample.append(parzen(i/1000, gray_w, gw_s))
pic_pdf = plt.subplot(1, 1, 1)
pic_pdf.set_title('p(x|w)')
x = np.arange(0, 1, 1/1000)
line1, = pic_pdf.plot(x, pdf_rsample, 'r', label="p(x|w) of red")
line2, = pic_pdf.plot(x, pdf_wsample, 'b', label="p(x|w) of white")
plt.legend(loc="best")
pic_pdf.set_xlabel('x')
pic_pdf.set_ylabel('f(x)')

# 分子比较
plt.figure(4)
postpro_r = []
postpro_w = []
for i in range(1000):
    postpro_r.append(pri_pre_r*pdf_rsample[i])
    postpro_w.append(pri_pre_w*pdf_wsample[i])
pic_postpro = plt.subplot(1, 1, 1)
pic_postpro.set_title('postpro')
x = np.arange(0, 1, 1/1000)
line1, = pic_postpro.plot(x, postpro_r, 'r', label="postpro of red")
line2, = pic_postpro.plot(x, postpro_w, 'b', label="postpro of white")
plt.legend(loc="best")
pic_postpro.set_xlabel('x')
pic_postpro.set_ylabel('f(x)')
plt.show()


# rgb通道
# 均值与协方差矩阵
rgb_r_eql = np.mean(rgb_r, axis=0)
rgb_w_eql = np.mean(rgb_w, axis=0)
cov_r = np.zeros((3, 3))
cov_w = np.zeros((3, 3))
cov_r = calculateMRV(rgb_r, cov_r, rgb_r_eql)
cov_w = calculateMRV(rgb_w, cov_w, rgb_w_eql)
# 3维比较
rgb_result = np.zeros_like(fish_rgb)
for i in range(x):
    for j in range(y):
        if np.sum(fish_rgb[i][j]) == 0:
            continue
        elif pri_pre_r*multivariate_normal.pdf(fish_rgb[i][j], rgb_r_eql, cov_r) > pri_pre_w*multivariate_normal.pdf(fish_rgb[i][j], rgb_w_eql, cov_w):
            # print(fish_rgb[i][j])
            rgb_result[i][j] = [0.5, 0, 0.5]
        else:
            rgb_result[i][j] = [0, 1, 0]
plt.figure(3)
pic_rgb = plt.subplot(1, 1, 1)
pic_rgb.set_title('RGB fish')
pic_rgb.imshow(fish_rgb)
plt.figure(4)
pic_rgbresult = plt.subplot(1, 1, 1)
pic_rgbresult.set_title('RGB result')
pic_rgbresult.imshow(rgb_result)
plt.show()

# 保存参数

with open(config.bayes_root, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['0', gr_eql, gr_s, gw_eql, gw_s])
    writer.writerow(['1', pri_pre_r, pri_pre_w])
    writer.writerow(['2', rgb_r_eql[0], rgb_r_eql[1],
                    rgb_r_eql[2], rgb_w_eql[0], rgb_w_eql[1], rgb_w_eql[2]])
    writer.writerow(['3', cov_r[0][0], cov_r[0][1], cov_r[0][2]])
    writer.writerow(['4', cov_r[1][0], cov_r[1][1], cov_r[1][2]])
    writer.writerow(['5', cov_r[2][0], cov_r[2][1], cov_r[2][2]])
    writer.writerow(['6', cov_w[0][0], cov_w[0][1], cov_w[0][2]])
    writer.writerow(['7', cov_w[1][0], cov_w[1][1], cov_w[1][2]])
    writer.writerow(['8', cov_w[2][0], cov_w[2][1], cov_w[2][2]])
