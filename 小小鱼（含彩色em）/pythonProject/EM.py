import numpy
from utils import config
import csv
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from utils.tools import pdf, calculateMRV, savepara

#加载文件
array = config.array_root
mask = config.mask_root
img = Image.open(config.tarin_image)
with open(array) as f:
    reader = csv.reader(f)
    x = list(reader)
    Array = np.array(x,dtype=numpy.float64)

with open(mask) as f:
    reader = csv.reader(f)
    x = list(reader)
    Mask = np.array(x,dtype=numpy.float64)
"""
print(Array)
print(Mask)
"""

#获取小小鱼的色彩与灰度值
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
fish_rgb = img_rgb * mask_rgb/255
fish_gary = img_gary * Mask/255
#print(np.shape(fish_rgb))
#print("fish_rgb",fish_rgb)

"""
#根据label将红色和白色的像素点分开
gray_r = []
gray_w = []
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

#先验概率
pri_pre_r = len(gray_r)/(len(gray_r)+len(gray_w))
pri_pre_w = 1-pri_pre_r
#print(pri_pre_w)
#print(pri_pre_r)
"""
"""
采用灰度

#用正态分布估计类条件概率密度函数的分布

gr_eql = np.mean(gray_r)
gw_eql = np.mean(gray_w)
gr_s = np.std(gray_r)
gw_s = np.std(gray_w)




#贝叶斯分类
gray_result = np.zeros_like(img_gary)
for i in range(len(fish_gary)):
    for j in range(len(fish_gary[0])):
        if(fish_gary[i][j]!=0):
           # print(fish_gary[i][j])
            pdf_r = pdf(fish_gary[i][j], gr_eql, gr_s)
            pdf_w = pdf(fish_gary[i][j], gw_eql, gw_s)
           # print("i",i,"j",j, pdf_r , pdf_w)
            gray_result[i][j] = fish_gary[i][j]
        if fish_gary[i][j] == 0:
            gray_result[i][j] = 0
        elif pri_pre_r*pdf_r > pri_pre_w*pdf_w:
            gray_result[i][j] = 255
       #     #print("hong")
        else:
            gray_result[i][j] = 100
            #print("bai")
#结果可视化
#像素分割与原图对比
plt.figure(1)
pic_gary = plt.subplot(1, 1, 1)
pic_gary.set_title('fish_gary')
pic_gary.imshow(fish_gary,cmap='gray')
plt.figure(2)
pic_garyresult = plt.subplot(1, 1, 1)
pic_garyresult.set_title('gray result')
pic_garyresult.imshow(gray_result, cmap='gray')
#函数图像化
plt.figure(3)
pdf_rsample = []
pdf_wsample = []
for i in range(1000):
    pdf_rsample.append(pdf(i/1000,gr_eql,gr_s))
    pdf_wsample.append(pdf(i/1000,gw_eql,gw_s))
pic_pdf = plt.subplot(1,1,1)
pic_pdf.set_title('p(x|w)')
x = np.arange(0, 1, 1/1000)
line1, = pic_pdf.plot(x,pdf_rsample, 'r',label="p(x|w) of red")
line2, = pic_pdf.plot(x,pdf_wsample, 'b',label="p(x|w) of white")
plt.legend(loc="best")
pic_pdf.set_xlabel('x')
pic_pdf.set_ylabel('f(x)')

#分子比较
plt.figure(4)
postpro_r = []
postpro_w = []
for i in range(1000):
    postpro_r.append(pri_pre_r*pdf_rsample[i])
    postpro_w.append(pri_pre_w*pdf_wsample[i])
pic_postpro = plt.subplot(1,1,1)
pic_postpro.set_title('postpro')
x = np.arange(0, 1, 1/1000)
line1, = pic_postpro.plot(x,postpro_r, 'r',label="postpro of red")
line2, = pic_postpro.plot(x,postpro_w, 'b',label="postpro of white")
plt.legend(loc="best")
pic_postpro.set_xlabel('x')
pic_postpro.set_ylabel('f(x)')
plt.show()
"""

#rgb通道
#均值与协方差矩阵
guess_pri_r = 0.7
guess_pri_w = 0.3
guess_rgbr_eql = np.array([0.6,0.35,0.15])
guess_wrgb_eql = np.array([0.7,0.75,0.75])
guess_cov_r = np.array([[0.60,0.25,0.05],[0.25,0.18,0.13],[0.26,0.13,0.23]])
guess_cov_w = np.array([[0.1,0.2,0.20],[0.15,0.25,0.36],[0.20,0.36,0.61]])
pri_pre_r = guess_pri_r
pri_pre_w = guess_pri_w
rgb_r_eql = guess_rgbr_eql
rgb_w_eql = guess_wrgb_eql
cov_r = guess_cov_r
cov_w = guess_cov_w
#3维比较
pic_rgb = plt.subplot(1, 1, 1)
plt.figure(0)
pic_rgb.set_title('RGB fish')
pic_rgb.imshow(fish_rgb)
rgb_result = np.zeros_like(fish_rgb)
for y in range(5):
    print("iter", y)
    rgb_r = []
    red = [0,0,0]
    white = [0,0,0]
    rgb_w = []
    countr = 0
    countw = 0
    for i in range(len(fish_rgb)):
        for j in range(len(fish_rgb[0])):
            if np.sum(fish_rgb[i][j]) == 0:
                continue
            elif pri_pre_r*multivariate_normal.pdf(fish_rgb[i][j], rgb_r_eql, cov_r) >pri_pre_w*multivariate_normal.pdf(fish_rgb[i][j], rgb_w_eql, cov_w):
            #print(fish_rgb[i][j])
                rgb_r.append(fish_rgb[i][j])
                red = red + fish_rgb[i][j]
                rgb_result[i][j] = [0.5, 0, 0.5]
                countr = countr + 1
            else:
                rgb_w.append((fish_rgb[i][j]))
                white = white+fish_rgb[i][j]
                rgb_result[i][j] = [0, 1, 0]
                countw = countw + 1
    rgb_r = np.array(rgb_r)
    rgb_w = np.array(rgb_w)
    pri_pre_r = countr/(countr+countw)
    pri_pre_w = 1 - pri_pre_r
    rgb_r_eql = red/countr
    rgb_w_eql = white/countw
    cov_r = np.zeros((3, 3))
    cov_w = np.zeros((3, 3))
    cov_r = calculateMRV(rgb_r, cov_r, rgb_r_eql)
    cov_w = calculateMRV(rgb_w, cov_w, rgb_w_eql)

    plt.figure(y+1)
    pic_rgbresult = plt.subplot(1, 1, 1)
    pic_rgbresult.set_title('RGB result')
    pic_rgbresult.imshow(rgb_result)
    plt.show()

#保存参数

with open(config.embayes_root, 'w') as f:
    writer = csv.writer(f)
    #writer.writerow(['0',gr_eql,gr_s,gw_eql,gw_s])
    writer.writerow(['1',pri_pre_r,pri_pre_w])
    writer.writerow(['2',rgb_r_eql[0],rgb_r_eql[1],rgb_r_eql[2],rgb_w_eql[0],rgb_w_eql[1],rgb_w_eql[2]])
    writer.writerow(['3',cov_r[0][0],cov_r[0][1],cov_r[0][2]])
    writer.writerow(['4', cov_r[1][0], cov_r[1][1], cov_r[1][2]])
    writer.writerow(['5', cov_r[2][0], cov_r[2][1], cov_r[2][2]])
    writer.writerow(['6', cov_w[0][0], cov_w[0][1], cov_w[0][2]])
    writer.writerow(['7', cov_w[1][0], cov_w[1][1], cov_w[1][2]])
    writer.writerow(['8', cov_w[2][0], cov_w[2][1], cov_w[2][2]])

