import cv2
import csv
import numpy as np
import pandas as pd
from scipy import io
from scipy.stats import norm, multivariate_normal


# 计算先验概率
def get_prior(arr_1, arr_2):
    prior_1 = len(arr_1) / (len(arr_2))
    prior_2 = 1 - prior_1

    return prior_1, prior_2


# 计算协方差矩阵
def get_mean_cov(color_1_pixels, color_2_pixels):
    color_1_mean = np.mean(color_1_pixels, axis=0)
    color_2_mean = np.mean(color_2_pixels, axis=0)
    color_1_cov = np.cov(color_1_pixels, rowvar=False)
    color_2_cov = np.cov(color_2_pixels, rowvar=False)

    return color_1_mean, color_2_mean, color_1_cov, color_2_cov


# 读取样本数据
def get_whole_sample(img_file, mat_file):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = []
    with open(mat_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 将每一行的数据转换为浮点数（或者根据你的需要选择适当的数据类型）
            row_data = [float(val) for val in row]
            data.append(row_data)
    # 将数据列表转换为NumPy数组
    mask_mat = np.array(data, dtype=np.uint8)
    fg_pixels = []
    bg_pixels = []
    x = len(mask_mat)
    y = np.shape(mask_mat)
    img_lst = img.tolist()

    height, width = img.shape[0], img.shape[1]

    for i in range(height):
        for j in range(width):
            if mask_mat[i, j] == 1:
                fg_pixels.append(img_lst[i][j])
            else:
                bg_pixels.append(img_lst[i][j])

    fg_pixels = np.array(fg_pixels)
    bg_pixels = np.array(bg_pixels)

    return fg_pixels, bg_pixels


# 生成mask(csv)
def generate_mask_csv(
    out_dir, img_dir, img_name, height_range, width_range, priors, means, covs, weight=1
):
    print("GENERATE_MASK_CSV: the mask of " +
          img_dir + "/" + img_name + " generating...")

    img = cv2.imread(img_dir + "/" + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_mask = np.zeros(img.shape[:2])

    fg_prob = priors[0] * multivariate_normal.pdf(img, means[0], covs[0])
    bg_prob = priors[1] * multivariate_normal.pdf(img, means[1], covs[1])

    img_mask[weight * fg_prob > bg_prob] = 1
    img_mask[: height_range[0], :] = 0
    img_mask[height_range[1]:, :] = 0
    img_mask[:, : width_range[0]] = 0
    img_mask[:, width_range[1]:] = 0

    kernel = np.ones([5, 5])
    img_mask = cv2.dilate(img_mask, kernel)
    img_mask = cv2.erode(img_mask, kernel)

    # 将掩码保存为CSV文件
    csv_file = "Mask_of_" + img_name + ".csv"
    with open(out_dir + "/" + csv_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(img_mask.astype(int))

    print(
        "GENERATE_MASK_CSV: the mask of "
        + img_dir
        + "/"
        + img_name
        + " saved as "
        + out_dir
        + "/"
        + csv_file
    )

    mask_path = out_dir + "/" + csv_file
    return mask_path


# 生成mask(mat)
def generate_mask_mat(
    out_dir, img_dir, img_name, height_range, width_range, priors, means, covs, weight=1
):
    print("GENERATE_MASK: the mask of " + img_dir +
          "/" + img_name + " generating...")

    img = cv2.imread(img_dir + "/" + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_mask = np.zeros(img.shape[:2])

    fg_prob = priors[0] * multivariate_normal.pdf(img, means[0], covs[0])
    bg_prob = priors[1] * multivariate_normal.pdf(img, means[1], covs[1])

    img_mask[weight * fg_prob > bg_prob] = 1
    img_mask[: height_range[0], :] = 0
    img_mask[height_range[1]:, :] = 0
    img_mask[:, : width_range[0]] = 0
    img_mask[:, width_range[1]:] = 0

    kernel = np.ones([5, 5])
    img_mask = cv2.dilate(img_mask, kernel)
    img_mask = cv2.erode(img_mask, kernel)

    mask_file = "Mask_of_" + img_name + ".csv"
    mask_name = "Mask"
    io.savemat(out_dir + "/" + mask_file, {mask_name: img_mask})

    print(
        "GENERATE_MASK: the mask of "
        + img_dir
        + "/"
        + img_name
        + " saved as "
        + out_dir
        + "/"
        + mask_file
    )
    mask_path = out_dir + "/" + mask_file
    return mask_path
