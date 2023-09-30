import cv2
import csv
import numpy as np
import pandas as pd
from scipy import io
from scipy.stats import norm, multivariate_normal


# task_1-1
def get_fish_sample(mat_file, mat_name):
    mat = pd.reader(mat_file)
    gray_arr = mat[mat_name][:, 0]
    rgb_arr = mat[mat_name][:, 1:4]
    label_arr = mat[mat_name][:, -1]

    return gray_arr, rgb_arr, label_arr


# 获取两种颜色的像素
def get_two_color_pixels(pixel_arr, label_arr):
    pixel_lst = pixel_arr.tolist()
    label_lst = label_arr.tolist()

    color_1_pixels = []
    color_2_pixels = []
    for i in range(len(pixel_lst)):
        if label_lst[i] == -1:
            color_1_pixels.append(pixel_lst[i])
        else:
            color_2_pixels.append(pixel_lst[i])
    color_1_pixels = np.array(color_1_pixels)
    color_2_pixels = np.array(color_2_pixels)

    return color_1_pixels, color_2_pixels


# 计算先验概率
def get_prior(arr_1, arr_2):
    prior_1 = len(arr_1) / (len(arr_2))
    prior_2 = 1 - prior_1

    return prior_1, prior_2


# 计算标准差
def get_mean_std(white_pixels, red_pixels):
    white_mean = np.mean(white_pixels)
    red_mean = np.mean(red_pixels)
    white_std = np.std(white_pixels)
    red_std = np.std(red_pixels)

    return white_mean, red_mean, white_std, red_std


# 计算协方差矩阵
def get_mean_cov(color_1_pixels, color_2_pixels):
    color_1_mean = np.mean(color_1_pixels, axis=0)
    color_2_mean = np.mean(color_2_pixels, axis=0)
    color_1_cov = np.cov(color_1_pixels, rowvar=False)
    color_2_cov = np.cov(color_2_pixels, rowvar=False)

    return color_1_mean, color_2_mean, color_1_cov, color_2_cov


# 获取图像对应的mask
def get_mask(mat_file, mat_name):
    mask_mat = io.loadmat(mat_file)[mat_name]
    return mask_mat


# 获取灰度图像的mask结果
def get_img_gray_masked(img, mask):
    img = cv2.imread(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_masked = img_gray * mask / 255

    return img_gray_masked


# 获取RGB图像的mask结果
def get_img_rgb_masked(img_name, mask):
    img = cv2.imread(img_name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.array([mask, mask, mask]).transpose(1, 2, 0)
    img_rgb_masked = img_rgb * mask / 255

    return img_rgb_masked


# 灰度图像分割
def segment_img_gray(img, mask, priors, means, stds, weight=1):
    img_gray_masked = get_img_gray_masked(img, mask)
    rtn_img_gray = np.zeros_like(img_gray_masked)

    val_1 = priors[0] * norm.pdf(img_gray_masked, means[0], stds[0])
    val_2 = priors[1] * norm.pdf(img_gray_masked, means[1], stds[1])

    mask_idx = np.bool8(mask)
    color_1_idx = weight * val_1 > val_2

    rtn_img_gray[mask_idx] = 100
    rtn_img_gray[color_1_idx] = 255

    return rtn_img_gray


# RGB图像分割
def segment_img_rgb(img, mask, priors, means, covs, weight=1):
    img_rgb_masked = get_img_rgb_masked(img, mask)
    rtn_img_rgb = np.zeros_like(img_rgb_masked)

    val_1 = priors[0] * \
        multivariate_normal.pdf(img_rgb_masked, means[0], covs[0])
    val_2 = priors[1] * \
        multivariate_normal.pdf(img_rgb_masked, means[1], covs[1])

    mask_idx = np.bool8(mask)
    color_1_idx = weight * val_1 > val_2

    rtn_img_rgb[mask_idx] = [255, 0, 0]
    rtn_img_rgb[color_1_idx] = [255, 255, 255]

    return rtn_img_rgb


# 保存灰度图像
def img_gray_save(save_dir, prefix, img_name, img_arr):
    img_to_save = save_dir + "/" + prefix + "_" + img_name
    cv2.imwrite(img_to_save, img_arr)


# 保存RGB图像
def img_rgb_save(save_dir, prefix, img_name, img_arr):
    img_to_save = save_dir + "/" + prefix + "_" + img_name
    img_arr = cv2.cvtColor(img_arr.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_to_save, img_arr)


# task_1-2
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


# 读取RGB图像的像素值
def get_fg_rbg_pixels(pixel_arr, label_arr):
    pixel_lst = pixel_arr.tolist()
    label_lst = label_arr.tolist()

    white_pixels = []
    red_pixels = []
    for i in range(len(pixel_lst)):
        if label_lst[i] == -1:
            white_pixels.append(pixel_lst[i])
        else:
            red_pixels.append(pixel_lst[i])
    white_pixels = np.array(white_pixels)
    red_pixels = np.array(red_pixels)

    return white_pixels, red_pixels


# 生成mask
def generate_mask(
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
