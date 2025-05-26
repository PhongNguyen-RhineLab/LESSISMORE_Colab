import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json # Import json để đọc file cấu hình

# Bạn có thể cần cài đặt lại xplique nếu chưa có
# !pip install xplique tensorflow_addons

# Cấu hình môi trường GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Kiểm tra GPU có sẵn
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Running on CPU (this will be very slow).")

# --- QUAN TRỌNG: Thiết lập thư mục gốc của dự án ---
project_root_dir = "/content/LESSISMORE_Colab"
os.chdir(project_root_dir)
print(f"Current working directory set to: {os.getcwd()}")

# Thiết lập thư mục đầu ra MỚI
output_image_directory = os.path.join(project_root_dir, "image", "test_result", "cub")
os.makedirs(output_image_directory, exist_ok=True)
print(f"Output directory ensured: {output_image_directory}")

# Import CubSubModularExplanationV2 từ models
from models.submodular_cub_v2 import CubSubModularExplanationV2

# Import các lớp từ xplique (chắc chắn đã cài đặt)
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)

import matplotlib
from sklearn import metrics
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry # Uncomment nếu bạn sử dụng SAM

# Cấu hình Matplotlib (bỏ qua font Times New Roman nếu gây lỗi)
# plt.rc('font', family="Times New Roman")
plt.style.use('seaborn-v0_8-darkgrid')

# --- Các hàm hỗ trợ để xử lý và lưu ảnh ---
def process_image_for_display_or_save(img):
    """
    Chuẩn bị ảnh cho việc hiển thị hoặc lưu (chuyển sang RGB nếu cần).
    Input:
        img: numpy array của ảnh (BGR hoặc Grayscale).
    Output:
        img_processed: numpy array của ảnh (RGB).
    """
    if len(img.shape) == 3:
        img_processed = img[:, :, ::-1]  # BGR sang RGB
    else:
        img_processed = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_processed

def save_image_to_file(img_array, filename_base, output_dir):
    """
    Lưu ảnh vào file.
    Input:
        img_array: numpy array của ảnh (dạng RGB).
        filename_base: Tên file cơ sở (ví dụ: "my_image.png").
        output_dir: Thư mục để lưu ảnh.
    """
    full_path = os.path.join(output_dir, filename_base)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(img_array)
    plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Image saved to: {full_path}")

# --- Hàm SubRegionDivision (Được định nghĩa lại nếu không import từ utils) ---
# Nếu hàm này đã được import từ utils.py, bạn có thể xóa định nghĩa này ở đây
def SubRegionDivision(image, mode="slico"):
    element_sets_V = []
    if mode == "slico":
        # Check if ximgproc is available, it might not be in default Colab
        try:
            slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=30, ruler = 20.0)
            slic.iterate(20)
            label_slic = slic.getLabels()
            number_slic = slic.getNumberOfSuperpixels()
            for i in range(number_slic):
                img_copp = image.copy()
                img_copp = img_copp * (label_slic == i)[:,:, np.newaxis]
                element_sets_V.append(img_copp)
        except AttributeError:
            print("cv2.ximgproc not found. Please ensure opencv-contrib-python is installed.")
            print("Falling back to a simpler segmentation if possible or raise error.")
            # Fallback or error handling for missing ximgproc
            # For demonstration, let's just return empty if ximgproc is missing
            return []
    elif mode == "seeds":
        try:
            seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
            seeds.iterate(image,10)
            label_seeds = seeds.getLabels()
            number_seeds = seeds.getNumberOfSuperpixels()
            for i in range(number_seeds):
                img_copp = image.copy()
                img_copp = img_copp * (label_seeds == i)[:,:, np.newaxis]
                element_sets_V.append(img_copp)
        except AttributeError:
            print("cv2.ximgproc not found. Please ensure opencv-contrib-python is installed.")
            print("Falling back to a simpler segmentation if possible or raise error.")
            return []
    return element_sets_V

# Khởi tạo mô hình CubSubModularExplanationV2
# Đảm bảo đường dẫn tới cfg_path là TUYỆT ĐỐI
smdl = CubSubModularExplanationV2(
    cfg_path="/content/LESSISMORE_Colab/configs/cub/submodular_cfg_cub_tf-resnet-v2.json",
    k=50,
    lambda1=1,
    lambda2=1,
    lambda3=1,
    lambda4=1)

# Tải và xử lý ảnh đầu vào
image_path = os.path.join(project_root_dir, "examples", "Crested_Auklet_0059_794929.jpg")
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
image = cv2.resize(image, (224, 224))

# --- Hàm visualization đã chỉnh sửa để lưu file ---
def visualization(image_orig, submodular_image_set, saved_json_file, algorithm_mode, output_dir=output_image_directory):
    """
    Visualize results and save to files.
    """
    insertion_ours_images = []
    # deletion_ours_images = [] # Không dùng

    insertion_image = submodular_image_set[0]
    insertion_ours_images.append(insertion_image)
    # deletion_ours_images.append(image_orig - insertion_image) # Không dùng
    for smdl_sub_mask in submodular_image_set[1:]:
        insertion_image = insertion_image.copy() + smdl_sub_mask
        insertion_ours_images.append(insertion_image)
        # deletion_ours_images.append(image_orig - insertion_image) # Không dùng

    insertion_ours_images_input_results = np.array(saved_json_file["consistency_score"])

    ours_best_index = np.argmax(insertion_ours_images_input_results)
    x = [(insertion_ours_image.sum(-1) != 0).sum() / (image_orig.shape[0] * image_orig.shape[1]) for insertion_ours_image in insertion_ours_images]
    i = len(x)

    # --- Lưu ảnh hiển thị (ax2) ---
    fig_img, ax_img = plt.subplots(figsize=(10, 10)) # Tạo figure mới cho ảnh
    ax_img.spines["left"].set_visible(False)
    ax_img.spines["right"].set_visible(False)
    ax_img.spines["top"].set_visible(False)
    ax_img.spines["bottom"].set_visible(False)
    ax_img.xaxis.set_visible(False)
    ax_img.yaxis.set_visible(False)
    ax_img.set_title(f'Ours ({algorithm_mode})', fontsize=24)
    ax_img.set_facecolor('white')

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = (image_orig - insertion_ours_images[ours_best_index]).mean(-1)
    mask[mask > 0] = 1

    dilate = cv2.dilate(mask, kernel, 3)
    edge = dilate - mask
    image_debug = image_orig.copy()
    image_debug[mask > 0] = image_debug[mask > 0] * 0.5
    image_debug[edge > 0] = np.array([0, 0, 255])

    img_to_save = process_image_for_display_or_save(image_debug)
    save_image_to_file(img_to_save, f"{algorithm_mode}_image_best_index.png", output_dir)
    plt.close(fig_img) # Đóng figure ảnh để giải phóng bộ nhớ


    # --- Tạo và lưu biểu đồ đường (ax3) ---
    fig_plot, ax_plot = plt.subplots(figsize=(12, 8)) # Tạo figure mới cho biểu đồ đường
    ax_plot.set_xlim((0, 1))
    ax_plot.set_ylim((0, 1))
    ax_plot.tick_params(axis='x', labelsize=20)
    ax_plot.tick_params(axis='y', labelsize=20)
    ax_plot.set_title(f'Insertion ({algorithm_mode})', fontsize=24)
    ax_plot.set_ylabel('Recognition Score', fontsize=22)
    ax_plot.set_xlabel('Percentage of image revealed', fontsize=22)

    x_ = x[:i]
    ours_y = insertion_ours_images_input_results[:i]
    ax_plot.plot(x_, ours_y, color='dodgerblue', linewidth=3.5)

    ax_plot.scatter(x_[-1], ours_y[-1], color='dodgerblue', s=54)
    ax_plot.plot([x_[ours_best_index], x_[ours_best_index]], [0, 1], color='red', linewidth=3.5)

    save_image_to_file(fig_plot, f"{algorithm_mode}_insertion_plot.png", output_dir)
    plt.close(fig_plot) # Đóng figure biểu đồ đường


    auc = metrics.auc(x, insertion_ours_images_input_results)
    print(f"For {algorithm_mode} mode:")
    print(f"  Highest confidence: {insertion_ours_images_input_results.max():.4f}")
    print(f"  Final confidence: {insertion_ours_images_input_results[-1]:.4f}")
    print(f"  Insertion AUC: {auc:.4f}")
    print("-" * 30)


# --- Chạy quá trình Submodular Explanation ---
# Chạy với chế độ SLIC
element_sets_V_slic = SubRegionDivision(image, mode="slico")
if element_sets_V_slic: # Chỉ chạy nếu Superpixel Division thành công
    smdl.k = len(element_sets_V_slic)
    submodular_image_slic, submodular_image_set_slic, saved_json_file_slic = smdl(element_sets_V_slic)
    visualization(image, submodular_image_set_slic, saved_json_file_slic, algorithm_mode="slico")

# Chạy với chế độ SEEDS
element_sets_V_seeds = SubRegionDivision(image, mode="seeds")
if element_sets_V_seeds: # Chỉ chạy nếu Superpixel Division thành công
    smdl.k = len(element_sets_V_seeds)
    submodular_image_seeds, submodular_image_set_seeds, saved_json_file_seeds = smdl(element_sets_V_seeds)
    visualization(image, submodular_image_set_seeds, saved_json_file_seeds, algorithm_mode="seeds")


print(f"\nAll visualization outputs saved as PNG files in: {output_image_directory}")
print(f"Check the 'Files' section (folder icon on the left) to download them.")