import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json  # Import json để đọc file cấu hình

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

# Thiết lập thư mục đầu ra MỚI cho CUB
output_cub_directory = os.path.join(project_root_dir, "image", "test_result", "cub")
os.makedirs(output_cub_directory, exist_ok=True)
print(f"Output directory for CUB ensured: {output_cub_directory}")

# Thiết lập thư mục đầu ra MỚI cho SAM
output_sam_directory = os.path.join(project_root_dir, "image", "test_result", "sam")
os.makedirs(output_sam_directory, exist_ok=True)
print(f"Output directory for SAM ensured: {output_sam_directory}")

# Import CubSubModularExplanationV2 từ models
from models.submodular_cub_v2 import CubSubModularExplanationV2

# Import các lớp từ xplique (chắc chắn đã cài đặt)
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)

import matplotlib
from sklearn import metrics
# Import SAM components
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

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


def save_image_to_file(img_data, filename_base, output_dir):
    """
    Lưu ảnh hoặc biểu đồ vào file.
    Input:
        img_data: numpy array của ảnh (dạng RGB) HOẶC matplotlib.figure object.
        filename_base: Tên file cơ sở (ví dụ: "my_image.png").
        output_dir: Thư mục để lưu ảnh.
    """
    full_path = os.path.join(output_dir, filename_base)

    if isinstance(img_data, np.ndarray):
        # Đây là một mảng ảnh, sử dụng plt.imshow
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(img_data)
        plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    elif isinstance(img_data, plt.Figure):
        # Đây là một đối tượng figure của Matplotlib, lưu trực tiếp
        img_data.savefig(full_path, bbox_inches='tight')
        plt.close(img_data)  # Đóng figure để giải phóng bộ nhớ
    else:
        print(f"Unsupported data type for saving: {type(img_data)}")
        return

    print(f"Image/Plot saved to: {full_path}")


# --- Hàm SubRegionDivision (Được định nghĩa lại nếu không import từ utils) ---
# Nếu hàm này đã được import từ utils.py, bạn có thể xóa định nghĩa này ở đây
def SubRegionDivision(image, mode="slico"):
    element_sets_V = []
    if mode == "slico":
        try:
            slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=30, ruler=20.0)
            slic.iterate(20)
            label_slic = slic.getLabels()
            number_slic = slic.getNumberOfSuperpixels()
            for i in range(number_slic):
                img_copp = image.copy()
                img_copp = img_copp * (label_slic == i)[:, :, np.newaxis]
                element_sets_V.append(img_copp)
        except AttributeError:
            print("cv2.ximgproc for SLIC not found. Ensure opencv-contrib-python is installed and properly linked.")
            return []
    elif mode == "seeds":
        try:
            seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2],
                                                       num_superpixels=50, num_levels=3)
            seeds.iterate(image, 10)
            label_seeds = seeds.getLabels()
            number_seeds = seeds.getNumberOfSuperpixels()
            for i in range(number_seeds):
                img_copp = image.copy()
                img_copp = img_copp * (label_seeds == i)[:, :, np.newaxis]
                element_sets_V.append(img_copp)
        except AttributeError:
            print("cv2.ximgproc for SEEDS not found. Ensure opencv-contrib-python is installed and properly linked.")
            return []
    return element_sets_V


# Khởi tạo mô hình CubSubModularExplanationV2
smdl = CubSubModularExplanationV2(
    cfg_path=os.path.join(project_root_dir, "configs", "cub", "submodular_cfg_cub_tf-resnet-v2.json"),
    k=50,
    lambda1=1,
    lambda2=1,
    lambda3=1,
    lambda4=1)

# Tải và xử lý ảnh đầu vào
original_image_path = os.path.join(project_root_dir, "examples", "Crested_Auklet_0059_794929.jpg")
image_for_cub = cv2.imread(original_image_path)
if image_for_cub is None:
    raise FileNotFoundError(f"Image not found at {original_image_path}")
image_for_cub = cv2.resize(image_for_cub, (224, 224))


# --- Hàm visualization đã chỉnh sửa để lưu file ---
def visualization(image_orig, submodular_image_set, saved_json_file, algorithm_mode, output_dir_name):
    """
    Visualize results and save to files.
    """
    insertion_ours_images = []

    insertion_image = submodular_image_set[0]
    insertion_ours_images.append(insertion_image)
    for smdl_sub_mask in submodular_image_set[1:]:
        insertion_image = insertion_image.copy() + smdl_sub_mask
        insertion_ours_images.append(insertion_image)

    insertion_ours_images_input_results = np.array(saved_json_file["consistency_score"])

    ours_best_index = np.argmax(insertion_ours_images_input_results)
    x = [(insertion_ours_image.sum(-1) != 0).sum() / (image_orig.shape[0] * image_orig.shape[1]) for
         insertion_ours_image in insertion_ours_images]
    i = len(x)

    # --- Tạo và lưu ảnh hiển thị (ax2) ---
    fig_img, ax_img = plt.subplots(figsize=(10, 10))
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
    save_image_to_file(img_to_save, f"{algorithm_mode}_image_best_index.png", output_dir_name)
    plt.close(fig_img)

    # --- Tạo và lưu biểu đồ đường (ax3) ---
    fig_plot, ax_plot = plt.subplots(figsize=(12, 8))
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

    save_image_to_file(fig_plot, f"{algorithm_mode}_insertion_plot.png", output_dir_name)  # Pass the figure object
    plt.close(fig_plot)

    auc = metrics.auc(x, insertion_ours_images_input_results)
    print(f"For {algorithm_mode} mode:")
    print(f"  Highest confidence: {insertion_ours_images_input_results.max():.4f}")
    print(f"  Final confidence: {insertion_ours_images_input_results[-1]:.4f}")
    print(f"  Insertion AUC: {auc:.4f}")
    print("-" * 30)


# --- Chạy quá trình Submodular Explanation cho CUB (SLIC & SEEDS) ---
print("\n--- Running Submodular Explanation with SLIC ---")
element_sets_V_slic = SubRegionDivision(image_for_cub, mode="slico")
if element_sets_V_slic:  # Chỉ chạy nếu Superpixel Division thành công
    smdl.k = len(element_sets_V_slic)
    submodular_image_slic, submodular_image_set_slic, saved_json_file_slic = smdl(element_sets_V_slic)
    visualization(image_for_cub, submodular_image_set_slic, saved_json_file_slic, algorithm_mode="slico",
                  output_dir_name=output_cub_directory)

print("\n--- Running Submodular Explanation with SEEDS ---")
element_sets_V_seeds = SubRegionDivision(image_for_cub, mode="seeds")
if element_sets_V_seeds:  # Chỉ chạy nếu Superpixel Division thành công
    smdl.k = len(element_sets_V_seeds)
    submodular_image_seeds, submodular_image_set_seeds, saved_json_file_seeds = smdl(element_sets_V_seeds)
    visualization(image_for_cub, submodular_image_set_seeds, saved_json_file_seeds, algorithm_mode="seeds",
                  output_dir_name=output_cub_directory)

print(f"\nAll CUB visualization outputs saved as PNG files in: {output_cub_directory}")

# --- Chạy quá trình Submodular Explanation cho HSIC (Grad-based) ---
print("\n--- Running Submodular Explanation with HSIC (Grad-based) ---")
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

# Tải lại mô hình để đảm bảo nó ở đúng state
# Dòng này cần được thực thi lại sau khi K.clear_session()
model = keras.models.load_model(smdl.cfg["recognition_model"]["model_path"])
input_img_tf = smdl.convert_prepare_image(image_for_cub)
# Lấy nhãn dự đoán để tính toán HSIC
predicted_label_idx = model(np.array([input_img_tf])).numpy().argmax()
labels_ohe = np.array([tf.one_hot(predicted_label_idx, smdl.cfg["recognition_model"]["num_classes"])])

explainer = HsicAttributionMethod(model)
# explanations sẽ là một mảng NumPy
explanations = explainer(np.array([input_img_tf]), labels_ohe)[0]  # Lấy mask từ batch 1

# Cần giải phóng bộ nhớ của mô hình TensorFlow và explainer
del model
del explainer
K.clear_session()  # Xóa session của Keras để giải phóng bộ nhớ GPU/CPU


def partition_by_mulit_grad(image, explanation_mask, grad_size=10, grad_num_per_set=4):
    """
    Divide the image into grad_size x grad_size areas, divide according to eplanation_mask, each division has grad_num_per_set grads.
    """
    partition_number = int(grad_size * grad_size / grad_num_per_set)
    components_image_list = []
    pool_z = cv2.resize(explanation_mask, (grad_size, grad_size))

    pool_z_flatten = pool_z.flatten()
    index = np.argsort(- pool_z_flatten)  # From high to low

    for i in range(partition_number):
        binary_mask_flat = np.zeros_like(pool_z_flatten, dtype=np.uint8)  # Sử dụng dtype=np.uint8
        binary_mask_flat[index[i * grad_num_per_set: (i + 1) * grad_num_per_set]] = 1
        binary_mask_resized = cv2.resize(
            binary_mask_flat.reshape((grad_size, grad_size)),  # reshape về 2D trước khi resize
            (image.shape[1], image.shape[0]),  # cv2.resize expects (width, height)
            interpolation=cv2.INTER_NEAREST)

        # Đảm bảo binary_mask_resized có 3 kênh để nhân với ảnh
        if len(binary_mask_resized.shape) == 2:
            binary_mask_resized = binary_mask_resized[:, :, np.newaxis]

        components_image_list.append(
            (image * binary_mask_resized).astype(np.uint8)
        )

    return components_image_list


element_sets_V_hsic = partition_by_mulit_grad(image_for_cub, explanations)  # Sử dụng image_for_cub đã resize
if element_sets_V_hsic:
    smdl.k = len(element_sets_V_hsic)
    submodular_image_hsic, submodular_image_set_hsic, saved_json_file_hsic = smdl(element_sets_V_hsic)
    visualization(image_for_cub, submodular_image_set_hsic, saved_json_file_hsic, algorithm_mode="hsic",
                  output_dir_name=output_cub_directory)  # Lưu vào thư mục CUB

# --- Chạy quá trình Submodular Explanation với SAM ---
print("\n--- Running Submodular Explanation with SAM ---")

# Load image for SAM (SAM often prefers original resolution or specific input size)
# Cần tải lại ảnh gốc, không phải ảnh đã resize
image_for_sam = cv2.imread(original_image_path)
if image_for_sam is None:
    raise FileNotFoundError(f"SAM image not found at {original_image_path}")

# Initialize SAM model
# Chắc chắn đã tải checkpoint sam_vit_h_4b8939.pth vào ckpt/pytorch_model/
sam_checkpoint_path = os.path.join(project_root_dir, "ckpt", "pytorch_model", "sam_vit_h_4b8939.pth")
if not os.path.exists(sam_checkpoint_path):
    print(f"SAM checkpoint not found at {sam_checkpoint_path}. Please download it.")
    # Exit or handle gracefully if SAM checkpoint is missing
else:
    sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
    sam_model.to(device)  # Chuyển SAM model sang GPU

    mask_generator = SamAutomaticMaskGenerator(sam_model, stability_score_thresh=0.0)  # Thử 0.0 để có nhiều mask hơn
    # mask_generator = SamAutomaticMaskGenerator(sam_model, stability_score_thresh=0.2)

    # Generate masks (SAM expects RGB input)
    masks = mask_generator.generate(cv2.cvtColor(image_for_sam, cv2.COLOR_BGR2RGB))


    def processing_sam_concepts(sam_masks, image):
        """
        Process the regions divided by SAM to prevent intersection of sub-regions.
            sam_mask: Masks generated by Segment Anything Model
        """
        num = len(sam_masks)
        mask_sets_V = [mask['segmentation'].astype(np.uint8) for mask in sam_masks]

        # Ensure all masks are the same size as the image for consistent operations
        image_h, image_w = image.shape[:2]
        processed_masks = []
        for mask_data in sam_masks:
            mask_resized = cv2.resize(mask_data['segmentation'].astype(np.uint8), (image_w, image_h),
                                      interpolation=cv2.INTER_NEAREST)
            processed_masks.append(mask_resized)

        # Logic to prevent intersection of sub-regions
        for i in range(num - 1):
            for j in range(i + 1, num):
                intersection_region = (processed_masks[i] + processed_masks[j] == 2).astype(np.uint8)
                if intersection_region.sum() == 0:
                    continue  # No intersection
                else:
                    proportion_1 = intersection_region.sum() / processed_masks[i].sum()
                    proportion_2 = intersection_region.sum() / processed_masks[j].sum()
                    if proportion_1 > proportion_2:
                        processed_masks[j] -= intersection_region
                    else:
                        processed_masks[i] -= intersection_region

        element_sets_V_sam = []
        for mask in processed_masks:
            # Only add masks that are not completely empty after intersection handling
            if mask.mean() > 0.0005:  # Threshold to filter out tiny or empty masks
                element_sets_V_sam.append(image * mask[:, :, np.newaxis])

        # Add the remaining background as a component
        # Calculate sum of all current elements to get the combined mask
        combined_mask_sum = np.array(element_sets_V_sam).sum(0)
        # If sum is not zero, convert to binary mask, otherwise it's just black
        if combined_mask_sum.sum() > 0:
            remaining_background_mask = (combined_mask_sum.sum(axis=-1) == 0).astype(np.uint8)
            element_sets_V_sam.append(image * remaining_background_mask[:, :, np.newaxis])
        else:
            # If all elements were tiny, add the whole image as background
            element_sets_V_sam.append(image)

        return element_sets_V_sam


    # Resize image for SAM's input to the same size as model's expected input (224, 224) if needed for smdl
    # However, SAM output masks are at original image resolution.
    # We should resize the image *after* segmenting, if smdl expects 224x224.
    # Or, resize the masks to 224x224 before applying them to the 224x224 image_for_cub.
    # Let's resize the image_for_sam to 224x224 *before* passing to smdl for consistency
    # with previous CUB operations.
    resized_image_for_smdl = cv2.resize(image_for_sam, (224, 224))

    element_sets_V_sam = processing_sam_concepts(masks, resized_image_for_smdl)  # Use resized image for SAM concepts
    if element_sets_V_sam:
        smdl.k = len(element_sets_V_sam)
        submodular_image_sam, submodular_image_set_sam, saved_json_file_sam = smdl(element_sets_V_sam)
        visualization(resized_image_for_smdl, submodular_image_set_sam, saved_json_file_sam, algorithm_mode="sam",
                      output_dir_name=output_sam_directory)  # Lưu vào thư mục SAM

print(f"\nAll SAM visualization outputs saved as PNG files in: {output_sam_directory}")
print(f"\nCheck the 'Files' section (folder icon on the left) to download them.")