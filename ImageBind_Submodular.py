import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json # Thêm import json để xử lý file cấu hình nếu cần

# Cấu hình môi trường GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- QUAN TRỌNG: Thiết lập thư mục gốc của dự án ---
# Đảm bảo thư mục làm việc hiện tại là project_root_dir
project_root_dir = "/content/LESSISMORE_Colab"
# Sử dụng try-except để thay đổi thư mục một cách an toàn hơn
try:
    os.chdir(project_root_dir)
    print(f"Current working directory set to: {os.getcwd()}")
except FileNotFoundError:
    print(f"Error: Project root directory not found at {project_root_dir}")
    print("Please ensure the '/content/LESSISMORE_Colab' path is correct.")
    exit() # Thoát nếu không thể thay đổi thư mục

# Kiểm tra GPU có sẵn
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Running on CPU (this will be very slow).")


# Thiết lập thư mục đầu ra MỚI cho ImageBind
output_imagebind_directory = os.path.join(project_root_dir, "image", "test_result", "imagebind_results")
os.makedirs(output_imagebind_directory, exist_ok=True)
print(f"Output directory for ImageBind results ensured: {output_imagebind_directory}")

# Import các lớp từ xplique
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)

import matplotlib
from sklearn import metrics

# Cấu hình Matplotlib
matplotlib.get_cachedir()
# plt.rc('font', family="Times New Roman") # Comment out nếu font Times New Roman không có
plt.style.use('seaborn-v0_8-darkgrid') # Đã cập nhật style để tránh cảnh báo

# --- Hàm hỗ trợ để xử lý và lưu ảnh ---
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
        plt.close(img_data) # Đóng figure để giải phóng bộ nhớ
    else:
        print(f"Unsupported data type for saving: {type(img_data)}")
        return

    print(f"Image/Plot saved to: {full_path}")


# --- Hàm SubRegionDivision (Được định nghĩa lại nếu không import từ utils) ---
def SubRegionDivision(image, mode="slico"):
    element_sets_V = []
    if mode == "slico":
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
            print("cv2.ximgproc for SLIC not found. Ensure opencv-contrib-python is installed and properly linked.")
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
            print("cv2.ximgproc for SEEDS not found. Ensure opencv-contrib-python is installed and properly linked.")
            return []
    return element_sets_V

# Import ImageBind components
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class ImageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, vision_inputs):
        """
        Input:
            vision_inputs: torch.size([B,C,W,H])
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        inputs = {
            "vision": vision_inputs,
        }

        with torch.no_grad():
            embeddings = self.base_model(inputs)

        return embeddings["vision"]

def transform_vision_data(image):
    """
    Input:
        image: An image read by opencv [w,h,c]
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    data_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = data_transform(image)
    return image

# --- Load ImageBind Model ---
# Ensure ImageBind model is downloaded to ckpt/pytorch_model/
imagebind_model_path = os.path.join(project_root_dir, "ckpt", "pytorch_model", "imagebind_huge.pth")
if not os.path.exists(imagebind_model_path):
    print(f"ImageBind model checkpoint not found at {imagebind_model_path}.")
    print("Please ensure 'imagebind_huge.pth' is in the 'ckpt/pytorch_model/' directory.")
    # Exit or handle gracefully if model is missing
    # For now, let's proceed assuming it's loaded by pretrained=True,
    # but for local files, explicit loading is better.
    # model = imagebind_model.imagebind_huge(pretrained=False) # if loading local
    # model.load_state_dict(torch.load(imagebind_model_path, map_location='cpu'))

model = imagebind_model.imagebind_huge(pretrained=True) # Will download if not found
model.eval()
model.to(device)
vis_model = ImageBindModel_Super(model)
print("Load ImageBind model completed.")

# --- Test ImageBind feature extraction (optional, for debugging) ---
test_image_path = os.path.join(project_root_dir, "examples", "dog_image.jpg")
test_image = cv2.imread(test_image_path)
if test_image is None:
    print(f"Error: Test image not found at {test_image_path}")
else:
    test_image_transformed = transform_vision_data(test_image)
    vis_feature = vis_model(test_image_transformed.unsqueeze(0).to(device))
    print(f"The output size of the visual feature is {vis_feature.shape}.")

# --- Visualize SLIC superpixels (original code snippet) ---
# This part is just for display, not directly related to submodular explanation output
img_for_slic = cv2.imread(test_image_path)
if img_for_slic is None:
    print(f"Error: Image for SLIC not found at {test_image_path}")
else:
    img_for_slic = cv2.resize(img_for_slic, (224, 224))
    slic = cv2.ximgproc.createSuperpixelSLIC(img_for_slic, region_size=30, ruler = 20.0)
    slic.iterate(20)
    mask_slic = slic.getLabelContourMask()
    mask_inv_slic = cv2.bitwise_not(mask_slic)
    img_slic_display = cv2.bitwise_and(img_for_slic, img_for_slic, mask = mask_inv_slic)
    # imshow(img_slic_display) # Removed imshow as it's for interactive display

# --- Prepare data for Submodular Explanation with ImageBind ---
image_for_explanation = cv2.imread(test_image_path)
if image_for_explanation is None:
    print(f"Error: Image for explanation not found at {test_image_path}")
    exit()
image_for_explanation = cv2.resize(image_for_explanation, (224, 224))

element_sets_V = SubRegionDivision(image_for_explanation, mode="slico")

from models.submodular_vit_torch import MultiModalSubModularExplanation

text_list=[
    "A dog.", "A car.", "A bird.", "An airplane.", "A bicycle.",
    "A boat.", "A cat.", "A chair", "A cow.", "A diningtable.",
    "A horse.", "A motorbike.", "A person.", "A pottedplant.", "A sheep.",
    "A sofa.", "A train.", "A tvmonitor."]
text_modal_input = data.load_and_transform_text(text_list, device)
input_text = {
    "text": text_modal_input
}
with torch.no_grad():
    semantic_feature = model(input_text)["text"]

# Khởi tạo explainer
explainer = MultiModalSubModularExplanation(
    vis_model, semantic_feature, transform_vision_data, device=device, lambda1 = 0, lambda2 = 0.1, lambda4=5)

if element_sets_V: # Chỉ chạy nếu SubRegionDivision tạo ra các phần tử
    explainer.k = len(element_sets_V)
    submodular_image, submodular_image_set, saved_json_file = explainer(element_sets_V, id=0)

    # --- Hàm visualization đã chỉnh sửa để lưu file ---
    def visualization(image_orig, submodular_image_set, saved_json_file, algorithm_mode, output_dir=output_imagebind_directory, index=None):
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

        if index is None:
            ours_best_index = np.argmax(insertion_ours_images_input_results)
        else:
            ours_best_index = index # Sử dụng index được truyền vào

        x = [(insertion_ours_image.sum(-1)!=0).sum() / (image_orig.shape[0] * image_orig.shape[1]) for insertion_ours_image in insertion_ours_images]
        i = len(x)

        # --- Tạo và lưu ảnh hiển thị ---
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
        mask[mask>0] = 1

        dilate = cv2.dilate(mask, kernel, 3)
        edge = dilate - mask
        image_debug = image_orig.copy()
        image_debug[mask>0] = image_debug[mask>0] * 0.5
        image_debug[edge>0] = np.array([0,0,255])

        img_to_save = process_image_for_display_or_save(image_debug)
        save_image_to_file(img_to_save, f"{algorithm_mode}_image_best_index_{index if index is not None else 'auto'}.png", output_dir)
        plt.close(fig_img)


        # --- Tạo và lưu biểu đồ đường ---
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

        save_image_to_file(fig_plot, f"{algorithm_mode}_insertion_plot_{index if index is not None else 'auto'}.png", output_dir)
        plt.close(fig_plot)


        auc = metrics.auc(x, insertion_ours_images_input_results)
        print(f"For {algorithm_mode} mode (index={index if index is not None else 'auto'}):")
        print(f"  Highest confidence: {insertion_ours_images_input_results.max():.4f}")
        print(f"  Final confidence: {insertion_ours_images_input_results[-1]:.4f}")
        print(f"  Insertion AUC: {auc:.4f}")
        print("-" * 30)

    # Chạy visualization cho các index cụ thể
    print("\n--- Visualizing ImageBind results for specific indices ---")
    visualization(image_for_explanation, submodular_image_set, saved_json_file, algorithm_mode="imagebind", output_dir=output_imagebind_directory, index=1)
    visualization(image_for_explanation, submodular_image_set, saved_json_file, algorithm_mode="imagebind", output_dir=output_imagebind_directory, index=10)
else:
    print("Skipping Submodular Explanation for ImageBind because element_sets_V is empty.")


print(f"\nAll ImageBind visualization outputs saved as PNG files in: {output_imagebind_directory}")
print(f"Check the 'Files' section (folder icon on the left) to download them.")