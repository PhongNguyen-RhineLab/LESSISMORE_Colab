import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json # Ensure json is imported for consistent utility usage

# --- QUAN TRỌNG: Thiết lập thư mục gốc của dự án ---
# Đảm bảo thư mục làm việc hiện tại là project_root_dir
# Giả định cấu trúc thư mục của bạn là LESSISMORE_Colab/ (project root)
# và script này nằm trong LESSISMORE_Colab/ (hoặc một thư mục con).
# Nếu script này nằm trong một thư mục con, ví dụ: LESSISMORE_Colab/scripts/
# bạn cần điều chỉnh project_root_dir tương ứng.
project_root_dir = "/content/LESSISMORE_Colab"
# Sử dụng try-except để thay đổi thư mục một cách an toàn hơn
try:
    os.chdir(project_root_dir)
    print(f"Current working directory set to: {os.getcwd()}")
except FileNotFoundError:
    print(f"Error: Project root directory not found at {project_root_dir}")
    print("Please ensure the '/content/LESSISMORE_Colab' path is correct and mounted.")
    exit() # Thoát nếu không thể thay đổi thư mục

# Kiểm tra GPU có sẵn
import torch
# Code 1 dùng "cuda:0", Code 2 dùng "cuda:1". Chuyển về "cuda:0" hoặc đơn giản là "cuda"
# để nó tự chọn GPU khả dụng đầu tiên nếu có nhiều GPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is NOT available. Running on CPU (this will be very slow).")

# Thiết lập thư mục đầu ra MỚI cho QuiltNet
output_quiltnet_directory = os.path.join(project_root_dir, "image", "test_result", "quiltnet_results")
os.makedirs(output_quiltnet_directory, exist_ok=True)
print(f"Output directory for QuiltNet results ensured: {output_quiltnet_directory}")


# Import các lớp từ xplique (chưa dùng nhưng giữ lại để đồng bộ)
# from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
#                                   SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
#                                   GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)

import matplotlib
from sklearn import metrics

# Cấu hình Matplotlib
matplotlib.get_cachedir()
plt.rc('font', family="Times New Roman") # Giữ font Times New Roman nếu có, nếu không sẽ có cảnh báo
plt.style.use('seaborn-v0_8-darkgrid') # Đã cập nhật style để tránh cảnh báo, như trong Code 1

# --- Hàm hỗ trợ để xử lý và lưu ảnh (từ Code 1) ---
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


# --- Hàm SubRegionDivision (được định nghĩa lại nếu không import từ utils) ---
# Đảm bảo hàm này được định nghĩa hoặc import. Trong code 2, nó được import từ utils.
# Để đồng bộ với code 1, nếu 'utils.py' không được đảm bảo,
# ta có thể định nghĩa lại nó tại đây, tương tự như cách Code 1 đã làm.
# Tuy nhiên, nếu 'utils.py' tồn tại và chứa hàm này, việc import là tốt nhất.
# Giả sử `utils.py` có trong `project_root_dir`.
try:
    from utils import SubRegionDivision
    print("SubRegionDivision imported from utils.py")
except ImportError:
    print("Warning: Could not import SubRegionDivision from utils.py. Defining locally.")
    # Fallback definition if utils.py isn't found or SubRegionDivision isn't in it
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
                seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
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


# Import QuiltNet components
from open_clip import create_model_from_pretrained, get_tokenizer
import torch
from torchvision import transforms

class QuiltModel_Super(torch.nn.Module):
    def __init__(self,
                 download_root=os.path.join(project_root_dir, ".checkpoints", "QuiltNet-B-32"), # Updated path
                 device="cuda"):
        super().__init__()
        # 'hf-hub:wisdomik/QuiltNet-B-32' sẽ tự động tải xuống nếu không tìm thấy trong cache_dir
        self.model, _ = create_model_from_pretrained('hf-hub:wisdomik/QuiltNet-B-32', cache_dir=download_root)
        self.device = device
        self.model.to(self.device) # Ensure model is moved to device during init
        self.model.eval() # Set to eval mode

    def forward(self, vision_inputs):
        """
        Input:
            vision_inputs: torch.size([B,C,W,H])
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        with torch.no_grad():
            image_features = self.model.encode_image(vision_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

data_transform = transforms.Compose(
    [
        transforms.Resize(
            (224,224), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

def transform_vision_data(image):
    """
    Input:
        image: An image read by opencv [w,h,c] (BGR format expected from cv2.imread)
    Output:
        image: After preproccessing, is a tensor [c,w,h]
    """
    # cv2 reads image as BGR, PIL expects RGB. Convert before PIL.fromarray
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = data_transform(image)
    return image

# Instantiate QuiltNet model
vis_model = QuiltModel_Super(device=device) # Pass device to constructor
print("Load Quilt model completed.")

# --- Visualize SLIC superpixels (original code snippet from Code 2) ---
# This part is just for display, not directly related to submodular explanation output
image_path_for_slic = os.path.join(project_root_dir, "examples", "lungaca1.jpeg")
img_for_slic = cv2.imread(image_path_for_slic)
if img_for_slic is None:
    print(f"Error: Image for SLIC not found at {image_path_for_slic}")
else:
    img_for_slic = cv2.resize(img_for_slic, (224,224))
    slic = cv2.ximgproc.createSuperpixelSLIC(img_for_slic, region_size=30, ruler = 20.0)
    slic.iterate(20)
    mask_slic = slic.getLabelContourMask()
    # label_slic = slic.getLabels() # Not used for display
    # number_slic = slic.getNumberOfSuperpixels() # Not used for display
    mask_inv_slic = cv2.bitwise_not(mask_slic)
    img_slic_display = cv2.bitwise_and(img_for_slic, img_for_slic, mask =  mask_inv_slic)
    # The original Code 2 had `imshow(img_slic)`, we'll save this to file instead
    save_image_to_file(process_image_for_display_or_save(img_slic_display),
                       "slic_superpixels_visualization.png", output_quiltnet_directory)


# --- Prepare data for Submodular Explanation with QuiltNet ---
image_for_explanation_path = os.path.join(project_root_dir, "examples", "lungaca1.jpeg")
image_for_explanation = cv2.imread(image_for_explanation_path)
if image_for_explanation is None:
    print(f"Error: Image for explanation not found at {image_for_explanation_path}")
    exit()
image_for_explanation = cv2.resize(image_for_explanation, (224, 224))

element_sets_V = SubRegionDivision(image_for_explanation, mode="slico")

from models.submodular_vit_torch import MultiModalSubModularExplanation

template = 'a histopathology slide showing '
labels = ["lung adenocarcinoma",
        "benign lung",
        "lung squamous cell carcinoma"
]

tokenizer = get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
texts = tokenizer([template + l for l in labels], context_length=77).to(device)

with torch.no_grad():
    # In Code 2, semantic_feature was multiplied by 100 before normalization,
    # and then normalized. In Code 1, ImageBind outputs were used directly.
    # We'll keep the Code 2 logic for semantic_feature as it's specific to QuiltNet's context.
    semantic_feature = vis_model.model.encode_text(texts) * 100
    # semantic_feature /= semantic_feature.norm(dim=-1, keepdim=True) * 100 # Original Code 2 had this commented out.
    # Let's normalize it after scaling by 100 if that was the intent, or remove the *100 if normalization handles it.
    # For now, keeping as original Code 2's effectively uncommented line:
    # semantic_feature /= semantic_feature.norm(dim=-1, keepdim=True) # Normalized implicitly by QuiltModel_Super in forward

# Khởi tạo explainer
explainer = MultiModalSubModularExplanation(
    vis_model, semantic_feature, transform_vision_data, device=device, lambda1 = 0, lambda2 = 0.05, lambda3 = 1, lambda4=1)

if element_sets_V: # Chỉ chạy nếu SubRegionDivision tạo ra các phần tử
    explainer.k = len(element_sets_V)
    submodular_image, submodular_image_set, saved_json_file = explainer(element_sets_V, id=0)

    # --- Hàm visualization đã chỉnh sửa để lưu file (từ Code 1) ---
    def visualization(image_orig, submodular_image_set, saved_json_file, algorithm_mode, output_dir, index=None):
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
        # Điều chỉnh figsize cho phù hợp với Code 2 nếu muốn lớn hơn (24,8)
        fig_img, ax_img = plt.subplots(figsize=(10, 10)) # Đổi lại 10,10 như Code 1 hoặc 24,8 như Code 2
        ax_img.spines["left"].set_visible(False)
        ax_img.spines["right"].set_visible(False)
        ax_img.spines["top"].set_visible(False)
        ax_img.spines["bottom"].set_visible(False)
        ax_img.xaxis.set_visible(False)
        ax_img.yaxis.set_visible(False)
        ax_img.set_title(f'Ours ({algorithm_mode})', fontsize=24) # Đổi lại 54 như Code 2 nếu muốn lớn hơn
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
        fig_plot, ax_plot = plt.subplots(figsize=(12, 8)) # Đổi lại 24,8 như Code 2 nếu muốn lớn hơn
        ax_plot.set_xlim((0, 1))
        ax_plot.set_ylim((0, 1))
        ax_plot.tick_params(axis='x', labelsize=20) # Đổi lại 36 như Code 2 nếu muốn lớn hơn
        ax_plot.tick_params(axis='y', labelsize=20) # Đổi lại 36 như Code 2 nếu muốn lớn hơn
        ax_plot.set_title(f'Insertion ({algorithm_mode})', fontsize=24) # Đổi lại 54 như Code 2 nếu muốn lớn hơn
        ax_plot.set_ylabel('Recognition Score', fontsize=22) # Đổi lại 44 như Code 2 nếu muốn lớn hơn
        ax_plot.set_xlabel('Percentage of image revealed', fontsize=22) # Đổi lại 44 như Code 2 nếu muốn lớn hơn

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

    # Chạy visualization cho các index cụ thể, tương tự Code 1
    print("\n--- Visualizing QuiltNet results for specific indices ---")
    visualization(image_for_explanation, submodular_image_set, saved_json_file, algorithm_mode="quiltnet", output_dir=output_quiltnet_directory, index=1)
    visualization(image_for_explanation, submodular_image_set, saved_json_file, algorithm_mode="quiltnet", output_dir=output_quiltnet_directory, index=10)
else:
    print("Skipping Submodular Explanation for QuiltNet because element_sets_V is empty.")


print(f"\nAll QuiltNet visualization outputs saved as PNG files in: {output_quiltnet_directory}")
print(f"Check the 'Files' section (folder icon on the left) to download them.")