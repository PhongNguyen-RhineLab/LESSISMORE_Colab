import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import clip
import torch
from torchvision import transforms
from sklearn import metrics # Đảm bảo sklearn được import

# --- Cấu hình môi trường GPU (Đảm bảo đã chọn GPU runtime trong Colab) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- QUAN TRỌNG: Điều chỉnh os.chdir hoặc đường dẫn ---
# Nếu script của bạn nằm trong /content/LESSISMORE_Colab/
# và các module như 'utils' nằm trong cùng thư mục (hoặc thư mục con),
# bạn KHÔNG NÊN sử dụng os.chdir("../").
# Nếu bạn đã chạy lệnh này và file của bạn bị lỗi đường dẫn, hãy xóa nó.
# os.chdir("../") # Comment hoặc xóa dòng này nếu script của bạn đã ở đúng vị trí

# --- Đảm bảo các module cục bộ được import đúng đường dẫn ---
# Thêm đường dẫn tới thư mục gốc của dự án vào sys.path
# Điều này giúp Python tìm thấy 'utils' và 'models'
import sys
project_root = "/content/LESSISMORE_Colab"
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import SubRegionDivision # Đảm bảo SubRegionDivision được import tường minh nếu cần
from models.submodular_vit_torch import MultiModalSubModularExplanation

# --- Thiết lập thư mục đầu ra cho ảnh ---
output_image_directory = "/content/LESSISMORE_Colab/image/test_result/Clip_result"
os.makedirs(output_image_directory, exist_ok=True)
print(f"Output directory ensured: {output_image_directory}")


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

def save_image_to_file(img_array, filename_base="output_image.png", output_dir=output_image_directory):
    """
    Lưu ảnh vào file thay vì hiển thị trực tiếp.
    Input:
        img_array: numpy array của ảnh (dạng RGB).
        filename_base: Tên file cơ sở (ví dụ: "my_image.png").
        output_dir: Thư mục để lưu ảnh. Mặc định là thư mục đã thiết lập.
    """
    full_path = os.path.join(output_dir, filename_base)

    plt.figure(figsize=(8, 8)) # Tạo một figure mới cho mỗi ảnh
    plt.axis('off') # Tắt trục
    plt.imshow(img_array)
    plt.savefig(full_path, bbox_inches='tight', pad_inches=0) # Lưu ảnh, cắt bỏ khoảng trắng thừa
    plt.close() # Đóng plot để giải phóng bộ nhớ
    print(f"Image saved to: {full_path}") # In ra đường dẫn để xác nhận

class CLIPModel_Super(torch.nn.Module):
    def __init__(self,
                 type="ViT-L/14",
                 download_root=None,
                 device = "cuda"):
        super().__init__()
        self.device = device
        # Đảm bảo gói 'clip' đã được cài đặt và import đúng
        self.model, _ = clip.load(type, device=self.device, download_root=download_root)

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
        image: An image read by opencv [w,h,c] (BGR format)
    Output:
        image: After preproccessing, is a tensor [c,w,h] (Normalized)
    """
    image = Image.fromarray(image[:,:,::-1]) # Chuyển BGR sang RGB cho PIL
    image = data_transform(image)
    return image

# --- Khởi tạo và xử lý ảnh đầu vào ---
image_path = "/content/LESSISMORE_Colab/examples/Crested_Auklet_0059_794929.jpg"
image_raw = cv2.imread(image_path)
if image_raw is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

image_transformed = transform_vision_data(image_raw)
print(f"Transformed image shape: {image_transformed.shape}")

# Instantiate model
vis_model = CLIPModel_Super("ViT-L/14", device=device, download_root=".checkpoints/CLIP")
vis_model.eval()
vis_model.to(device)
print("CLIP model loaded successfully.")

vis_feature = vis_model(image_transformed.unsqueeze(0).to(device))
print(f"The output size of the visual feature is {vis_feature.shape}.")

# --- Xử lý Superpixel SLIC và lưu ảnh ---
img_resized = cv2.resize(image_raw, (224,224))
slic = cv2.ximgproc.createSuperpixelSLIC(img_resized, region_size=30, ruler = 20.0)
slic.iterate(20)
mask_slic = slic.getLabelContourMask()
label_slic = slic.getLabels()
number_slic = slic.getNumberOfSuperpixels()
mask_inv_slic = cv2.bitwise_not(mask_slic)
img_slic = cv2.bitwise_and(img_resized, img_resized, mask = mask_inv_slic)

# Lưu ảnh SLIC thay vì imshow
save_image_to_file(process_image_for_display_or_save(img_slic), "slic_output.png")
print("SLIC image saved as slic_output.png")

# --- Chuẩn bị dữ liệu cho Submodular Explanation ---
element_sets_V = SubRegionDivision(img_resized, mode="slico")

text_list=[
    "A dog.", "A car.", "A bird.", "An airplane.", "A bicycle.",
    "A boat.", "A cat.", "A chair", "A cow.", "A diningtable.",
    "A horse.", "A motorbike.", "A person.", "A pottedplant.", "A sheep.",
    "A sofa.", "A train.", "A tvmonitor."]
text_modal_input = clip.tokenize(text_list).to(device)

with torch.no_grad():
    semantic_feature = vis_model.model.encode_text(text_modal_input) * 100

explainer = MultiModalSubModularExplanation(
    vis_model, semantic_feature, transform_vision_data, device=device, lambda1 = 0, lambda2 = 0.05, lambda3 = 1, lambda4=1)
explainer.k = len(element_sets_V)
submodular_image, submodular_image_set, saved_json_file = explainer(element_sets_V, id=0)

# --- Hàm visualization đã chỉnh sửa để lưu file ---
def visualization(image_orig, submodular_image_set, saved_json_file, index=None, output_prefix="vis_output", output_dir=output_image_directory):
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

    if index is None:
        ours_best_index = np.argmax(insertion_ours_images_input_results)
    else:
        ours_best_index = index
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
    ax_img.set_title('Ours', fontsize=24) # Giảm fontsize cho phù hợp với figsize nhỏ hơn
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
    ax_img.imshow(img_to_save)

    plt.savefig(os.path.join(output_dir, f"{output_prefix}_image_index_{index}.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig_img) # Đóng figure ảnh để giải phóng bộ nhớ


    # --- Tạo và lưu biểu đồ đường (ax3) ---
    fig_plot, ax_plot = plt.subplots(figsize=(12, 8)) # Tạo figure mới cho biểu đồ đường
    ax_plot.set_xlim((0, 1))
    ax_plot.set_ylim((0, 1))
    ax_plot.tick_params(axis='x', labelsize=20) # Giảm fontsize
    ax_plot.tick_params(axis='y', labelsize=20) # Giảm fontsize
    ax_plot.set_title('Insertion', fontsize=24) # Giảm fontsize
    ax_plot.set_ylabel('Recognition Score', fontsize=22) # Giảm fontsize
    ax_plot.set_xlabel('Percentage of image revealed', fontsize=22) # Giảm fontsize

    x_ = x[:i]
    ours_y = insertion_ours_images_input_results[:i]
    ax_plot.plot(x_, ours_y, color='dodgerblue', linewidth=3.5)

    ax_plot.scatter(x_[-1], ours_y[-1], color='dodgerblue', s=54)
    ax_plot.plot([x_[ours_best_index], x_[ours_best_index]], [0, 1], color='red', linewidth=3.5)

    plt.savefig(os.path.join(output_dir, f"{output_prefix}_plot_index_{index}.png"), bbox_inches='tight')
    plt.close(fig_plot) # Đóng figure biểu đồ đường


    auc = metrics.auc(x, insertion_ours_images_input_results)
    print(f"For index {index}:")
    print(f"  Highest confidence: {insertion_ours_images_input_results.max():.4f}")
    print(f"  Final confidence: {insertion_ours_images_input_results[-1]:.4f}")
    print(f"  Insertion AUC: {auc:.4f}")

# --- Gọi hàm visualization để tạo và lưu các file ảnh ---
visualization(img_resized, submodular_image_set, saved_json_file, index=0, output_prefix="vis_output_0", output_dir=output_image_directory)
visualization(img_resized, submodular_image_set, saved_json_file, index=4, output_prefix="vis_output_4", output_dir=output_image_directory)
visualization(img_resized, submodular_image_set, saved_json_file, index=10, output_prefix="vis_output_10", output_dir=output_image_directory)

print("\nAll visualization outputs saved as PNG files in your specified directory.")
print(f"Check the 'Files' section (folder icon on the left) under: {output_image_directory} to download them.")