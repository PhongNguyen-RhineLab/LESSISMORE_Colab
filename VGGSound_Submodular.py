import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from PIL import Image
# matplotlib.get_cachedir() # Không cần thiết cho Colab
# plt.rc('font', family="Times New Roman") # Bỏ comment nếu font gây lỗi, hoặc bỏ dòng này để dùng font mặc định

# Cấu hình môi trường GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- QUAN TRỌNG: Thiết lập thư mục gốc của dự án ---
project_root_dir = "/content/LESSISMORE_Colab"
try:
    # Đảm bảo thư mục gốc tồn tại và chuyển đến đó
    if not os.path.exists(project_root_dir):
        print(f"Project root directory not found at {project_root_dir}. Please make sure it's mounted or created.")
        # Nếu đang chạy từ thư mục mẹ (ví dụ /content), có thể cần tạo nó
        # os.makedirs(project_root_dir, exist_ok=True)
        # Nếu project_root_dir là nơi chứa các file, hãy đảm bảo nó được tải lên
        exit() # Thoát nếu thư mục không tồn tại

    os.chdir(project_root_dir)
    print(f"Current working directory set to: {os.getcwd()}")
except Exception as e:
    print(f"Error changing directory: {e}")
    print("Please ensure the '/content/LESSISMORE_Colab' path is correct and accessible.")
    exit()

# Kiểm tra GPU có sẵn
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Running on CPU (this will be very slow).")

# Thiết lập thư mục đầu ra MỚI cho Audio Explanation
output_audio_directory = os.path.join(project_root_dir, "image", "test_result", "AudioBind_result")
os.makedirs(output_audio_directory, exist_ok=True)
print(f"Output directory for AudioBind results ensured: {output_audio_directory}")


# https://github.com/facebookresearch/ImageBind
from imagebind import data
from imagebind.data import waveform2melspec, get_clip_timepoints
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# from utils import vggsound_template, vggsound_classes # Nếu utils không có, bạn cần tự định nghĩa hoặc bỏ qua

from sklearn import metrics

import torch
from torchvision import transforms

import torchaudio
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

clip_sampler = ConstantClipsPerVideoSampler(
    clip_duration=2, clips_per_video=3
)


# --- Hàm hỗ trợ để xử lý và lưu ảnh/biểu đồ ---
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
        img_data.savefig(full_path, bbox_inches='tight', pad_inches=0) # Thêm pad_inches=0 để loại bỏ khoảng trắng
        plt.close(img_data) # Đóng figure để giải phóng bộ nhớ
    else:
        print(f"Unsupported data type for saving: {type(img_data)}")
        return

    print(f"Image/Plot saved to: {full_path}")


def denormalize(tensor, mean, std):
    """
    对归一化后的张量进行反归一化处理。

    参数:
    tensor: 归一化后的张量
    mean: 均值
    std: 标准差

    返回:
    反归一化后的张量
    """
    return tensor * std + mean


def visualize_audio_spectrogram(tensor, filename, output_dir, mean=-4.268, std=9.138, channel=0):
    """
    对音频频谱张量进行反归一化并可视化并 lưu.

    参数:
    tensor: 形状为 (C, T, H, W) hoặc (C, H, W) của spectrogram, hoặc (B, C, T, H, W) từ read_audio
    filename: tên file để lưu
    output_dir: thư mục để lưu
    mean: 归一化时使用的均值
    std: 归一化时使用的标准差
    channel: kênh để可视化 (mặc định 0)
    """
    # Xử lý input tensor có thể có batch dimension (từ read_audio)
    if tensor.ndim == 4: # If it's (B, C, H, W) or (B, C, T, H, W)
        tensor = tensor[0] # Lấy batch đầu tiên nếu có
    if tensor.ndim == 3 and tensor.shape[0] > 1: # If it's (C, H, W) or (C, T, H, W)
        # Giả sử chúng ta muốn lấy một kênh và một thời gian cụ thể (nếu có T)
        # Nếu tensor là [C, 1, H, W] thì lấy [channel, 0, :, :]
        # Nếu tensor là [C, H, W] thì lấy [channel, :, :]
        if tensor.shape[1] == 1: # [C, 1, H, W]
            tensor = tensor[channel, 0, :, :]
        else: # [C, H, W]
            tensor = tensor[channel, :, :]
    elif tensor.ndim == 2: # Already [H, W]
        pass # Không cần xử lý

    # 对张量进行反归一化
    denormalized_tensor = denormalize(tensor, mean, std)

    # 绘制反归一化后的频谱图
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(denormalized_tensor, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(im, format='%+2.0f dB')
    ax.set_title('Denormalized Audio Spectrogram')
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Mel Frequency Bins')
    save_image_to_file(fig, filename, output_dir)


def visualize_rgb_spectrogram(spectrogram, filename, output_dir):
    """
    可视化 RGB 梅尔频谱图并 lưu.

    参数:
    spectrogram: 形状为 (3, 128, 204) 的梅尔频谱图（3 个通道）
    filename: tên file để lưu
    output_dir: thư mục để lưu
    """
    # Ensure the input spectrogram has 3 channels
    assert spectrogram.shape[0] == 3, "Spectrogram must have 3 channels."

    # Convert the spectrogram from (3, H, W) to (H, W, 3)
    spectrogram_rgb = np.transpose(spectrogram, (1, 2, 0))

    # Display the RGB Mel spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(spectrogram_rgb, aspect='auto', origin='lower')
    ax.set_title('RGB Spectrogram')
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Mel Frequency Bins')
    save_image_to_file(fig, filename, output_dir)


def convert_mfcc(spectrogram, mean=-4.268, std=9.138, channel=0):
    # spectrogram có thể là [B, C, T, H, W] hoặc [C, T, H, W]
    if spectrogram.ndim == 5: # [B, C, T, H, W]
        spectrogram = spectrogram[0] # Lấy batch đầu tiên
    if spectrogram.ndim == 4: # [C, T, H, W]
        spectrogram = spectrogram[channel, 0, :, :] # Lấy kênh và thời điểm đầu tiên
    elif spectrogram.ndim == 3: # [C, H, W] (nếu transform đã bỏ T)
        spectrogram = spectrogram[channel, :, :] # Lấy kênh
    else:
        # Nếu đã là [H, W] thì không làm gì
        pass

    denormalized_tensor = denormalize(spectrogram, mean, std)
    return denormalized_tensor


class ImageBindModel_Super(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, audio_inputs):
        """
        Input:
            audio_inputs: torch.size([B,C,W,H]) -> should be [B,C,T,W,H] for video/audio-like input
        Output:
            embeddings: a d-dimensional vector torch.size([B,d])
        """
        inputs = {
            "audio": audio_inputs,
        }

        with torch.no_grad():
            embeddings = self.base_model(inputs)

        return embeddings["audio"]

def read_audio(
    audio_path,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    mean= -4.268,
    std= 9.138
):
    waveform, sr = torchaudio.load(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sample_rate
        )
    all_clips_timepoints = get_clip_timepoints(
        clip_sampler, waveform.size(1) / sample_rate
    )
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = waveform[
            :,
            int(clip_timepoints[0] * sample_rate) : int(
                clip_timepoints[1] * sample_rate
            ),
        ]
        waveform_melspec = waveform2melspec(
            waveform_clip, sample_rate, num_mel_bins, target_length
        )
        all_clips.append(waveform_melspec)

    normalize = transforms.Normalize(mean=mean, std=std)
    all_clips = [normalize(ac).to(device) for ac in all_clips]

    all_clips = torch.stack(all_clips, dim=0)
    return all_clips.cpu().numpy()

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

audio_model = ImageBindModel_Super(model)
print("load imagebind model")

# --- Đường dẫn file audio: Cập nhật để phù hợp với Colab ---
# Đặt file audio vào thư mục examples hoặc tải lên Colab
# Ví dụ: nếu bạn đặt nó vào project_root_dir/examples/
audio_file_name = "bZadyuv2utE_000080.flac" # Giả sử tên file
audio_path_colab = os.path.join(project_root_dir, "datasets", "vggsound", audio_file_name)

# Kiểm tra sự tồn tại của file audio
if not os.path.exists(audio_path_colab):
    print(f"Error: Audio file not found at {audio_path_colab}.")
    print("Please upload 'bZadyuv2utE_000080.flac' to your '/content/LESSISMORE_Colab/datasets/vggsound/' directory in Colab.")
    exit()

audio_input = read_audio(audio_path_colab, device)

label = 35 # Nhãn dự đoán cho audio này

audio_feature = audio_model(torch.from_numpy(audio_input).unsqueeze(0).to(device))
print("The output size of the audio feature is {}.".format(audio_feature.shape))

visualize_audio_spectrogram(audio_input, "original_audio_spectrogram.png", output_audio_directory, channel = 0)


def Partition_by_patch(image, partition_size=(8, 8)):
    """
    Chia spectrogram thành các patch nhỏ.

    Args:
        image (numpy.ndarray): Spectrogram với shape [C, T, H, W] hoặc [C, H, W]
                               (sau khi read_audio, nó là numpy array).
                               Nếu là [B, C, H, W] thì lấy batch đầu tiên.
        partition_size (tuple): Kích thước lưới phân chia (ví dụ: (8, 8)).

    Returns:
        list: Danh sách các patch (numpy arrays).
    """
    if image.ndim == 4: # if input is [B, C, H, W] or [B, C, T, H, W] from read_audio
        image_to_partition = image[0] # Take the first batch
    elif image.ndim == 3: # if input is [C, H, W]
        image_to_partition = image # No batch dim
    elif image.ndim == 5: # if input is [B, C, T, H, W]
        image_to_partition = image[0] # Take the first batch
    else:
        raise ValueError(f"Unsupported image dimension for Partition_by_patch: {image.ndim}")

    if image_to_partition.ndim == 4: # Assuming it's now [C, T, H, W]
        C, T, H, W = image_to_partition.shape
    elif image_to_partition.ndim == 3: # Assuming it's now [C, H, W]
        C = image_to_partition.shape[0]
        T = 1 # Treat as single time step
        H = image_to_partition.shape[1]
        W = image_to_partition.shape[2]
    else:
        raise ValueError(f"Unsupported image_to_partition dimension: {image_to_partition.ndim}")

    # Calculate patch dimensions for H and W
    patch_height = H // partition_size[0]
    patch_width = W // partition_size[1]

    components_image_list = []
    for i in range(partition_size[0]):
        for j in range(partition_size[1]):
            # Create an empty array of the same shape as image_to_partition
            image_tmp = np.zeros_like(image_to_partition)

            # Assign the patch values
            # Handle both [C, T, H, W] and [C, H, W] cases
            if image_to_partition.ndim == 4: # [C, T, H, W]
                 image_tmp[:, :,
                           int(i * patch_height): int((i + 1) * patch_height),
                           int(j * patch_width): int((j + 1) * patch_width)] = image_to_partition[:, :,
                                                                                   int(i * patch_height): int((i + 1) * patch_height),
                                                                                   int(j * patch_width): int((j + 1) * patch_width)]
            elif image_to_partition.ndim == 3: # [C, H, W]
                image_tmp[:,
                          int(i * patch_height): int((i + 1) * patch_height),
                          int(j * patch_width): int((j + 1) * patch_width)] = image_to_partition[:,
                                                                                  int(i * patch_height): int((i + 1) * patch_height),
                                                                                  int(j * patch_width): int((j + 1) * patch_width)]

            components_image_list.append(image_tmp)
    return components_image_list

# Đảm bảo audio_input có đúng shape mà Partition_by_patch mong đợi
# read_audio trả về [num_clips, C, H, W] (nếu T=1) hoặc [num_clips, C, T, H, W]
# Partition_by_patch sẽ lấy clip đầu tiên [0]
element_sets_V = Partition_by_patch(audio_input)

# visualize_audio_spectrogram(element_sets_V[0]+element_sets_V[-4], channel = 0) # Không gọi trực tiếp imshow
# Lưu một ví dụ về patch
visualize_audio_spectrogram(element_sets_V[0] + element_sets_V[-4], "example_patch_combination.png", output_audio_directory, channel=0)


from models.submodular_audio_efficient_plus import AudioSubModularExplanationEfficientPlus

# --- Đường dẫn file semantic features: Cập nhật để phù hợp với Colab ---
semantic_feature_path = os.path.join(project_root_dir, "ckpt", "semantic_features", "vggsound_imagebind_cls.pt")
if not os.path.exists(semantic_feature_path):
    print(f"Error: Semantic features file not found at {semantic_feature_path}.")
    print("Please ensure 'vggsound_imagebind_cls.pt' is in the 'ckpt/semantic_features/' directory.")
    exit()

semantic_feature = torch.load(semantic_feature_path, map_location=device) * 0.05

def transform_audio_data(audio_numpy):
    audio = torch.from_numpy(audio_numpy)
    # Ensure audio data has a time dimension if it's missing (e.g., [C, H, W] -> [C, 1, H, W])
    if audio.ndim == 3:
        audio = audio.unsqueeze(1) # Add time dimension as 1
    return audio

explainer = AudioSubModularExplanationEfficientPlus(
        audio_model, semantic_feature, transform_audio_data, device=device,
        lambda1=0.01,
        lambda2=0.,
        lambda3=20.,
        lambda4=5.)

explainer.k = len(element_sets_V)
# Truyền batch dimension nếu model mong đợi [B, C, T, H, W]
# explainer mong đợi audio_numpy -> transform_audio_data -> torch.Tensor
# element_sets_V là list of numpy arrays, mỗi numpy array là [C, T, H, W] (hoặc [C, H, W])
# explainer(element_sets_V) sẽ xử lý việc tạo batch từ list này
submodular_image, submodular_image_set, saved_json_file = explainer(element_sets_V, id=label)

# submodular_image_set[:21].sum(0).shape # Không gọi trực tiếp shape
# saved_json_file["consistency_score"][20] # Không gọi trực tiếp để hiển thị

visualize_audio_spectrogram(submodular_image_set[:21].sum(0), "submodular_image_set_sum_21.png", output_audio_directory, channel = 0)


def visualization(image_orig_full, submodular_image_set, saved_json_file, algorithm_mode, output_dir_name, index=None):
    """
    Visualize results and save to files for audio spectrogram.
    """
    insertion_ours_images = []
    # deletion_ours_images = [] # Không dùng cho biểu đồ này

    # For insertion, the first image is an empty (black) image
    insertion_image = np.zeros_like(image_orig_full[0]) # Lấy spectrogram gốc từ batch đầu tiên và tạo mảng 0 tương ứng
    insertion_ours_images.append(insertion_image)

    # Build up the insertion sequence
    current_insertion = np.zeros_like(image_orig_full[0])
    for smdl_sub_mask in submodular_image_set:
        current_insertion = current_insertion.copy() + smdl_sub_mask
        insertion_ours_images.append(current_insertion)

    insertion_ours_images_input_results = np.array(
        [saved_json_file["baseline_score"]] + saved_json_file["consistency_score"])

    if index is None:
        ours_best_index = np.argmax(insertion_ours_images_input_results)
    else:
        ours_best_index = index

    # Calculate percentage of image revealed
    x = [(np.sum(insertion_img != 0) / np.sum(image_orig_full[0] != 0)) for insertion_img in insertion_ours_images] # Tính tỷ lệ pixel không bằng 0
    # Đảm bảo x nằm trong khoảng [0, 1] và có độ dài bằng với kết quả dự đoán
    # x = np.linspace(0, 1, len(insertion_ours_images_input_results)) # Một cách khác đơn giản hơn

    i = len(x)

    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title('Spectrogram Attribution (Channel 0)', fontsize=24) # Giảm cỡ chữ
    ax1.set_facecolor('white')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_title('Spectrogram Attribution (Channel 1)', fontsize=24) # Giảm cỡ chữ
    ax2.set_facecolor('white')

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_title('Spectrogram Attribution (Channel 2)', fontsize=24) # Giảm cỡ chữ
    ax3.set_facecolor('white')


    ax4 = fig.add_subplot(gs[:, 1])
    ax4.set_xlim((0, 1))
    ax4.set_ylim((0, 1))
    ax4.set_title(f'Insertion ({algorithm_mode})', fontsize=24) # Giảm cỡ chữ
    ax4.set_ylabel('Recognition Score', fontsize=22) # Giảm cỡ chữ
    ax4.set_xlabel('Percentage of image revealed', fontsize=22) # Giảm cỡ chữ
    ax4.tick_params(axis='both', which='major', labelsize=16) # Giảm cỡ chữ

    x_ = x[:i]
    ours_y = insertion_ours_images_input_results[:i]
    ax4.plot(x_, ours_y, color='dodgerblue', linewidth=3.5)  # draw curve
    ax4.set_facecolor('white')
    ax4.spines['bottom'].set_color('black')
    ax4.spines['bottom'].set_linewidth(2.0)
    ax4.spines['top'].set_color('none')
    ax4.spines['left'].set_color('black')
    ax4.spines['left'].set_linewidth(2.0)
    ax4.spines['right'].set_color('none')

    ax4.scatter(x_[-1], ours_y[-1], color='dodgerblue', s=54)  # Plot latest point
    ax4.fill_between(x_, ours_y, color='dodgerblue', alpha=0.1)

    ax4.axvline(x=x_[ours_best_index], color='red', linewidth=3.5)  # 绘制红色垂直线


    # --- Hiển thị spectrogram attribution ---
    # spectrograms have shape [C, T, H, W] or [C, H, W].
    # convert_mfcc expects it to be [C, T, H, W] or [C, H, W]
    # submodular_image_set[ours_best_index] has shape [C, T, H, W] or [C, H, W]
    # We pass it directly to convert_mfcc and specify channel.

    # Lấy hình ảnh từ submodular_image_set[ours_best_index]
    img_to_display_att = insertion_ours_images[ours_best_index]

    ax1.imshow(convert_mfcc(img_to_display_att, channel=0), aspect='auto', origin='lower', cmap='viridis')
    ax2.imshow(convert_mfcc(img_to_display_att, channel=1), aspect='auto', origin='lower', cmap='viridis')
    ax3.imshow(convert_mfcc(img_to_display_att, channel=2), aspect='auto', origin='lower', cmap='viridis') # Kênh 2 nếu có

    # Lưu biểu đồ tổng thể
    save_image_to_file(fig, f"{algorithm_mode}_explanation_plot_{index if index is not None else 'auto'}.png", output_dir_name)
    plt.close(fig) # Đóng figure sau khi lưu

    auc = metrics.auc(x, insertion_ours_images_input_results)

    print("For {} mode (index={}):".format(algorithm_mode, index if index is not None else 'auto'))
    print("  Highest confidence: {:.4f}\n  Final confidence: {:.4f}\n  Insertion AUC: {:.4f}".format(
        insertion_ours_images_input_results.max(), insertion_ours_images_input_results[-1], auc))
    print("-" * 30)

# Chạy visualization cho các index cụ thể
print("\n--- Visualizing AudioBind results ---")
visualization(audio_input, submodular_image_set, saved_json_file, algorithm_mode="audiobind", output_dir_name=output_audio_directory, index=21) # Sử dụng index = 21 như trong code gốc
visualization(audio_input, submodular_image_set, saved_json_file, algorithm_mode="audiobind", output_dir_name=output_audio_directory, index=30)
visualization(audio_input, submodular_image_set, saved_json_file, algorithm_mode="audiobind", output_dir_name=output_audio_directory) # Auto best index

print(f"\nAll AudioBind visualization outputs saved as PNG files in: {output_audio_directory}")
print(f"Check the 'Files' section (folder icon on the left) to download them.")