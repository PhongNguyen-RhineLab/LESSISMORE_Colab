import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir("../")

import cv2
import numpy as np
import  matplotlib.pyplot as plt

plt.style.use('seaborn')

from utils import *
import matplotlib

from models.submodular_cub_v2 import CubSubModularExplanationV2
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap, SobolAttributionMethod, HsicAttributionMethod)

matplotlib.get_cachedir()
plt.rc('font', family="Times New Roman")

from sklearn import metrics
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def imshow(img):
    """
    Visualizing images inside jupyter notebook
    """
    plt.axis('off')
    if len(img.shape)==3:
        img = img[:,:,::-1] 	# transform image to rgb
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    plt.show()
def SubRegionDivision(image, mode="slico"):
    element_sets_V = []
    if mode == "slico":
        slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=30, ruler = 20.0)
        slic.iterate(20)     # The number of iterations, the larger the better the effect
        label_slic = slic.getLabels()        # Get superpixel label
        number_slic = slic.getNumberOfSuperpixels()  # Get the number of superpixels

        for i in range(number_slic):
            img_copp = image.copy()
            img_copp = img_copp * (label_slic == i)[:,:, np.newaxis]
            element_sets_V.append(img_copp)
    elif mode == "seeds":
        seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
        seeds.iterate(image,10)  # The input image size must be the same as the initialization shape and the number of iterations is 10
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()

        for i in range(number_seeds):
            img_copp = image.copy()
            img_copp = img_copp * (label_seeds == i)[:,:, np.newaxis]
            element_sets_V.append(img_copp)
    return element_sets_V
smdl = CubSubModularExplanationV2(
        cfg_path="configs/cub/submodular_cfg_cub_tf-resnet-v2.json",
        k=50,
        lambda1=1,
        lambda2=1,
        lambda3=1,
        lambda4=1)
image_path = "examples/Crested_Auklet_0059_794929.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
def visualization(image, submodular_image_set, saved_json_file):
    insertion_ours_images = []
    deletion_ours_images = []

    insertion_image = submodular_image_set[0]
    insertion_ours_images.append(insertion_image)
    deletion_ours_images.append(image - insertion_image)
    for smdl_sub_mask in submodular_image_set[1:]:
        insertion_image = insertion_image.copy() + smdl_sub_mask
        insertion_ours_images.append(insertion_image)
        deletion_ours_images.append(image - insertion_image)

    insertion_ours_images_input_results = np.array(saved_json_file["consistency_score"])

    ours_best_index = np.argmax(insertion_ours_images_input_results)
    x = [(insertion_ours_image.sum(-1)!=0).sum() / (image.shape[0] * image.shape[1]) for insertion_ours_image in insertion_ours_images]
    i = len(x)

    fig, [ax2, ax3] = plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 1.5]}, figsize=(24,8))
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_title('Ours', fontsize=54)
    ax2.set_facecolor('white')

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.title('Insertion', fontsize=54)
    plt.ylabel('Recognition Score', fontsize=44)
    plt.xlabel('Percentage of image revealed', fontsize=44)

    x_ = x[:i]
    ours_y = insertion_ours_images_input_results[:i]
    ax3.plot(x_, ours_y, color='dodgerblue', linewidth=3.5)  # draw curve

    # plt.legend(["Ours"], fontsize=40, loc="upper left")
    plt.scatter(x_[-1], ours_y[-1], color='dodgerblue', s=54)  # Plot latest point

    kernel = np.ones((3, 3), dtype=np.uint8)
    plt.plot([x_[ours_best_index], x_[ours_best_index]], [0, 1], color='red', linewidth=3.5)  # 绘制红色曲线

    # Ours
    mask = (image - insertion_ours_images[ours_best_index]).mean(-1)
    mask[mask>0] = 1

    dilate = cv2.dilate(mask, kernel, 3)
    # erosion = cv2.erode(dilate, kernel, iterations=3)
    # dilate = cv2.dilate(erosion, kernel, 2)
    edge = dilate - mask
    # erosion = cv2.erode(dilate, kernel, iterations=1)

    image_debug = image.copy()

    image_debug[mask>0] = image_debug[mask>0] * 0.5
    image_debug[edge>0] = np.array([0,0,255])
    ax2.imshow(image_debug[...,::-1])

    auc = metrics.auc(x, insertion_ours_images_input_results)

    print("Highest confidence: {}\nfinal confidence: {}\nInsertion AUC: {}".format(insertion_ours_images_input_results.max(), insertion_ours_images_input_results[-1], auc))

element_sets_V = SubRegionDivision(image, mode="slico")
smdl.k = len(element_sets_V)
submodular_image, submodular_image_set, saved_json_file = smdl(element_sets_V)
visualization(image, submodular_image_set, saved_json_file)

element_sets_V = SubRegionDivision(image, mode="seeds")
smdl.k = len(element_sets_V)
submodular_image, submodular_image_set, saved_json_file = smdl(element_sets_V)
visualization(image, submodular_image_set, saved_json_file)