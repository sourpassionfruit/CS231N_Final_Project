import copy
import math

import numpy as np
import torch
import torchvision
from scipy.ndimage.filters import gaussian_filter


def get_mask(center, size, r):
    # TODO: Optimize this.
    y, x = np.ogrid[-center[0] : size[0] - center[0], -center[1] : size[1] - center[1]]
    keep = x * x + y * y <= 1
    mask = np.zeros(size)
    mask[keep] = 1
    mask = gaussian_filter(mask, sigma=r)
    return torch.Tensor(mask / mask.max())


def greydanus(network, images, kernel_size=5):
    """
    Generate saliency scores for a set of images based on a given neural network

    Input:
    network: a torch.Module to calculate saliency using.
    images: a torch.Tensor of shape (N, C, H, W) to calculate saliency scores for.
    kernel_size: kernel size for the Gaussian filter.

    Output:
    saliency_scores: attention maps of the network for each of the input images.
    """

    N, C, H, W = images.size()
    N = len(images)

    horizontal_repetitions = math.ceil(W / kernel_size)
    vertical_repetitions = math.ceil(H / kernel_size)

    original = network(images)

    saliency_scores = torch.zeros((N, vertical_repetitions, horizontal_repetitions), device=images.device)

    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    gauss_transform = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    blurred_images = gauss_transform(images)

    # print("Computing Greydanus saliency maps")
    # print("".join(["_"] * (vertical_repetitions)))

    # Go through each positioning of the mask.
    for i in range(vertical_repetitions):
        for j in range(horizontal_repetitions):
            # Get the mask for this position.
            mask = get_mask(center=[i * kernel_size, j * kernel_size], size=[H, W], r=kernel_size)
            mask = mask.reshape(1, 1, H, W).to(images.device)  # Put it in right shape for broadcast

            # Use mask to merge the images with their blurred versions
            interpolated_obs = (1 - mask) * images + mask * blurred_images

            # Pass the mixed images through the network.
            perturbed = network(interpolated_obs)

            # Compute saliency scores for the neighborhood using the Greydanus formula.
            saliency_scores[:, i, j] = torch.sum(
                (original - perturbed) ** 2 * 0.5, dim=1
            )
        # print(f".", end="", flush=True)
    # print()

    # Re-expand the saliency scores to the full image size. This will apply bilinear interpolation.
    rescale = torchvision.transforms.Resize((H, W))
    scaled_saliency_scores = rescale(saliency_scores)

    return scaled_saliency_scores


class SaliencyModulatedDropout(object):
    def __init__(self, network, num_mod_dropouts, r_backg_blur, r_saliency, p_keep=1.0):
        self.num_mod_dropouts = num_mod_dropouts
        self.r_backg_blur = r_backg_blur
        self.r_saliency = r_saliency
        self.p_keep = p_keep
        
        sigma = 0.3 * ((self.r_backg_blur - 1) * 0.5 - 1) + 0.8
        self.blur = torchvision.transforms.GaussianBlur(self.r_backg_blur, sigma)

        self.cnn = copy.deepcopy(network)  # Frozen copy of the network.
        self.cnn.eval()

    def __call__(self, img, label):
        # Get the saliency scores for each pixel of each image.
        with torch.no_grad():
            saliency_scores = greydanus(self.cnn, img, kernel_size=self.r_saliency)
                
        # Make a blurred copy of each image.
        gaussian_blurred_img = self.blur(img)
        
        # Scale the saliency scores to the 0-1 range.
        mod_sal_scores_sum = torch.sum(saliency_scores, dim=(1, 2))[:, None, None]
        float_1 = torch.tensor(1, dtype=torch.float32, device=img.device)
        mod_sal_scores_sum = torch.where(mod_sal_scores_sum == 0, float_1, mod_sal_scores_sum)
        mod_saliency_scores = saliency_scores / mod_sal_scores_sum

        # Scale the saliency scores so that the largest is p_keep.
        max_mod_saliency_scores = torch.amax(mod_saliency_scores, dim=(1, 2))
        scaling_factors = torch.where(
            max_mod_saliency_scores == 0, float_1, (self.p_keep / max_mod_saliency_scores))
        scaling_factors = scaling_factors[:, None, None]
        mod_saliency_scores = mod_saliency_scores * scaling_factors
        
        # Apply the modulated dropout.
        mod_imgs = []
        mod_labels = [label] * self.num_mod_dropouts
        for i in range(self.num_mod_dropouts):
            # Get mask for which pixels need to be blurred vs not
            mod_score_truth_vals = (
                torch.rand(mod_saliency_scores.shape, device=img.device) < mod_saliency_scores)
            # Combine the blurred and unblurred images using the pixel mask.
            aug_img = torch.where(mod_score_truth_vals[:, None, :, :], img, gaussian_blurred_img)
            mod_imgs.append(aug_img)
        mod_imgs = torch.cat(mod_imgs, dim=0)
        mod_labels = torch.cat(mod_labels, dim=0)

        return mod_imgs, mod_labels