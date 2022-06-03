import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from saliency import greydanus
from scipy.stats import entropy
from mnist import CNN

def produce_saliency_images(cnn, model_name):
    train_data = datasets.MNIST(
        root="data",
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())

    # print(train_data)

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_data, batch_size=100, shuffle=True, num_workers=1
        ),
        "test": torch.utils.data.DataLoader(
            test_data, batch_size=100, shuffle=False, num_workers=1
        ),
    }

    display_factor = 4

    # choose a model
    # models = torch.load("results_05182022.pth")
    # cnn = models['train_orig_100'][0]
    # print(type(cnn))
    # cnn = torch.load("CLUSTER/MNIST/models/cnn_mnist.pth")

    cnn.eval()
    with torch.no_grad():
        for images, labels in loaders["test"]:
            saliency_scores = greydanus(cnn, images)
            print(saliency_scores.shape)

            for i, img_ in enumerate(images):
                print(saliency_scores[i])
                m = cm.ScalarMappable(
                    cmap="jet"
                )  # This line needs to be inside loop, otherwise color scale will be different
                saliency_img = np.ascontiguousarray(m.to_rgba(saliency_scores[i])[:, :, :3])
                saliency_img = cv2.resize(
                    saliency_img,
                    (
                        saliency_img.shape[0] * display_factor,
                        saliency_img.shape[1] * display_factor,
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )
                saliency_img = cv2.cvtColor(
                    saliency_img.astype("float32"), cv2.COLOR_BGR2RGB
                )

                # save saliency image only
                # cv2.imshow("", saliency_img)
                file_name = "saliency_scores/" + model_name + "/" + model_name + "_" + str(i) + '.png'
                img_tmp = cv2.convertScaleAbs(saliency_img, alpha=(255.0))
                cv2.imwrite(file_name, img_tmp)

                img = np.squeeze(img_.cpu().numpy())
                img = cv2.resize(
                    img,
                    (img.shape[0] * display_factor, img.shape[1] * display_factor),
                    interpolation=cv2.INTER_CUBIC,
                )
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.addWeighted(saliency_img, 0.7, img, 1 - 0.7, 0, img)

                # show image
                # cv2.imshow("", img)

                # save cureent image
                # file_name = "saliency_images/" + model_name + "/" + model_name + "_" + str(i) + '.png'
                # img = cv2.convertScaleAbs(img, alpha=(255.0))
                # cv2.imwrite(file_name, img)

                # cv2.waitKey()

                # control how many images we produce
                if i >= 99:
                    break

            # only need data from the first batch
            break

BG = 1.0/(84*84)
def preprocess(smap):
    smap = smap.flatten()
    smap = smap / np.sum(smap)
    return smap
def preprocess2(smap):
    smap = smap.flatten()
    smap -= BG
    smap = np.clip(smap, a_min=0, a_max=None)
    smap = smap / np.sum(smap)
    return smap

def computeCC(saliency_map, gt_saliency_map):
    saliency_map = preprocess2(saliency_map)
    gt_saliency_map = preprocess2(gt_saliency_map)
    score = np.corrcoef([gt_saliency_map, saliency_map])[0][1]
    return score

def computeKL(saliency_map, gt_saliency_map):
    epsilon = 2.2204e-16 #MIT benchmark
    saliency_map = preprocess2(saliency_map)
    saliency_map = np.clip(saliency_map.flatten(), a_min=epsilon, a_max=None)
    saliency_map = saliency_map / np.sum(saliency_map)
    gt_saliency_map = preprocess2(gt_saliency_map)

    return entropy(gt_saliency_map, saliency_map)

if __name__ == "__main__":

    # # ruh code to produce images
    # models = torch.load("results_05182022.pth")
    # for model_name in models:
    #     cnn = models[model_name][0]
    #     produce_saliency_images(cnn, model_name)
    # # main()

    # getting saliency scores
    map_orig = cv2.imread('saliency_images_overlap/train_orig_100/train_orig_100_0.png')
    map_orig = map_orig.astype(np.float64)
    map_aug_exp = cv2.imread('saliency_images_overlap/train_data_blur/train_data_blur_0.png')
    map_aug_exp = map_aug_exp.astype(np.float64)
    scoreKL = computeKL(map_orig, map_aug_exp)
    print(scoreKL)
