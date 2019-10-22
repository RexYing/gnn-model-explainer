import pickle

import torch
import cv2
import numpy as np

from explainer import explain

use_cuda = torch.cuda.is_available()


def load_model(path):
    model = torch.load(path)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    return model


def load_cg(path):
    cg = pickle.load(open(path))
    return cg


def save(mask_cg):
    mask = mask_cg.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask

    cv2.imwrite("mask.png", np.uint8(255 * mask))


#####
#
# 1) Load trained GNN model
# 2) Load a query computation graph
#
#####
MODEL_PATH = "gcn-vanilla.pt"
CG_PATH = "1.pt"
model = load_model(MODEL_PATH)
original_cg = load_cg(CG_PATH)


#####
#
# Set parameters of explainer
#
#####
tv_beta = 3
learning_rate = 0.1
max_iterations = 500
l1_coeff = 0.01
tv_coeff = 0.2


# Initialize cg mask
blurred_cg1 = cv2.GaussianBlur(original_cg, (11, 11), 5)
blurred_cg2 = np.float32(cv2.medianBlur(original_cg, 11)) / 255
mask_init = np.ones((28, 28), dtype=np.float32)

# Convert to torch variables
cg = explain.preprocess_cg(original_cg)
blurred_cg = explain.preprocess_cg(blurred_cg2)
mask = explain.numpy_to_torch(mask_init)

if use_cuda:
    upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
else:
    upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224))
optimizer = torch.optim.Adam([mask], lr=learning_rate)

target = torch.nn.Softmax()(model(cg))
category = np.argmax(target.cpu().data.numpy())
print("Category with highest probability", category)
print("Optimizing.. ")

for i in range(max_iterations):
    upsampled_mask = upsample(mask)

    # Use the mask to perturb the input computation graph
    perturbed_input = cg.mul(upsampled_mask) + blurred_cg.mul(1 - upsampled_mask)

    noise = np.zeros((224, 224, 3), dtype=np.float32)
    cv2.randn(noise, 0, 0.2)
    noise = explain.numpy_to_torch(noise)
    perturbed_input = perturbed_input + noise

    outputs = torch.nn.Softmax()(model(perturbed_input))
    loss = (
        l1_coeff * torch.mean(torch.abs(1 - mask))
        + tv_coeff * explain.tv_norm(mask, tv_beta)
        + outputs[0, category]
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Optional: clamping seems to give better results
    mask.data.clamp_(0, 1)

upsampled_mask = upsample(mask)
save(upsampled_mask)
