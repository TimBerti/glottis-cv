import torch
from torchvision.transforms import v2

import matplotlib.pyplot as plt
from skimage import segmentation, io
from skimage.segmentation import mark_boundaries
from lime import lime_image
from joblib import Parallel, delayed

import os


root = 'data/images'

preprocess = v2.Compose([
    v2.Resize((256, 256)),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = torch.load('models/ResNet18_pretrained-epoch8-accuracy0.9829.pt')
model.eval()
device = next(model.parameters()).device.type

def batch_predict(images):
    batch = torch.stack(tuple(preprocess(i) for i in images), dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def create_lime_image(root, cls, filename):
    image = io.imread(f'{root}/train/{cls}/{filename}')

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, batch_predict, 
                                            top_labels=1, hide_color=0, num_samples=1000, 
                                            segmentation_fn=lambda image: segmentation.slic(image, 
                                                                                            n_segments=50, compactness=15, 
                                                                                            max_size_factor=2, min_size_factor=0.5
                                                                                            )
                                            )
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=3, hide_rest=False)
    img_boundry = mark_boundaries(temp/255.0, mask, outline_color=(0, 1, 1))
    plt.imsave(f'{root}/lime/{cls}/{filename}', img_boundry)

if __name__ == '__main__':
    for cls in os.listdir(f'{root}/train'):
        print('Creating lime images for class', cls)
        Parallel(n_jobs=4)(delayed(create_lime_image)(root, cls, filename) for filename in os.listdir(f'{root}/train/{cls}')[::10])