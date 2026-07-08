import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]

templates = [
    "a photo of a {}", "an image of a {}", "a picture of a {}",
    "this is a photo of a {}", "a {} in the scene"
]

text_inputs = []
for t in templates:
    for c in cifar10_classes:
        text_inputs.append(t.format(c))

text_tokens = clip.tokenize(text_inputs).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Test với 1 ảnh
testset = CIFAR10(root='./data', train=False, download=True)
img, label = testset[5]   # thay số để thử ảnh khác

image_input = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T).squeeze(0)
    probs = similarity.softmax(dim=-1)

top_probs, top_labels = probs.topk(5)

print("Dự đoán:")
for i in range(5):
    print(f"{i+1}. {text_inputs[top_labels[i]]}: {top_probs[i].item():.4f}")

plt.imshow(img)
plt.title(f"True: {cifar10_classes[label]}\nPred: {text_inputs[top_labels[0]]}")
plt.axis('off')
plt.show()