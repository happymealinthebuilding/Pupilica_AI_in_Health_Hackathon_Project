#utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2

class GradCAMpp:
    # ... (Your GradCAMpp class code here) ...
    def __init__(self, model, target_layer_name='features'):
        self.model = model
        self.model.eval()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.activations, self.gradients = {}, {}
        target_layer = self.model._modules.get(target_layer_name)
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output):
        self.activations['activation'] = output
    def save_gradient(self, module, grad_in, grad_out):
        self.gradients['gradient'] = grad_out[0]
    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None: target_class = output.argmax().item()
        one_hot_output = torch.zeros(output.size()).to(self.device)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        gradients = self.gradients['gradient']
        activations = self.activations['activation']
        
        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
        alpha_denom[alpha_denom == 0] = 1e-7
        alpha = alpha_num.div(alpha_denom)
        
        weights = (gradients * alpha).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        cam = cam.squeeze().cpu().detach().numpy()
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class ScoreCAM:
    # ... (Your ScoreCAM class code here) ...
    def __init__(self, model, target_layer_name='features'):
        self.model = model
        self.model.eval()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.target_layer = self.model._modules.get(target_layer_name)
    def __call__(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None: target_class = output.argmax().item()
        activations = self.target_layer(input_tensor).detach()
        upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[2:])
        heatmaps = []
        for i in range(activations.shape[1]):
            mask = upsample(activations[:, i:i+1, :, :])
            masked_input = input_tensor * mask
            with torch.no_grad():
                output_masked = self.model(masked_input)
            score = output_masked.softmax(dim=1)[:, target_class].item()
            heatmaps.append(score * mask.squeeze().cpu().numpy())
        cam = np.sum(heatmaps, axis=0)
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def overlay_heatmap(image, heatmap, alpha=0.5):
    img = np.array(image.convert('RGB'))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img