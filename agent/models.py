import torch
import torch.nn.functional as F
from torchvision.transforms import functional as FT
import numpy as np

# For sort key
def heatCompare(x):
    return x[0]
def heatCompare2(x):
    return x[1]

def extract_peak(heatmap, max_pool_ks=7, min_score=0., max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    lx = heatmap.shape[1]
    ly = heatmap.shape[0]
    ind = torch.arange(heatmap.numel()).view(ly, lx)
    heatmap = heatmap[None, :, :]
    pad = max_pool_ks//2

    heatmap2 = F.pad(input=heatmap,pad=(pad,pad,pad,pad),value=-1000)
    maxmap = F.max_pool2d(heatmap2,kernel_size=max_pool_ks,stride=1)
    mask = heatmap==maxmap
    vals = heatmap[mask]
    indices = np.unravel_index(ind[mask[0]], (ly,lx))

    ret = []
    for i in range(vals.shape[0]): #linear sort through local maxima
        current = vals[i].item()
        if (current > min_score):
            ret.append([current,indices[1][i],indices[0][i]])

    ret.sort(key=heatCompare,reverse=True)
    if (len(ret)>max_det):
        ret = ret[:max_det]
    return ret

    

class Detector(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x)))))) + self.skip(x)

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 96, 128], n_output_channels=3, kernel_size=3, use_skip=True,use_bn=False):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        self.bn = use_bn
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])
        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):#forwards
            #if (self.bn):
            self.add_module('bn%d' % i, torch.nn.BatchNorm2d(c))
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:#backwards
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """

        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device) # normalize

        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            #if (self.bn):
                #z = self._modules['bn%d'%i](z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        z = self.classifier(z)
        return z
 

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: List of detections [(class_id, score, cx, cy), ...],
                    return no more than 100 detections per image
           Hint: Use extract_peak here
        """

        image= image[None]
        heatmap = self.forward(image)
        masterList = []
        c1 = extract_peak(heatmap[0][0],33)
        c2 = extract_peak(heatmap[0][1],33)
        c3 = extract_peak(heatmap[0][2],33)

        for (score, cx, cy) in c1:
            masterList.append((0,score,cx,cy))
        for (score, cx, cy) in c2:
            masterList.append((1,score,cx,cy))
        for (score, cx, cy) in c3:
            masterList.append((2,score,cx,cy))
        masterList.sort(key=heatCompare2,reverse=True)

        if (len(masterList)>100):
            return masterList[:100]
        else:
            return masterList

def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model(file_name):
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), file_name), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    fig, axs = subplots(3, 4)
    model = load_model()
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        for c, s, cx, cy in model.detect(im):
            ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
