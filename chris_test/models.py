import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-1, max_det=1):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    result = list()

    out = torch.nn.functional.max_pool2d(heatmap[None,None], kernel_size=max_pool_ks, padding=max_pool_ks//2, stride=1).view(-1)

    flat_hm = heatmap.view(-1)

    # get positions where peaks were found
    eq = torch.eq(flat_hm, out)
    not_eq = (~eq).float()
    eq = eq.float()

    max_positions = torch.zeros(flat_hm.shape)
    max_positions = eq*flat_hm - 255 * not_eq

    # get top max_det values with their flattened index
    if max_det < max_positions.shape[0]:
        values, indeces = torch.topk(max_positions, max_det)
    else:
        values, indeces = torch.topk(max_positions, max_positions.shape[0])

    for i in range(values.shape[0]):
        if values[i] > min_score:
            result.append((indeces[i]%heatmap.shape[1], indeces[i]//heatmap.shape[1]))

    return result


class Detector(torch.nn.Module):
    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))


    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=1, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        # z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        return self.classifier(z)

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: List of detections [(class_id, score, cx, cy), ...],
                    return no more than 100 detections per image
           Hint: Use extract_peak here
        """

        self.eval()

        result = list()

        output = self.forward(image)

        heatmap = output[0][0]

        peaks = extract_peak(heatmap)

        # if the puck was detected on screen, return True and coordinates, else return false and zeroes
        if (len(peaks) > 0):
            found_puck = True
            x = peaks[0][0]
            y = peaks[0][1]
        else:
            found_puck = False
            x = 0
            y = 0

        return found_puck, x, y

class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        self.network = torch.nn.Linear(2 * 400 * 300, 2)

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.network(x.view(x.size(0), -1))

class Controller(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(2, 2, bias=False)
    
    def forward(self, x):
        return self.network(x)

    # def act(self, aim_point):
    #     x = torch.as_tensor(aim_point.astype(np.float32))
    #     p = self.forward(x[None])[0]
    #     return pystk.Action(steer=p[0], acceleration=1)

def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('drive_data/0', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    fig, axs = subplots(3, 4)
    model = load_model()
    for i, ax in enumerate(axs.flat):
        im, puck = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        b, cx, cy = model.detect(im)
        ax.add_patch(patches.Circle((cx, cy), radius=max(2  / 2, 0.1), color='rgb'[0]))
        ax.axis('off')
    show()
