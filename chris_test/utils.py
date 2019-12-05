from PIL import Image
from torch.utils.data import Dataset, DataLoader
from . import dense_transforms


class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor(), min_size=20):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*.png')):
            self.files.append(im_f.replace('_img.png', ''))
        self.transform = transform
        self.min_size = min_size

        #self.files = self.files[:100]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import numpy as np
        b = self.files[idx]
        im = Image.open(b + '_img.png')
        puck = np.load(b + '_pos_ball.npz')
        data = im, [puck['arr_0']]

        if self.transform is not None:
            data = self.transform(*data)

        return data


def load_detection_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DetectionSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = DetectionSuperTuxDataset('drive_data/0')
    import torchvision.transforms.functional as F
    from pylab import show, subplots
    import matplotlib.patches as patches
    import numpy as np

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, puck = dataset[100+i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        print(puck)
        for p in puck:
            ax.add_patch(
                patches.Rectangle((p[0] - 0.55, p[1] - 0.55), 0.05, 0.05, fc='none', ec='r', lw=2))
        ax.axis('off')
    dataset = DetectionSuperTuxDataset('drive_data/0',
                                       transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0.5),
                                                                           dense_transforms.ToTensor(),
                                                                           dense_transforms.to_heatmap]))
    fig.tight_layout()
    # fig.savefig('box.png', bbox_inches='tight', pad_inches=0, transparent=True)

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, hm = dataset[100+i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        hm = hm.numpy().transpose([1, 2, 0])
        alpha = 0.25*hm.max(axis=2) + 0.75
        r = 1 - np.maximum(hm[:, :, 0], hm[:, :, 0])
        g = 1 - np.maximum(hm[:, :, 0], hm[:, :, 0])
        b = 1 - np.maximum(hm[:, :, 0], hm[:, :, 0])
        ax.imshow(np.stack((r, g, b, alpha), axis=2), interpolation=None)
        ax.axis('off')
    fig.tight_layout()
    # fig.savefig('heat.png', bbox_inches='tight', pad_inches=0, transparent=True)

    show()
