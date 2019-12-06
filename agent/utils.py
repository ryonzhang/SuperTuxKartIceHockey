from PIL import Image
from torch.utils.data import Dataset, DataLoader
from . import dense_transforms

class WorldTools:
    SCREEN_WIDTH = 400
    SCREEN_HEIGHT = 300

    def ray_trace_to_ground(numpy_vec4, view):
        assert view.shape == (4, 4)
        assert numpy_vec4.shape[0] == 4
        assert abs(numpy_vec4[3]) < 1e-7

        camera_location = np.array(list(np.linalg.pinv(view)[:3,3]) + [0])
        ground_y_coord = 0.3698124289512634,
        multiplier = (ground_y_coord - camera_location[1]) / numpy_vec4[1]
        result = camera_location + multiplier * numpy_vec4
        return result

    def view_to_global(numpy_vec4, view):
        assert numpy_vec4.shape[0] == 4
        view_inverse = np.linalg.pinv(view)
        return view_inverse @ numpy_vec4

    def homogeneous_to_euclidean(numpy_vec4):
        assert numpy_vec4.shape[0] == 4
        # dont want numerical errors to magnify... prolly dont need this check but whatever
        if abs(numpy_vec4[3]) <= 1e-4:
            result[3] = 0
            return numpy_vec4
        result = numpy_vec4 / numpy_vec4[3]
        result[3] = 0
        return result

    def screen_to_view(aim_point_image, proj, view):
        x, y, W, H = *aim_point_image, SCREEN_WIDTH, SCREEN_HEIGHT
        projection_inverse = np.linalg.pinv(proj)
        ndc_coords = np.array([float(x) / (W / 2) - 1, 1 - float(y) / (H / 2), 0, 1])
        return projection_inverse @ ndc_coords

    def screen_puck_to_world_puck(screen_puck_coords, proj, view):
        """
        Call this function with
        @param screen_puck_coords: [screen_puck.x, scren_puck.y]
        @param proj: camera.projection.T
        @param view: camera.view.T
        """
        view_puck_coords = homogeneous_to_euclidean(screen_to_view(screen_puck_coords, proj, view))
        view_puck_dir = view_puck_coords / np.linalg.norm(view_puck_coords)
        global_puck_dir = view_to_global(view_puck_dir, view)
        global_puck_dir = global_puck_dir / np.linalg.norm(global_puck_dir)
        return ray_trace_


class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor(), min_size=20):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*.png')):
            self.files.append(im_f.replace('.png', '').replace('image_', ''))
        self.transform = transform
        self.min_size = min_size

    def _filter(self, boxes):
        return [b for b in boxes if abs(b[3] - b[1]) * abs(b[2] - b[0]) >= self.min_size]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx): # We can change this method use different types of our collected data
        import numpy as np
        b = self.files[idx]
        im = Image.open('image_'+ b + '.png')
        label = np.load('label_'+ b + '.npy')
        nfo = np.load(b + '_boxes.npz')
        data = im, self._filter(nfo['karts']), self._filter(nfo['bombs']), self._filter(nfo['pickup'])
        if self.transform is not None:
            data = self.transform(*data)
        return data


def load_detection_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DetectionSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = DetectionSuperTuxDataset('dense_data/train')
    import torchvision.transforms.functional as F
    from pylab import show, subplots
    import matplotlib.patches as patches
    import numpy as np

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[100+i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='r', lw=2))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='g', lw=2))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='b', lw=2))
        ax.axis('off')
    dataset = DetectionSuperTuxDataset('dense_data/train',
                                       transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),
                                                                           dense_transforms.ToTensor(),
                                                                           dense_transforms.to_heatmap]))
    fig.tight_layout()
    # fig.savefig('box.png', bbox_inches='tight', pad_inches=0, transparent=True)

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, hm, size = dataset[100+i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        hm = hm.numpy().transpose([1, 2, 0])
        alpha = 0.25*hm.max(axis=2) + 0.75
        r = 1 - np.maximum(hm[:, :, 1], hm[:, :, 2])
        g = 1 - np.maximum(hm[:, :, 0], hm[:, :, 2])
        b = 1 - np.maximum(hm[:, :, 0], hm[:, :, 1])
        ax.imshow(np.stack((r, g, b, alpha), axis=2), interpolation=None)
        ax.axis('off')
    fig.tight_layout()
    # fig.savefig('heat.png', bbox_inches='tight', pad_inches=0, transparent=True)

    show()
