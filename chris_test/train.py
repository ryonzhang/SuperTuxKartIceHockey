import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import time
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train{}'.format(time.strftime('%m-%d-%H-%M'))), flush_secs=5)

    """
    Your code here, modify your HW3 code
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

    bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    l1 = torch.nn.L1Loss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=True)

    transform = list()
    transform = []
    transform.append(dense_transforms.RandomHorizontalFlip())
    transform.append(dense_transforms.ToTensor())
    transform.append(dense_transforms.to_heatmap)

    transform = dense_transforms.Compose(transform)

    train_data = load_detection_data('drive_data/0',batch_size=args.bs, num_workers=0, transform=transform)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()

        losses = []

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            heatmap_pred = model(img)

            # focal loss
            peak_loss = (0.25*((1-torch.exp(-bce(heatmap_pred, label)))**2.0)*bce(heatmap_pred, label)).mean()

            loss_val = peak_loss

            if train_logger is not None and global_step % 500 == 0:
                heatmap_prob = torch.sigmoid(heatmap_pred)
                image = torch.cat([heatmap_prob, label], 3).detach().cpu()
                train_logger.add_image('image', (torchvision.utils.make_grid(image, padding=5, pad_value=1) * 255).byte(), global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)


            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            losses.append(loss_val.detach().cpu())

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)
        print("epoch " + str(epoch) + ", loss=" + str(avg_loss))
        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')

    parser.add_argument('-bs', type=int, default=32)
    parser.add_argument('-patience', type=int, default=2)
    parser.add_argument('-factor', type=float, default=0.1)
    args = parser.parse_args()
    train(args)
