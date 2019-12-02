import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torchvision
import torch.nn.functional as F
import time


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = torch.tensor(gamma)
    def forward(self, heatmap_pred, label):
        BCE_loss = F.binary_cross_entropy_with_logits(heatmap_pred, label, reduction='none').cpu()
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        #F_loss = -self.alpha * torch.pow(torch.tensor(1.) - BCE_loss, self.gamma.to(heatmap_pred.dtype))
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train')+'/{}'.format(time.strftime('%m-%d-%H-%M')), flush_secs=1)
        #valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid')+'/{}'.format(time.strftime('%m-%d-%H-%M')), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss = FocalLoss(gamma=args.gamma)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    #valid_data = load_detection_data('dense_data/valid', num_workers=4, transform=dense_transforms.Compose([dense_transforms.ToTensor(),dense_transforms.ToHeatmap()]))
    #print("type",type(train_data.))
    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        #conf = ConfusionMatrix()
        print("epoch",epoch)
        for img, label, C in train_data:
            img, label = img.to(device), label.to(device)#.long()

            heatmap_pred = model(img)
            
            
            #if train_logger is not None and global_step % 100 == 0:
                
                #heatmap_prob = torch.sigmoid(heatmap_pred)
                #image = torch.cat([heatmap_prob, label], 3).detach().cpu()
                #train_logger.add_image('Postimage', (torchvision.utils.make_grid(heatmap_prob, padding=5, pad_value=1) * 255).byte())
                
                #train_logger.add_image('OGimage', img[0].detach().cpu(), global_step)
                #train_logger.add_image('label', label[0].detach().cpu(), global_step)
                #print(img.shape, heatmap_pred.shape)
            loss_val = loss(heatmap_pred, label).mean()
            
            if train_logger is not None:
                #print("loss_val", loss_val)
                train_logger.add_scalar('loss', loss_val, global_step)
            #conf.add(heatmap_pred.argmax(1), label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
    model.eval()
    save_model(model)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-bn', type=bool, default=False)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('-g', '--gamma', type=float, default=2., help="class dependent weight for cross entropy")
    #parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.3, 0.3, 0.3, 0.0), RandomHorizontalFlip(), ToTensor(),ToHeatmap()])')
    
    args = parser.parse_args()
    train(args)
