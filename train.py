import math
import time
import datetime
import torch
from utils.config import cfg
from ssd import FaceDetectionSSD
from utils.MultiBoxLoss import MultiBoxLoss
from utils.data_augment import preproc
from utils.priors import PriorBox
import torch.backends.cudnn as cudnn
import dataset_gen

#  Getting all the settings
img_dim = cfg['img_dim']
rgb_mean = (104, 117, 123)  # BGR order
num_classes = cfg['num_classes']
BATCH_SIZE = cfg['batch_size']
momentum = cfg['momentum']
weight_decay = cfg['weight_decay']
learning_rate = cfg['learning_rate']
gamma = cfg['gamma']
num_epochs = cfg['num_epochs']
gpu_train = cfg['gpu_train']

#  Setting net itself
net = FaceDetectionSSD('train', img_dim, num_classes)
print("Printing net...")
print(net)

if cfg['resume_training'] is not None:
    print('Loading resume network...')
    state_dict = torch.load(cfg['resume_training'][0])
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    # copying all the layers and theirs weights
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

# Setting net config
cudnn.benchmark = True  # Could probably improve computation speed
device = torch.device("cuda:0" if gpu_train else "cpu")
net = net.to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss()  # TODO: loss function

# TODO: Prior shiet
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)


def train():
    net.train()
    print("Loading train dataset...")
    # TODO: prepoc things
    train_dataset = dataset_gen.FaceDataset(cfg['dataset_path'], "train")
    print("Loading validation dataset...")
    val_dataset = dataset_gen.FaceDataset(cfg['dataset_path'], "val")
    # TODO: dataloader

    start_epoch = 1
    if cfg['resume_training'] is not None:
        start_epoch = cfg['resume_training'][1]

    save_interaval = 1
    epoch_size = math.ceil(len(trainloader) / BATCH_SIZE)
    for epoch in range(start_epoch, num_epochs + 1):
        for batch_ind, [images, targets] in enumerate(trainloader, 1):
            images.to(device)
            targets = [anno.to(device) for anno in targets]
            load_t0 = time.time()

            # Adjusting learning rate with the time
            adjust_learning_rate(optimizer, epoch, gamma, learning_rate)

            # Forward
            out = net(images)
            loss_l, loss_c = criterion(out, priors, targets)
            # TODO: loss stashing

            # Backward
            loss = cfg['loc_weight'] * loss_l + loss_c  # Total loss with alpha
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = batch_time * (BATCH_SIZE - batch_ind) * ((num_epochs - epoch) * BATCH_SIZE)

            # Printing iteration results
            print(
                'Epoch:{}/{} || Epochiter: {}/{} || L: {:.4f} C: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.format(
                    epoch, num_epochs, batch_ind * BATCH_SIZE, epoch_size, loss_l.item(),
                    loss_c.item(), learning_rate, batch_time, str(datetime.timedelta(seconds=eta))))

            # Saving weights
            if epoch % save_interaval == 0:
                torch.save(net.state_dict(), cfg['saving_path'] + 'FaceDetection_epoch_' + str(epoch) + '.pth')

            # Getting validation results
            # TODO: validation and accuracy

            # Printing graphs TODO: (classif loss for train and val, detection loss for train and val, accuracy for val)


def adjust_learning_rate(optimizer, epoch, gamma, init_lr):
    """Sets the learning rate to the initial LR every 10 epochs"""
    lr = init_lr * (gamma ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

train()
