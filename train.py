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
import torch.utils.data as data

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
criterion = MultiBoxLoss(num_classes, 0.35, 7)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)


def train():
    net.train()
    print("Loading train dataset...")

    train_dataset = dataset_gen.FaceDataset(cfg['dataset_path'], "train", preproc(img_dim, rgb_mean))
    trainloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset_gen.detection_collate)
    print("Loading validation dataset...")
    val_dataset = dataset_gen.FaceDataset(cfg['dataset_path'], "val", 500)
    valloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=dataset_gen.detection_collate)

    start_epoch = 1
    if cfg['resume_training'] is not None:
        start_epoch = cfg['resume_training'][1]

    save_interaval = 1
    epoch_size = len(train_dataset)
    loc_loss = list()
    conf_loss = list()
    total_loss = list()

    print("Staring training...")
    for epoch in range(start_epoch, num_epochs + 1):
        for batch_ind, [images, targets] in enumerate(trainloader, 1):
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]
            load_t0 = time.time()

            # Adjusting learning rate with the time
            adjust_learning_rate(optimizer, epoch, gamma, learning_rate)

            # Forward
            out = net(images)
            loss_l, loss_c = criterion(out, priors, targets)

            # Backward
            loss = cfg['loc_weight'] * loss_l + loss_c  # Total loss with alpha
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Saving losses
            loc_loss.append(loss_l.item())
            conf_loss.append(loss_c.item())
            total_loss.append(loss.item())

            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = batch_time * (len(trainloader) - batch_ind) * ((num_epochs - epoch) * BATCH_SIZE)

            # Printing results
            print(
                'Epoch:{}/{} || Images: {}/{} || L: {:.4f} C: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.format(
                    epoch, num_epochs, batch_ind * BATCH_SIZE, epoch_size, loss_l.item(),
                    loss_c.item(), learning_rate, batch_time, str(datetime.timedelta(seconds=eta))))

        # Saving weights and losses
        if epoch % save_interaval == 0:
            torch.save(net.state_dict(), cfg['saving_path'] + 'FaceDetection_epoch_' + str(epoch) + '.pth')
        with open(cfg['saving_path'] + "/stats/Epoch_" + str(epoch) + "_conf_loss.txt", 'w+') as f:
            f.write(str(conf_loss))
        with open(cfg['saving_path'] + "/stats/Epoch_" + str(epoch) + "_loc_loss.txt", 'w+') as f:
            f.write(str(loc_loss))
        with open(cfg['saving_path'] + "/stats/Epoch_" + str(epoch) + "_total_loss.txt", 'w+') as f:
            f.write(str(total_loss))

        # Getting validation results
        print("Starting validation check...")
        val_c_loss = list()
        val_l_loss = list()
        val_total_loss = list()
        with torch.no_grad():
            for val_ind, [images, v_targets] in enumerate(valloader, 1):
                v_images = images.to(device)
                v_targets = [anno.to(device) for anno in v_targets]
                v_out = net(v_images)
                val_loss_l, val_loss_c = criterion(v_out, priors, v_targets)
                val_loss = cfg['loc_weight'] * val_loss_l + val_loss_c
                print("Batch : {}/{} || L : {:.4f} C: {:.4f} T: {:.4f}".format(
                    val_ind, len(valloader), val_loss_l.item(), val_loss_c.item(), val_loss.item()))
                val_c_loss.append(val_loss_c.item())
                val_loss_l.append(val_loss_l.item())
                val_total_loss.append(val_loss.item())

            with open(cfg['saving_path'] + "/stats/Epoch_" + str(epoch) + "_val_conf_loss.txt", 'w+') as f:
                f.write(str(val_c_loss))
            with open(cfg['saving_path'] + "/stats/Epoch_" + str(epoch) + "_val_loc_loss.txt", 'w+') as f:
                f.write(str(val_l_loss))
            with open(cfg['saving_path'] + "/stats/Epoch_" + str(epoch) + "_val_total_loss.txt", 'w+') as f:
                f.write(str(val_total_loss))



        # Printing graphs TODO: (classif loss for train and val, detection loss for train and val, accuracy for val)


def adjust_learning_rate(optimizer, epoch, gamma, init_lr):
    """Sets the learning rate to the initial LR every 10 epochs"""
    lr = init_lr * (gamma ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

train()
