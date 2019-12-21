import torch
import cv2
import numpy as np
from utils.config import cfg
from ssd import FaceDetectionSSD
from utils.MultiBoxLoss import MultiBoxLoss
from utils.data_augment import preproc
from utils.priors import PriorBox
import torch.backends.cudnn as cudnn
import dataset_gen
import torch.utils.data as data


class PlusDataset(data.Dataset):
    """Face Detection Dataset with more than one class Object
        :param:
        anno_p: annotations path
        max_images: maximum images in dataset
        prepoc: prepocessing algorithm for each photo
        """
    def __init__(self, anno_p, max_images=None, preproc=None):
        self.preproc = preproc
        self.max_images = max_images

        anno_path = anno_p
        self.ids = self.parse_boxes(anno_path)  # actual iterable object

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread(img_id[0], cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        target = self.target_transform(img_id[1])

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

    def __len__(self):
        return len(self.ids)

    def parse_boxes(self, annotation_path):
        """
        :param
            annotation_path: path where all the boxes written
            max_images: maximum images in dataset
        :return: a tuple of (img_path, list(img_boxes, 1)) : here 1 stands for face class labeled with '1'
        """
        imgs = list()
        boxes = list()
        with open(annotation_path, 'r') as file:
            lines = file.read().splitlines()
            count = len(lines)
            i = 0
            while i < count:
                if lines[i].isdigit():
                    if 1 <= int(lines[i]):
                        box_vector = list()
                        imgs.append(lines[i - 1])
                        for j in range(i + 1, i + int(lines[i]) + 1):
                            cur_box = lines[j].split()
                            cur_box = [int(x) for x in cur_box]

                            box_vector.append(cur_box)

                        boxes.append(box_vector)

                    i += int(lines[i]) + 1
                else:
                    i += 1

        return tuple(zip(imgs, boxes))

    def target_transform(self, target):
        """
        Transforms a list(img_box, 1) annotation into a Tensor of bbox coords and label index
        Initilized with a dictionary lookup of classnames to indexes
        :returns
            tensor associated with image and target [faces_number, 5] where 5 stands for [xmin, ymin, xmax, ymax, class_num]
        images are 1024x*
        """
        res = np.empty((0, 5))
        for box in target:
            res = np.vstack((res, box))

        return res


full_anno = "C:/Maxim/Repositories/AI/Datasets/More classes/mixedT.txt"
saving_path = "C:/Maxim/Repositories/AI/Models/PlusHand1"
stats_path = "C:/Maxim/Repositories/AI/utils/stats/PlusHand1"


def train():
    #  Getting all the settings
    img_dim = cfg['img_dim']
    rgb_mean = (104, 117, 123)  # BGR order
    num_classes = cfg['num_classes']
    BATCH_SIZE = 7
    momentum = cfg['momentum']
    weight_decay = cfg['weight_decay']
    learning_rate = cfg['learning_rate']
    num_epochs = cfg['num_epochs']
    gpu_train = cfg['gpu_train']

    #  Setting net itself and loss storing
    net = FaceDetectionSSD('train', img_dim, num_classes)
    print("Printing net...")
    print(net)
    loc_loss = list()
    conf_loss = list()
    total_loss = list()
    val_c_loss = list()
    val_l_loss = list()
    val_total_loss = list()

    print('Loading resume network...')
    state_dict = torch.load(cfg['resume_training'][0])
    net.load_state_dict(state_dict)

    # Setting net config
    cudnn.benchmark = True  # Could probably improve computation speed
    device = torch.device("cuda:0" if gpu_train else "cpu")
    net = net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum,
                                weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, 7)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    net.train()
    print("Loading train dataset...")

    hand_dataset = PlusDataset(full_anno, preproc=preproc(img_dim, rgb_mean))
    handloader = data.DataLoader(hand_dataset, batch_size=BATCH_SIZE, num_workers=3, shuffle=True,
                                 collate_fn=dataset_gen.detection_collate)

    print("Loadion validation dataset...")
    val_dataset = dataset_gen.FaceDataset(cfg['dataset_path'], "val", max_images=700,
                                          preproc=preproc(img_dim, rgb_mean))
    valloader = data.DataLoader(val_dataset, batch_size=20, num_workers=3,
                                collate_fn=dataset_gen.detection_collate)

    start_epoch = 1
    if cfg['resume_training'] is not None:
        start_epoch = cfg['resume_training'][1]

    save_interaval = 1
    epoch_size = len(hand_dataset)
    print("Staring training...")
    for epoch in range(start_epoch, num_epochs + 1):
        for batch_ind, [images, targets] in enumerate(handloader, 1):
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]

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

            # Printing results
            print(
                'Epoch:{}/{} || Images: {}/{} || L: {:.4f} C: {:.4f} T: {:.4f}|| LR: {:.8f}'.format(
                    epoch, num_epochs, batch_ind * BATCH_SIZE, epoch_size, loss_l.item(),
                    loss_c.item(), loss.item(), learning_rate))

        # Saving weights and losses
        if epoch % save_interaval == 0:
            torch.save(net.state_dict(), saving_path + '/Hand_epoch_' + str(epoch) + '.pth')
        with open(stats_path + "/HandEpoch_" + str(epoch) + "_conf_loss.txt", 'w+') as f:
            f.write(str(conf_loss))
        with open(stats_path + "/HandEpoch_" + str(epoch) + "_loc_loss.txt", 'w+') as f:
            f.write(str(loc_loss))
        with open(stats_path + "/HandEpoch_" + str(epoch) + "_total_loss.txt", 'w+') as f:
            f.write(str(total_loss))

            # Getting validation results
            print("Starting validation check...")
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
                    val_l_loss.append(val_loss_l.item())
                    val_total_loss.append(val_loss.item())

                with open(stats_path + "/HandEpoch_" + str(epoch) + "_val_conf_loss.txt", 'w+') as f:
                    f.write(str(val_c_loss))
                with open(stats_path + "/HandEpoch_" + str(epoch) + "_val_loc_loss.txt", 'w+') as f:
                    f.write(str(val_l_loss))
                with open(stats_path + "/HandEpoch_" + str(epoch) + "_val_total_loss.txt", 'w+') as f:
                    f.write(str(val_total_loss))

            c = stats_path + "/HandEpoch_" + str(epoch) + "_conf_loss.txt"
            l = stats_path + "/HandEpoch_" + str(epoch) + "_loc_loss.txt"
            save_graph(c, l, epoch, 'green', 'blue', "Train loss")

            c = stats_path + "/HandEpoch_" + str(epoch) + "_val_loc_loss.txt"
            l = stats_path + "/HandEpoch_" + str(epoch) + "_val_conf_loss.txt"
            save_graph(c, l, epoch, 'green', 'blue', "Validation loss")


def save_graph(conf, loc, epoch, conf_color, loc_color, label):
    from matplotlib import pyplot as plt
    with open(conf, 'r') as f:
        y_data = str(f.readline())
        y_data = y_data[1:len(y_data) - 1].split(',')
        y_data = [float(y) for y in y_data]
    x_data = [int(x) for x in range(1, len(y_data) + 1)]
    with open(loc, 'r') as f:
        ly_data = str(f.readline())
        ly_data = ly_data[1:len(ly_data) - 1].split(',')
        ly_data = [float(y) for y in ly_data]

    fig, ax = plt.subplots()
    ax.set_title(label)
    ax.plot(x_data, y_data, label="Confidence loss", color=conf_color)
    ax.plot(x_data, ly_data, label="Location loss", color=loc_color)
    ax.set_xlabel("Batches")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.savefig(stats_path + "/HandEpoch_" + str(epoch) + "_" + label + "_graph.png")


if __name__ == '__main__':
    train()
