import torch
from utils.config import cfg
from ssd import FaceDetectionSSD
from utils.MultiBoxLoss import MultiBoxLoss
from utils.data_augment import preproc
from utils.priors import PriorBox
import torch.backends.cudnn as cudnn
import dataset_gen
import torch.utils.data as data


def train():
    #  Getting all the settings
    img_dim = cfg['img_dim']
    rgb_mean = (104, 117, 123)  # BGR order
    num_classes = cfg['num_classes']
    BATCH_SIZE = cfg['batch_size']
    momentum = cfg['momentum']
    weight_decay = cfg['weight_decay']
    learning_rate = cfg['learning_rate']
    num_epochs = cfg['num_epochs']
    gpu_train = cfg['gpu_train']

    #  Setting net itself and loss storing
    net = FaceDetectionSSD('train', img_dim, num_classes)
    print("Printing net...")
    print(net)

    if cfg['resume_training'] is not None:
        print('Loading resume network...')
        print(cfg['resume_training'])
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

    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, 7)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    net.train()
    print("Loading train dataset...")

    train_dataset = dataset_gen.FaceDataset(cfg['dataset_path'], "train", preproc=preproc(img_dim, rgb_mean))
    trainloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, collate_fn=dataset_gen.detection_collate)
    print("Loading validation dataset...")
    val_dataset = dataset_gen.FaceDataset(cfg['dataset_path'], "val", max_images=3000, preproc=preproc(img_dim, rgb_mean))
    valloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=dataset_gen.detection_collate)

    start_epoch = 1
    if cfg['resume_training'] is not None:
        start_epoch = cfg['resume_training'][1]

    save_interaval = 1
    epoch_size = len(train_dataset)
    print("Staring training...")
    for epoch in range(start_epoch, num_epochs + 1):
        loc_loss = list()
        conf_loss = list()
        total_loss = list()
        val_c_loss = list()
        val_l_loss = list()
        val_total_loss = list()
        for batch_ind, [images, targets] in enumerate(trainloader, 1):
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
                'Epoch:{}/{} || Images: {}/{} || L: {:.4f} C: {:.4f} T: {:.4f} || LR: {:.8f}'.format(
                    epoch, num_epochs, batch_ind * BATCH_SIZE, epoch_size, loss_l.item(),
                    loss_c.item(), loss.item(), learning_rate))

        # Saving weights and losses
        if epoch % save_interaval == 0:
            torch.save(net.state_dict(), cfg['saving_path'] + '/FaceDetection_epoch_' + str(epoch) + '.pth')

        with open(cfg['stats_path'] + "/Epoch_" + str(epoch) + "_train_loss.txt", 'w+') as f:
            f.write(f"stats:\nc: {str(conf_loss)}\nl: {str(loc_loss)}\nt: {str(total_loss)}\n\n")

            f.write(f"c{str(epoch)} = {get_avg(conf_loss, len(conf_loss))}\n")
            f.write(f"l{str(epoch)} = {get_avg(loc_loss, len(conf_loss))}\n")
            f.write(f"t{str(epoch)} = {get_avg(total_loss, len(conf_loss))}\n")

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

            with open(cfg['stats_path'] + "/Epoch_" + str(epoch) + "_val_loss.txt", 'w+') as f:
                f.write(f"stats:\nc: {str(val_c_loss)}\nl: {str(val_l_loss)}\nt: {str(val_total_loss)}\n\n")

                f.write(f"c{str(epoch)} = {get_avg(val_c_loss, len(val_c_loss))}\n")
                f.write(f"l{str(epoch)} = {get_avg(val_l_loss, len(val_l_loss))}\n")
                f.write(f"t{str(epoch)} = {get_avg(val_total_loss, len(val_total_loss))}\n")

        save_graph(conf_loss, loc_loss, epoch, 'green', 'blue', "Train loss")

        save_graph(val_c_loss, val_l_loss, epoch, 'green', 'blue', "Validation loss")


def save_graph(conf, loc, epoch, conf_color, loc_color, label):
    from matplotlib import pyplot as plt
    x_data = range(1, len(conf) + 1)

    fig, ax = plt.subplots()
    ax.set_title(label)
    ax.plot(x_data, conf, label="Confidence loss", color=conf_color)
    ax.plot(x_data, loc, label="Location loss", color=loc_color)
    ax.set_xlabel("Batches")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.savefig(cfg['stats_path'] + "/Epoch_" + str(epoch) + "_" + label + "_graph.png")


def get_avg(lst, each):
    res = list()
    count = 0
    cur_sum = 0
    for elem in lst:
        cur_sum += elem
        count += 1
        if count % each == 0:
            res.append(cur_sum / each)
            cur_sum = 0
            count = 0
    if count != 0:
        res.append(cur_sum / count)
    return res


if __name__ == '__main__':
    train()
