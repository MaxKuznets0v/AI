import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import cv2
import os

"""Parameters"""
is_training = True
MAX_FACES = 10
learning_rate = 0.1
num_epochs = 10
batch_size = 10
dim = (288, 288)
MODEL_PATH = r"C:\Maxim\PyCharm Community Edition 2019.2\Projects\AI\Models"
dataset_path = r"C:\Maxim\PyCharm Community Edition 2019.2\Projects\AI\Datasets"


def get_gen(path, face_count, dataset):
    """inputs: path - dataset path
    face_count - max faces on the photo
    returns: a pair of image name and its boxes"""
    labels = list()
    boxes = list()

    if dataset == "train":
        file = os.path.join(path, "train_boxes.txt")
        im_path = os.path.join(path, "train")
    elif dataset == "test":
        file = os.path.join(path, "val_boxes.txt")
        im_path = os.path.join(path, "val")
    else:
        raise RuntimeError("dataset='train' or 'test'")

    with open(file, 'r') as test_file:
        lines = test_file.read().splitlines()
        count = len(lines)
        i = 0
        max_ph = 1000  # debug thing
        ph = 0  # debug thing

        while i < count and ph < max_ph:
            if lines[i].isdigit():
                if 1 <= int(lines[i]) <= face_count:
                    box_vector = list()  # faces' coords
                    labels.append(os.path.join(im_path, lines[i - 1]))
                    for j in range(i + 1, i + int(lines[i]) + 1):
                        box_vector += lines[j][1:len(lines[j]) - 1].split(',')

                    box_vector = [int(x) for x in box_vector]
                    while len(box_vector) < 4 * face_count:
                        box_vector.append(0)

                    boxes.append(box_vector)
                    ph += 1

                i += int(lines[i]) + 1
            else:
                i += 1

    return labels, boxes


class FaceDataset(torch.utils.data.Dataset):
    """Construct dataset which loads images for batch"""
    def __init__(self, labels, boxes):
        self.labels = (labels, [False for i in range(len(labels))])
        self.boxes = boxes

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index):
        image = cv2.imread(self.labels[0][index])
        # Adjusting box coordinates
        if not self.labels[1][index]:  # checks whether image was used before
            height, width = image.shape[:2]
            x_coef = width / dim[0]
            y_coef = height / dim[1]

            for i_ in range(len(self.boxes[index])):
                if i_ % 2 == 0:
                    self.boxes[index][i_] = int(self.boxes[index][i_] / x_coef)
                else:
                    self.boxes[index][i_] = int(self.boxes[index][i_] / y_coef)
            self.labels[1][index] = True

        # Turning photo into a gray
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype("float32") / 255
        return [image, torch.tensor(self.boxes[index])]


# d = torch.utils.data.TensorDataset(torch.tensor(dataset[0]), torch.tensor(dataset[1]))
train = get_gen(dataset_path, MAX_FACES, dataset="train")
test = get_gen(dataset_path, MAX_FACES, dataset="test")
train_dataset = FaceDataset(train[0], train[1])
test_dataset = FaceDataset(test[0], test[1])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class FaceDetect(torch.nn.Module):
    def __init__(self):
        super(FaceDetect, self).__init__()
        # 288x288
        self.clayer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=12, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # 140x140
        self.clayer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=8, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        # 68x68
        self.clayer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=6, stride=1, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # 33x33
        self.drop_out = nn.Dropout()
        self.l1 = nn.Linear(64 * 33 * 33, 512)
        self.l2 = nn.Linear(512, MAX_FACES * 4)

    def forward(self, x):
        out = self.clayer1(x)
        out = self.clayer2(out)
        out = self.drop_out(out)
        out = self.clayer3(out)
        out = out.reshape(out.size(0), -1)
        # out = out.view(-1, self.num_flat_features(out))
        out = self.drop_out(out)
        out = self.l1(out)
        out = self.l2(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


"""Training"""
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FaceDetect()
#model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

log_interval = 1
loss_list = list()
for epoch in range(1, num_epochs + 1):
    if not is_training:
        break
    for batch_idx, [images, targets] in enumerate(train_loader, 1):
        # Forward
        images = images.unsqueeze(1)
        # images = images.to(device)
        # targets = targets.to(device)
        net_out = model(images.float())
        loss = criterion(net_out, targets.float())
        loss_list.append(loss.data.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy check
        if batch_idx % log_interval == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                   epoch, batch_idx * len(images), len(train_loader) * batch_size,
                          100. * batch_idx / len(train_loader), loss.data.item()))

    # Epoch Checkpoints
    cur_state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict()
        }
    torch.save(cur_state, os.path.join(MODEL_PATH, str(epoch) + "epmodel.pth"))
    with open(os.path.join(dataset_path, "loss_list.txt"), 'w') as ls:
        ls.write(str(loss_list))

    torch.save(model, os.path.join(MODEL_PATH, str(epoch) + "epmodel.pth"))

torch.save(model, os.path.join(MODEL_PATH, "1fmodel.pth"))