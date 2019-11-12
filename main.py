import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import cv2
import os
# C:\Maxim\PyCharm Community Edition 2019.2\Projects\AI\Datasets\wider_face_train_bbx_gt.txt

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

    print(labels[len(labels) - 1])
    return labels, boxes


def get_train_dataset(path, face_count):  # gets dataset
    """inputs: path - dataset path
       face_count - max faces on the photo
       returns: a pair of image and its boxes"""
    images = list()
    boxes = list()

    with open(os.path.join(path, "train_boxes.txt"), 'r') as test_file:
        lines = test_file.read().splitlines()
        im_path = os.path.join(path, "train")
        count = len(lines)
        i = 0
        max_ph = 1000  # debug thing
        ph = 0  # debug thing

        while i < count and ph < max_ph:
            if lines[i].isdigit():
                if 1 <= int(lines[i]) <= face_count:
                    box_vector = list()  # faces' coords
                    image = cv2.imread(os.path.join(im_path, lines[i - 1]))
                    for j in range(i + 1, i + int(lines[i]) + 1):
                        box_vector += lines[j][1:len(lines[j]) - 1].split(',')

                    # Adjusting box coordinates
                    height, width = image.shape[:2]
                    x_coef = width / dim[0]
                    y_coef = height / dim[1]
                    box_vector = [int(x) for x in box_vector]

                    for i_ in range(len(box_vector)):
                        if i_ % 2 == 0:
                            box_vector[i_] = int(box_vector[i_] / x_coef)
                        else:
                            box_vector[i_] = int(box_vector[i_] / y_coef)
                    while len(box_vector) < 4 * face_count:
                        box_vector.append(0)

                    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    image = image.astype("float32") / 255
                    images.append(image)

                    boxes.append(box_vector)
                    ph += 1

                i += int(lines[i]) + 1
            else:
                i += 1

    return images, boxes


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


class ConvNet(nn.Module):
    # def __init__(self):
    #     super(ConvNet, self).__init__()
    #     # 1024x768
    #     self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=12, stride=1, padding=2),
    #                                 nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
    #     # 256x192
    #     self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=12, stride=1, padding=2),
    #                                 nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    #     self.drop_out = nn.Dropout()
    #     # 128x96
    #     self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=12, stride=1, padding=2),
    #                                 nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    #     # 64x48
    #     self.layer4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=12, stride=1, padding=2),
    #                                 nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    #     # 32x24
    #     self.layer5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=12, stride=1, padding=2),
    #                                 nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
    #     # 16x12
    #     self.drop_out = nn.Dropout()
    #     self.fc1 = nn.Linear(16 * 12 * 64, 10000)
    #     self.fc2 = nn.Linear(10000, max_faces * 4)
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 12)
        self.conv2 = nn.Conv2d(32, 64, 12)
        self.drop_out = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(64, 128, 12)
        self.conv4 = nn.Conv2d(128, 256, 8)
        self.conv5 = nn.Conv2d(256, 512, 4)
        self.drop_out = nn.Dropout(0.1)
        self.fc1 = nn.Linear(512 * 6 * 10, 100)
        self.fc2 = nn.Linear(100, MAX_FACES * 4)

    def forward(self, x):
        # out = self.layer1(x)
        # out = self.layer2(out)
        # out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # return out
        x = F.max_pool2d(F.relu(self.conv1(x)), 4)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# # Осуществляем оптимизацию путем стохастического градиентного спуска
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# # Создаем функцию потерь
# criterion = nn.NLLLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.MSELoss()

# total_step = len(train)
# loss_list = []
# acc_list = []
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train):
#         # Прямой запуск
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss_list.append(loss.item())
#
#         # Обратное распространение и оптимизатор
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # Отслеживание точности
#         total = labels.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         correct = (predicted == labels).sum().item()
#         acc_list.append(correct / total)
#
#         if (i + 1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
#                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
#                           (correct / total) * 100))

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

    #torch.save(model, os.path.join(MODEL_PATH, str(epoch) + "epmodel.pth"))

# torch.save(model, os.path.join(MODEL_PATH, "1fmodel.pth"))
exit()
m = FaceDetect()
#m.load_state_dict(torch.load(os.path.join(MODEL_PATH, "1fmodel.pth")))
m = torch.load(os.path.join(MODEL_PATH, "1fmodel.pth"))
m.eval()
test_im = cv2.imread(os.path.join(dataset_path, "train/23_Shoppers_Shoppers_23_661.jpg"))
test_im = cv2.resize(test_im, dim)
imm = test_im
test_im = cv2.cvtColor(test_im, cv2.COLOR_RGB2GRAY)
test_im = test_im.astype("float32") / 255
print("1 after normalizing", test_im)
print("\n")

imgg = cv2.imread(os.path.join(dataset_path, "train/23_Shoppers_Shoppers_23_661.jpg"))
imgg = cv2.resize(imgg, dim)
imgg = cv2.cvtColor(imgg, cv2.COLOR_RGB2GRAY)
imgg = imgg.astype("float32") / 255
print("2 after normalizing", imgg)

test_im = torch.tensor([test_im])
test_im = test_im.unsqueeze(1)
o = m(test_im)
with torch.no_grad():
    out = m(test_im)
    box = [int(out[0][0] * 10000), int(out[0][1] * 10000), int(out[0][2] * 10000), int(out[0][3]) * 10000]
cv2.rectangle(imm, (abs(box[0]), abs(box[1])), (abs(box[2]), abs(box[3])), (255, 255, 0), 3)
cv2.imshow("s", imm)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

"""Testing"""
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    max_loss = 15  # euristic check
    for images, boxes in test_loader:
        images = images.unsqueeze(1)
        outputs = model(images.float())

        loss = outputs - boxes
        loss_vec = list()
        for ten in loss:
            err = 0
            for elem in ten:
                err += abs(elem)
            loss_vec.append(err)

        total += boxes.size(0)
        for er in loss_vec:
            if er < max_loss:
                correct += 1

    test_loss = (total - correct) / len(test_loader)
    print('nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)n'.format(
        test_loss, correct, len(test_loader),
        100. * correct / len(test_loader)))

torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'conv_net_model.ckpt'))