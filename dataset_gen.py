import cv2
import os
import torch
import torch.utils.data as data
from utils.config import cfg
import numpy as np

WIDER_CLASSES = ( '__background__', 'face')


# def transform(im_path, resize=True):
#     image = cv2.imread(im_path)
#     if resize:
#         image = cv2.resize(image, cfg.dim)
#     image = image.transpose(2, 0, 1)
#     image = torch.from_numpy(image).float()
#     image = Variable(image.cuda())
#     return image
#
#
# def get_gen(path, face_count, dataset):
#     """inputs: path - dataset path
#     face_count - max faces on the photo
#     returns: a pair of image name and its boxes"""
#     labels = list()
#     boxes = list()
#
#     if dataset == "train":
#         file = os.path.join(path, "train_boxes.txt")
#         im_path = os.path.join(path, "train")
#     elif dataset == "test":
#         file = os.path.join(path, "val_boxes.txt")
#         im_path = os.path.join(path, "val")
#     else:
#         raise RuntimeError("dataset='train' or 'test'")
#
#     with open(file, 'r') as test_file:
#         lines = test_file.read().splitlines()
#         count = len(lines)
#         i = 0
#         max_ph = 1000000000000000000000  # debug thing
#         ph = 0  # debug thing
#
#         while i < count and ph < max_ph:
#             if lines[i].isdigit():
#                 if 1 <= int(lines[i]) <= face_count:
#                     box_vector = list()  # faces' coords
#                     labels.append(os.path.join(im_path, lines[i - 1]))
#                     for j in range(i + 1, i + int(lines[i]) + 1):
#                         box_vector += lines[j][1:len(lines[j]) - 1].split(',')
#
#                     box_vector = [int(x) for x in box_vector]
#                     while len(box_vector) < 4 * face_count:
#                         box_vector.append(0)
#
#                     boxes.append(box_vector)
#                     ph += 1
#
#                 i += int(lines[i]) + 1
#             else:
#                 i += 1
#
#     return labels, boxes


# class FaceDataset(Dataset):
#     """Construct dataset which loads images for batch"""
#     def __init__(self, labels, boxes):
#         self.labels = (labels, [False for i in range(len(labels))])
#         self.boxes = boxes
#
#     def __len__(self):
#         return len(self.boxes)
#
#     def __getitem__(self, index):
#         image = transform(self.labels[0][index])
#         # Adjusting box coordinates
#         if not self.labels[1][index]:  # checks whether image was used before
#             height, width = image.shape[:2]
#             x_coef = width / cfg.dim[0]
#             y_coef = height / cfg.dim[1]
#
#             for i_ in range(len(self.boxes[index])):
#                 if i_ % 2 == 0:
#                     self.boxes[index][i_] = int(self.boxes[index][i_] / x_coef)
#                 else:
#                     self.boxes[index][i_] = int(self.boxes[index][i_] / y_coef)
#             self.labels[1][index] = True
#
#         return [image, Variable(torch.tensor(self.boxes[index])).cuda()]


class FaceDataset(data.Dataset):
    """Face Detection Dataset Object

        input is image, target is annotation

        :param
            dataset_path (string): filepath to WIDER folder
            target_transform (callable, optional): transformation to perform on the
                target `annotation`
                (eg: take in caption string, return tensor of word indices)
        """
    def __init__(self, dataset_path, phase, preproc=None):
        self.dataset_path = dataset_path
        self.preproc = preproc
        if phase != "train" and phase != "test" and phase != "val":
            raise RuntimeError("Wrong phase! (Type train, test ot val)")

        self.img_path = os.path.join(self.dataset_path, phase + "/%s")
        anno_path = os.path.join(self.dataset_path, phase + "_boxes.txt")
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
        :param annotation_path: path where all the boxes written
        :return: a tuple of (img_path, list(img_boxes, 1)) : here 1 stands for face class labeled with '1'
        """
        imgs = list()
        boxes = list()

        with open(annotation_path, 'r') as file:
            lines = file.read().splitlines()
            count = len(lines)
            i = 0
            max_ph = 1000000000000000000000  # debug thing
            ph = 0  # debug thing

            while i < count and ph < max_ph:
                if lines[i].isdigit():
                    if 1 <= int(lines[i]):
                        box_vector = list()  # faces' coords
                        imgs.append(self.img_path % lines[i - 1])
                        for j in range(i + 1, i + int(lines[i]) + 1):
                            cur_box = lines[j][1:len(lines[j]) - 1].split(',')
                            cur_box = [int(x) for x in cur_box]
                            cur_box = [cur_box[0], cur_box[3], cur_box[2], cur_box[1], 1]  # [xmin, ymin, xmax, ymax, 1] where 1 states for class '1' - face
                            box_vector.append(cur_box)

                        boxes.append(box_vector)

                        ph += 1

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


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
