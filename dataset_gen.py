import cv2
import os
import torch
import torch.utils.data as data
import numpy as np

WIDER_CLASSES = ('__background__', 'face', 'hand', 'circle', 'straight')


class FaceDataset(data.Dataset):
    """Face Detection Dataset Object
        :param:
        dataset_path: path to dataset
        phase: could be 'train' or 'test' or 'val'
        max_images: maximum images in dataset
        prepoc: prepocessing algorithm for each photo
        """
    def __init__(self, dataset_path, phase, max_images=None, preproc=None):
        self.dataset_path = dataset_path
        self.preproc = preproc
        self.max_images = max_images
        if phase != "train" and phase != "test" and phase != "val":
            raise RuntimeError("Wrong phase! (Type train, test ot val)")

        self.img_path = os.path.join(self.dataset_path, phase + "/%s")
        anno_path = os.path.join(self.dataset_path, phase + "_boxes.txt")
        self.ids = self.parse_boxes(anno_path, self.max_images)  # actual iterable object

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

    def parse_boxes(self, annotation_path, max_images=None):
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
            ph = 0

            while i < count and (max_images is None or ph < max_images):
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
        stacks each target in one np array
        Transforms a list(img_box, 1) annotation into a Tensor of bbox coords and label index
        :returns
            np array associated with image size [faces_num, 5]
        """
        res = np.empty((0, 5))
        for box in target:
            res = np.vstack((res, box))

        return res


def detection_collate(batch):
    """
        Collate function is used if images in a batch have a different number
        of object

    :param:
        batch: list of images with its annotations
    :returns:
        tuple (images tensor, list of tensors for each image)
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
    res = (torch.stack(imgs, 0), targets)
    return res
