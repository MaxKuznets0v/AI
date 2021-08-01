import torch
import numpy as np
from utils.priors import PriorBox
import cv2
import torch.backends.cudnn as cudnn
from ssd import FaceDetectionSSD
from utils.box_utils import nms, decode
from utils.config import cfg


def detect_faces(img_path, save_path=None):
    print("Starting detection...")
    # Loading pretrained model
    net = FaceDetectionSSD("test", cfg['img_dim'], cfg['num_classes'])
    net.load_state_dict(torch.load(cfg['pretrained_model']))
    net.eval()
    cudnn.benchmark = True
    device = torch.device("cuda:0" if cfg['gpu_train'] else "cpu")
    net.to(device)
    rgb_mean = (104, 117, 123)  # BGR order

    # Processing image
    init_im = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if init_im is None:
        raise RuntimeError("Image does not exist!")

    image = np.float32(init_im)
    im_height, im_width, _ = image.shape
    scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    image -= rgb_mean
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)
    image = image.to(device)
    scale = scale.to(device)

    loc, conf = net(image)
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > cfg["conf_threshold"])[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:cfg["top_k"]]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep_ind, keep_count = nms(torch.from_numpy(boxes.astype(np.float32)), torch.from_numpy(scores.astype(np.float32)),
                               cfg["nms_threshold"], cfg['keep_top_k'])
    dets = dets[keep_ind[:keep_count], :]

    # keep top-K faster NMS
    try:
        dets = dets[:cfg["keep_top_k"], :]
    except:
        dets = dets.reshape((1, dets.size))[:cfg["keep_top_k"], :]

    face_count = 0
    # Show image and saving image results
    for b in dets:
        if b[4] < cfg['min_for_visual']:
            continue
        face_count += 1
        text = "{:.2f}%".format(b[4] * 100)
        b = list(b)
        for i in range(4): b[i] = int(b[i])
        cv2.rectangle(init_im, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        cx = b[0]
        cy = b[1] - 5
        cv2.putText(init_im, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    if save_path is not None:
        cv2.imwrite(save_path, init_im)
    print("Detection completed!")
    print(f"Found {face_count} faces!")
    return init_im
