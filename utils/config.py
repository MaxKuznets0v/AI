cfg = {
    'dataset_path' : "C:/Maxim/Repositories/AI/Datasets",
    'saving_path' : "C:/Maxim/Repositories/AI/Models",
    'stats_path' : "C:/Maxim/Repositories/AI/utils/stats",
    'batch_size' : 20,
    'num_epochs' : 100,
    'learning_rate' : 1e-3,
    'momentum' : 0.9,
    #'resume_training' : None,  # model path and current epoch
    'resume_training' : ("C:/Maxim/Repositories/Models/FaceDetection_epoch_9.pth", 10),  # model path and current epoch
    'name': 'FaceBoxes',
    'num_classes' : 2,  # by default face and background
    'img_dim' : 1024,  # 1024x1024
    'gamma' : 0.1,  # for learning rate adjustment
    'weight_decay' : 5e-4,  # for SGD
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,

    'nms_threshold': 0.3,
    'conf_threshold': 0.05,
    'min_for_visual': 0.5,
    'top_k': 5000,
    'keep_top_k': 750,
    'pretrained_model': "C:/Maxim/Repositories/AI/Models/FaceDetection_epoch_9.pth",
    'test_results': "C:/Maxim/Repositories/AI/utils/Test results",
    'show_image': False
}