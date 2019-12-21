cfg = {
    'dataset_path' : "C:/Maxim/Repositories/AI/Datasets",
    'saving_path' : "C:/Maxim/Repositories/AI/Models/5ClassModelFACESONLY",
    'stats_path' : "C:/Maxim/Repositories/AI/utils/stats/5ClassModelFACESONLY",
    'batch_size' : 20,
    'num_epochs' : 300,
    'learning_rate' : 1e-3,
    'momentum' : 0.9,
    # 'resume_training' : None,  # model path and current epoch
    'resume_training' : ("C:/Maxim/Repositories/AI/Models/5ClassModelFACESONLY/FaceDetection_epoch_144.pth", 145),  # model path and current epoch
    # 'resume_training' : ("C:/Maxim/Repositories/AI/Models/PlusHand/Hand_epoch_63.pth", 138),  # model path and current epoch
    'name': 'FaceDetection',
    'num_classes' : 5,  # '__background__', 'face', 'hand', 'circle', 'straight'
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
    # 'pretrained_model': "C:/Maxim/Repositories/AI/Models/PlusHand/Hand_epoch_72.pth",
    'pretrained_model': "C:/Maxim/Repositories/AI/Models/5ClassModelFACESONLY/FaceDetection_epoch_149.pth",
    'test_results': "C:/Maxim/Repositories/AI/utils/Test results",
    'show_image': True
}