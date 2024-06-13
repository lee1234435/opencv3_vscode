import yaml

data = {
    'train': "opencv3/opencv3_1/yolov8/data/images/train/",
    # 'test': 'opencv3/opencv3_1/yolov8/data/images/test/',
    'val': "opencv3/opencv3_1/yolov8/data/images/val/",
    'nc': 2,
    'names': {
        0: "pen_o",
        1: "pen_x"
    }
}

with open('opencv3/data.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)
