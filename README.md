## Landmark Detection with HRNet

This repository contains code to detect (track) landmark.

### Introduce

Landmark detection usually be used on facial tracking. For this project, we use the same concept to detect landmarks that we are interested in to calculate the object pose.


### Setup
1) Download the model weights [checkpoint.pth](https://drive.google.com/file/d/1TdS4kdbAgGp3pQht7WtQfeWuehBr2OWn/view?usp=sharing) and place the
file in to `/logs`.

2) Install following libary
```
ffmpeg-python>=0.2.0
matplotlib>=3.0.2
munkres>=1.1.2
numpy>=1.16
opencv-python>=3.4
Pillow>=5.4
vidgear>=0.1.4
torch>=1.2.0
torchvision>=0.4.0
tqdm>=4.26
tensorboard>=1.11
tensorboardX>=1.4
```

3) Download [dataset](https://drive.google.com/drive/folders/1fTNadbP4IOBb5TvD8_KBuImBt2Rxd7Bw?usp=sharing) (if you need to train) and place the
file in to `/datasets/monitor`.

4)  Get YOLOv3:

    - Clone [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/47b7c912877ca69db35b8af3a38d6522681b3bb3) in the folder ``./models/detectors`` and change the folder name from ``PyTorch-YOLOv3`` to ``yolo``  

    - Install YOLOv3 required packages  ``pip install -r requirements.txt`` (from folder `./models/detectors/yolo`)

    - Download the pre-trained weights running the script ``download_weights.sh`` from the ``weights`` folder

### Demo

```
python scripts/live-demo.py --camera_id 0
```

#### Training

```
python scripts/train_coco.py
```

For help:
```
python scripts/train_coco.py --help
```

#### Installation instructions

- Your folders should look like:
    ```
    simple-HRNet
    ├── datasets                (datasets - for training only)
    │  └── COCO                 (COCO dataset)
    ├── losses                  (loss functions)
    ├── misc                    (misc)
    │  └── nms                  (CUDA nms module - for training only)
    ├── models                  (pytorch models)
    │  └── detectors            (people detectors)
    │    └── yolo               (PyTorch-YOLOv3 repository)
    │      ├── ...
    │      └── weights          (YOLOv3 weights)
    ├── scripts                 (scripts)
    ├── testing                 (testing code)
    ├── training                (training code)
    └── weights                 (HRnet weights)
    ```

If you want to run the training script on COCO `scripts/train_coco.py`, you have to build the `nms` module first.  
  Please note that a linux machine with CUDA is currently required. 
  Built it with either: 
  - `cd misc; make` or
  - `cd misc/nms; python setup_linux.py build_ext --inplace`  

### Problem

Dataset too leak(only 266 image) need help to label.
Can use labelme to label and use [tool](https://github.com/m5823779/labelme2coco_keypoint) to do pre process.

### Acknowledgments

Our code builds upon [SimpleHRNet](https://github.com/stefanopini/simple-HRNet.git)
    
