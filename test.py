import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np
import torch.utils.data as data


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size = float('inf')):
    images=[]
    
    #디렉토리 경로가 존재하는 지 체크
    #없다면 에러 출력
    #assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)): #하위 디렉토리 검사
        for fnmae in fnames:
            if is_image_file(fnmae):
                path = os.path.join(root, fnmae)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def get_transform(grayscale = True, convert=True,method=Image.BICUBIC):
    transform_list =[]

    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    oszie = [286, 286]
    transform_list.append(transforms.Resize(oszie,method))
    transform_list.append(transforms.RandomCrop(256))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,),(0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    return transforms.Compose(transform_list)

