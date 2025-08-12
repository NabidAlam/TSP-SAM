import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from glob import glob
import os.path as osp
import pdb
from mypath import Path
import cv2
import copy

# several data augumentation strategies
def cv_random_flip(imgs, imgs_ycbcr, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(len(imgs_ycbcr)):
            imgs_ycbcr[i] = imgs_ycbcr[i].transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, imgs_ycbcr, label

def randomCrop(imgs, imgs_ycbcr, label):
    border = 30
    image_width = imgs[0].size[0]
    image_height = imgs[0].size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    
    for i in range(len(imgs)):
        imgs[i] = imgs[i].crop(random_region)
    for i in range(len(imgs_ycbcr)):
        imgs_ycbcr[i] = imgs_ycbcr[i].crop(random_region)
        
    return imgs, imgs_ycbcr, label.crop(random_region)

def randomRotation(imgs, imgs_ycbcr, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        for i in range(len(imgs)):
            imgs[i] = imgs[i].rotate(random_angle, mode)
        for i in range(len(imgs_ycbcr)):
            imgs_ycbcr[i] = imgs_ycbcr[i].rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return imgs, imgs_ycbcr, label

def colorEnhance(imgs, imgs_ycbcr):
    for i in range(len(imgs)-3):
        bright_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        imgs[i] = ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        imgs[i] = ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
    for i in range(len(imgs_ycbcr)):
        bright_intensity = random.randint(5, 15) / 10.0
        imgs[i+len(imgs)-3] = ImageEnhance.Brightness(imgs[i+len(imgs)-3]).enhance(bright_intensity)
        imgs_ycbcr[i] = ImageEnhance.Brightness(imgs_ycbcr[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        imgs[i+len(imgs)-3] = ImageEnhance.Contrast(imgs[i+len(imgs)-3]).enhance(contrast_intensity)
        imgs_ycbcr[i] = ImageEnhance.Contrast(imgs_ycbcr[i]).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        imgs[i+len(imgs)-3] = ImageEnhance.Color(imgs[i+len(imgs)-3]).enhance(color_intensity)
        imgs_ycbcr[i] = ImageEnhance.Color(imgs_ycbcr[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        imgs[i+len(imgs)-3] = ImageEnhance.Sharpness(imgs[i+len(imgs)-3]).enhance(sharp_intensity)
        imgs_ycbcr[i] = ImageEnhance.Sharpness(imgs_ycbcr[i]).enhance(sharp_intensity)
    return imgs, imgs_ycbcr

class dataLoader(data.Dataset):
    # preload data into memory
    def __init__(self, data_root, split='train', imgsize=352, clip_n=3, is_gray=False, name=None, scene=None):
        
        self.split = split
        self.is_gray = is_gray
        self.imgsize = imgsize
        self.clip_n = clip_n

        self.image_list = []
        self.extra_info = []
        self.name = name
        self.scene = scene
        
        if split == 'train':
            self.image_list = [osp.join(data_root, i) for i in os.listdir(data_root) if i.endswith('.jpg')]
            self.label_list = [osp.join(data_root.replace('TrainDataset', 'TrainDataset_GT'), i[:-4] + '.png') for i in os.listdir(data_root) if i.endswith('.jpg')]
        elif split == 'test':
            if name is not None:
                if scene is not None:
                    self.image_list = [osp.join(data_root, scene, name)]
                else:
                    self.image_list = [osp.join(data_root, name)]
            else:
                self.image_list = [osp.join(data_root, i) for i in os.listdir(data_root) if i.endswith('.avi') or i.endswith('.mp4')]
        
        self.image_list.sort()
        if split == 'train':
            self.label_list.sort()
            assert len(self.image_list) == len(self.label_list)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.imgsize, self.imgsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.imgsize, self.imgsize)),
            transforms.ToTensor()])
        self.loop_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.image_list)

    def __getitem__(self, item):
        # for data
        if self.split == 'train':
            data = Image.open(self.image_list[item]).convert('RGB')
            data = self.transform(data).unsqueeze(0)

            label = Image.open(self.label_list[item]).convert('L')
            label = self.gt_transform(label).unsqueeze(0)
        else:
            try:
                input_avi = cv2.VideoCapture(self.image_list[item])
            except:
                print(self.image_list[item])
            target_fps = round(input_avi.get(cv2.CAP_PROP_FPS))
            frame_num = 0
            image_seq = []
            while (input_avi.isOpened()):
                ret, frame = input_avi.read()
                if ret:
                    frame_num = frame_num + 1
                    image_seq.append(frame)
                    if frame_num % 1000 == 0:
                        print(frame_num)
                else:
                    break
            input_avi.release()
            data = []
            for i in range(len(image_seq)):
                data.append(self.loop_transform(Image.fromarray(cv2.cvtColor(image_seq[i], cv2.COLOR_BGR2RGB))).unsqueeze(0))

            label = self.image_list[item].split('/')[-1]

        return self.image_list[item], data, label

    def __len__(self):
        return self.size



def generate_point(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.array([mask.shape[1]//2, mask.shape[0]//2]), mask.shape
    else:
        area = []
        for j in range(len(contours)):
            area.append(cv2.contourArea(contours[j]))
        max_idx = np.argmax(area)
        M = cv2.moments(contours[max_idx])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return np.array([cx, cy]), mask.shape

class test_dataset(data.Dataset):
    def __init__(self, dataset, testsize=352, test_path=None, sequence=None, grid=8):
        self.dataset = dataset
        self.testsize = testsize
        self.grid = grid
        self.image_list = []
        self.gt_list = []
        
        if dataset == 'MoCA':
            data_root = Path.db_root_dir('MoCA') + 'TestDataset_per_sq'
            for scene in sorted(os.listdir(osp.join(data_root))):
                image_root = osp.join(data_root, scene)
                gt_root = image_root.replace('TestDataset_per_sq', 'TestDataset_GT_per_sq')
                img_list = sorted(glob(osp.join(image_root, '*.jpg')))
                gt_list = sorted(glob(osp.join(gt_root, '*.png')))
                for i in range(0, len(img_list), 3):
                    if len(img_list[i:i+3]) == 3:
                        self.image_list += [img_list[i:i+3]]
                        self.gt_list += [gt_list[i+1]]
        elif dataset == 'CAD2016':
            data_root = Path.db_root_dir('CAD2016')
            for scene in sorted(os.listdir(osp.join(data_root, 'img'))):
                image_root = osp.join(data_root, 'img', scene)
                gt_root = osp.join(data_root, 'gt', scene)
                img_list = sorted(glob(osp.join(image_root, '*.jpg')))
                gt_list = sorted(glob(osp.join(gt_root, '*.png')))
                for i in range(0, len(img_list), 3):
                    if len(img_list[i:i+3]) == 3:
                        self.image_list += [img_list[i:i+3]]
                        self.gt_list += [gt_list[i+1]]
        elif dataset == 'DAVIS':
            # Load from provided test_path / sequence
            if test_path is None or sequence is None:
                raise ValueError("For DAVIS, provide --test_path and --sequence")
            image_root = osp.join(test_path, sequence)
            gt_root = osp.join(test_path.replace('JPEGImages', 'Annotations'), sequence)
            img_list = sorted(glob(osp.join(image_root, '*.jpg')))
            gt_list = sorted(glob(osp.join(gt_root, '*.png'))) if os.path.exists(gt_root) else [None] * len(img_list)
            for i in range(len(img_list)):
                # For TSP-SAM, process in clips of 3 frames if possible, else single
                clip = img_list[max(0, i-1):min(len(img_list), i+2)]
                # Pad clip to 3 frames if shorter by duplicating the last frame
                while len(clip) < 3:
                    clip.append(clip[-1])  # Duplicate last
                self.image_list += [clip]
                self.gt_list += [gt_list[i] if gt_list[i] else None]
        elif dataset == 'TED':
            # Assume MP4 videos; extract frames on-the-fly
            if test_path is None or sequence is None:
                raise ValueError("For TED, provide --test_path and --sequence (e.g., video1.mp4)")
            video_path = osp.join(test_path, sequence)
            cap = cv2.VideoCapture(video_path)
            frame_list = []
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_path = f'temp_frame_{frame_num}.jpg'  # Temp save or in-memory
                    cv2.imwrite(frame_path, frame)
                    frame_list.append(frame_path)
                    frame_num += 1
                else:
                    break
            cap.release()
            # No GT for TED
            gt_list = [None] * len(frame_list)
            for i in range(len(frame_list)):
                clip = frame_list[max(0, i-1):min(len(frame_list), i+2)]
                while len(clip) < 3:
                    clip.append(clip[-1])  # Duplicate last
                self.image_list += [clip]
                self.gt_list += [gt_list[i]]
            # Clean temp frames if needed
        else:
            raise NotImplementedError(f"Dataset {dataset} not supported")
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.img_sam_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])

        self.index = 0
        self.size = len(self.gt_list)

    def load_data(self):
        imgs = []
        imgs_sam = []
        imgs_ycbcr = []

        for i in range(len(self.image_list[self.index])):
            imgs += [self.rgb_loader(self.image_list[self.index][i])]
            imgs_sam += [self.rgb_loader(self.image_list[self.index][i])]
            imgs_ycbcr += [self.ycbcr_loader(self.image_list[self.index][i])]
            
            imgs[i] = self.img_transform(imgs[i]).unsqueeze(0)
            imgs_sam[i] = (self.img_sam_transform(imgs_sam[i]) * 255.0).unsqueeze(0)
            imgs_ycbcr[i] = self.img_transform(imgs_ycbcr[i]).unsqueeze(0)
            
        # Robust scene extraction
        path_parts = self.image_list[self.index][-1].split(os.sep) if os.sep in self.image_list[self.index][-1] else self.image_list[self.index][-1].split('/')
        scene = path_parts[-3] if len(path_parts) >= 3 else self.dataset  # Fallback to dataset name if path short
        
        name = path_parts[-1]
        gt = self.binary_loader(self.gt_list[self.index]) if self.gt_list[self.index] else None
        gt = self.transform(gt) if gt is not None else None
        
        if gt is not None:
            gt_array = np.array(gt[0, :, :] * 255).astype(np.uint8)
            points, original_size = generate_point(gt_array)
        else:
            points = np.array([self.testsize//2, self.testsize//2])
            original_size = (self.testsize, self.testsize)
        
        # Normalize points to [0,1] range
        points = np.array([points[0] / original_size[1], points[1] / original_size[0]])
        
        self.index += 1
        self.index = self.index % self.size
    
        return imgs, imgs_ycbcr, imgs_sam, points, gt, name, scene

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
            
    def ycbcr_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('YCbCr')
            
    def __len__(self):
        return self.size
    
class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        
        self.image_path, self.label_path = [], []
        
        lst_pred = sorted(os.listdir(img_root))
        for l in lst_pred:
            self.image_path.append(osp.join(img_root, l))
            self.label_path.append(osp.join(label_root, 'GT', l))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        img_name = self.image_path[item]

        return pred, gt, img_name

    def __len__(self):
        return len(self.image_path)