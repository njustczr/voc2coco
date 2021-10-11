import os
import json
import cv2

class Yolo2coco(object):
    '''
    voc type dataset摆放格式：
    Args:
        rootpath(str): Path of yolo type data.
                       -rootpath
                           -images folder
                           -yolo_anno.txt
        phase(str): Save 'train' or 'val' json file.
        split(int): The number of train set.
    '''
    def __init__(self,
                 rootpath = '',
                 phase = 'train',
                 split = None):
        self.rootpath = rootpath
        self.phase = phase
        self.split = split
        self.yolo_anno = 'anno_hand_new.txt'
        self.dataset = {"info" : "fabric_defect",
                        "categories" : [],
                        "images" : [],
                        "annotations" : [],
                        "licenses" : ["none"]}
        self.label_dict = {"golve":1, "hand":2}
        self.classes = ["golve", "hand"]

    def save_file(self):
        folder = os.path.join(self.rootpath, 'annotations')
        if not os.path.exists(folder):
            os.makedirs(folder)
        json_name = os.path.join(self.rootpath, 'annotations/{}.json'.format(self.phase))
        with open(json_name, 'w') as f:
            json.dump(self.dataset, f)

    def __call__(self):
        if self.split == None:
            print('split must be an integer!')
            return
        if not type(self.split) == int:
            print('split must be an integer!')
            return
        # 建立标签和数字id的对应关系
        for i , cls in enumerate(self.classes, 1):
            self.dataset["categories"].append({'id': i,
                                               'name': cls,
                                               'supercategory': 'mark'})
        # 读取images文件夹的图片名称
        indexes = [f for f in os.listdir(os.path.join(self.rootpath, 'images'))]
        if self.phase == 'train':
            indexes = [line for i, line in enumerate(indexes) if i <= self.split]
        if self.phase == 'val':
            indexes = [line for i, line in enumerate(indexes) if i > self.split]

        #读取yolo格式的box信息
        with open(os.path.join(self.rootpath, self.yolo_anno)) as tr:
            annos = tr.readlines()

        count = 1
        for k, index in enumerate(indexes):
            im = cv2.imread(os.path.join(os.path.join(self.rootpath, 'images'), index))
            height, width, _ = im.shape
            self.dataset["images"].append({'file_name': index,
                                           'id': k,
                                           'width': width,
                                           'height': height})

            for ii, anno in enumerate(annos):
                parts = anno.strip().split()

                if parts[0] == index:
                    if parts[1] == "ignore":
                        cls_id = 0
                        # x_min, y_min, x_max, y_max
                        x1, y1, x2, y2 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                        width = max(0, x2 - x1)
                        height = max(0, y2 - y1)
                        self.dataset['annotations'].append({'area': width * height,
                                                            'bbox': [x1, y1, width, height],
                                                            'category_id': int(cls_id),
                                                            'id': count,
                                                            'image_id': k,
                                                            'iscrowd': 1,
                                                            # mask, 矩形是从左上角点按顺时针的四个顶点
                                                            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]})
                        count += 1
                    else:
                        cls_id = self.label_dict[parts[1]]
                        # x_min, y_min, x_max, y_max
                        x1, y1, x2, y2 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                        width = max(0, x2 - x1)
                        height = max(0, y2 - y1)
                        self.dataset['annotations'].append({'area': width * height,
                                                            'bbox': [x1, y1, width, height],
                                                            'category_id': int(cls_id),
                                                            'id': count,
                                                            'image_id': k,
                                                            'iscrowd': 0,
                                                            # mask, 矩形是从左上角点按顺时针的四个顶点
                                                            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]})
                        count += 1

        # save结果文件夹
        self.save_file()

if __name__ == "__main__":
    num = len(os.listdir(os.path.join('hand0722', 'images')))
    print(num)
    tool = Yolo2coco(rootpath='hand0722', phase='train', split=num)
    tool()





