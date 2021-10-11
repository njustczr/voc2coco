import os
import xml.etree.ElementTree as ET
import cv2
import json
import platform

class Voc2coco(object):
    """

    """
    def __init__(self,
                 rootpath='',
                 phase='train',
                 split=None):
        self.rootpath = rootpath
        self.phase = phase
        self.split = split

        self.dataset = {"info" : "fabric_defect",
                        "categories" : [],
                        "images" : [],
                        "annotations" : [],
                        "licenses" : ["none"]}
        self.label_dict = {"open-door":1, "person-enn":2, "person-other":3, "warn-all":4}
        self.classes = ["open-door", "person-enn", "person-other", "warn-all"]

    def save_file(self):
        folder = os.path.join(self.rootpath, 'annotations')
        if not os.path.exists(folder):
            os.makedirs(folder)
        json_name = os.path.join(self.rootpath, 'annotations/{}.json'.format(self.phase))
        with open(json_name, 'w') as f:
            json.dump(self.dataset, f)

    def parse_xml(self, xmlp):
        """
        Args:
        xmlp (str):     path to xml file
        Returns:
        bboxes (list):
            list of bboxes with tag (with mask/hat), (x1, y1, x2, y2, tag)
        """
        tree = ET.parse(xmlp)
        root = tree.getroot()
        # `ie` is the index of object in xml file
        result = []
        for elem in root:
            # print(elem.tag)
            if elem.tag == 'object':
                for obj_ in elem:
                    if obj_.tag == 'name':
                        class_name = obj_.text
                    if obj_.tag == 'bndbox':
                        bb = [int(x.text) for x in obj_]
                if platform.system()=='Windows':
                    # names[i][0:-4]
                    #result.append([xmlp.split('\\')[-1].split('.')[0]+'.jpg',class_name, bb[0], bb[1], bb[2], bb[3]])
                    result.append([xmlp.split('\\')[-1][0:-4] + '.jpg', class_name, bb[0], bb[1], bb[2], bb[3]])
                elif platform.system()=='Linux':
                    #result.append([xmlp.split('/')[-1].split('.')[0] + '.jpg', class_name, bb[0], bb[1], bb[2], bb[3]])
                    result.append([xmlp.split('/')[-1][0:-4] + '.jpg', class_name, bb[0], bb[1], bb[2], bb[3]])
        return result

    def __call__(self):
        if not isinstance(self.split, int):
            raise TypeError("split must be an integer")

        # 建立标签和数字id的对应关系
        for i, cls in enumerate(self.classes, 1):
            self.dataset["categories"].append({'id': i,
                                               'name': cls,
                                               'supercategory': 'mark'})

        # 读取images文件夹的图片名称
        indexes = [f for f in os.listdir(os.path.join(self.rootpath, 'images'))]
        if self.phase == 'train':
            indexes = [line for i, line in enumerate(indexes) if i <= self.split]
        if self.phase == 'val':
            indexes = [line for i, line in enumerate(indexes) if i > self.split]

        count = 1
        for k, index in enumerate(indexes):
            # xml_name = index.split('.')[0] + '.xml'
            xml_name = index[0:-4] + '.xml'
            xml_annos = self.parse_xml(os.path.join(os.path.join(self.rootpath, 'xmls'), xml_name))
            im = cv2.imread(os.path.join(os.path.join(self.rootpath, 'images'), index))
            print(xml_name)
            height, width, _ = im.shape
            self.dataset["images"].append({'file_name': index,
                                           'id': k,
                                           'width': width,
                                           'height': height})
            for xml_anno in xml_annos:
                parts = xml_anno

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


        self.save_file()

if __name__ == "__main__":
    num = len(os.listdir(os.path.join('E:\\qian_camera\\test', 'xmls')))
    print(num)
    tool = Voc2coco(rootpath='E:\\qian_camera\\test', phase='train', split=num)
    tool()





