import os
from DataSets.base_loader import BaseLoader
from config import  cfg
from logopr import mylog

class Loader(BaseLoader):
    def __init__(self, mode, img_transform=None):
        super(Loader, self).__init__(mode=mode, img_transform=img_transform)

        self.imgRoot = cfg.DATASET.ROOT
        self.img_ext = cfg.DATASET.IMGEXT

        info = "Load Dataset Shipsear - " + mode + " Mode"
        mylog.msg(info)


        loadImgPath = None
        if mode == 'Train':
            loadImgPath = os.path.join(self.imgRoot, 'Train')
        elif mode == 'Validation':
            loadImgPath = os.path.join(self.imgRoot, 'Validation')
        elif mode == 'Test':
            loadImgPath = os.path.join(self.imgRoot, 'Test')

        self.imags = self.find_imagesAndLabels(loadImgPath, self.img_ext)

    def find_imagesAndLabels(self, imgPath, imgExt):

        items = []
        img_ext = '.' + imgExt

        info = 'The category and quantity of image samples loaded are：'
        mylog.msg(info)


        for dir in os.listdir(imgPath):
            child = os.path.join(imgPath, dir)
            classIdStrIndex = child.rindex('/')
            classIdStr = child[classIdStrIndex + 1 : len(child)]
            classId = int(classIdStr)

            imgsCount = 0
            if os.path.isdir(child):
                for file in os.listdir(child):
                    if os.path.splitext(file)[1] == img_ext:
                        filepath = os.path.join(child, file)
                        item = {'classid':classId,'filepath':filepath}
                        items.append(item)
                        imgsCount += 1

            info = 'ClassId = ' + classIdStr + ", pics count = " + str(imgsCount)
            mylog.msg(info)


        return items

if __name__ == '__main__':
    cfg.DATASET.ROOT = '/home/workdir6/selects_by_pro_randomcrop'
    shipear = Loader('Train')
