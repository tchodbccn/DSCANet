
from torch.utils import data
from PIL import Image

class BaseLoader(data.Dataset):
    def __init__(self, mode, img_transform):
        '''

        :param mode:  ‘Train' or 'Validation'
        :param image_transforms: List of image transformation functions
        '''
        self.mode = mode
        self.train = (mode == 'Train')
        self.img_transform = img_transform
        self.imags = None

    def read_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img

    def do_transform(self, img):
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img

    def __getitem__(self, index):
        img_id = self.imags[index]['classid']
        img_path = self.imags[index]['filepath']
        img = self.read_image(img_path)
        img = self.do_transform(img)
        return img, img_id

    def __len__(self):
        return len(self.imags)

