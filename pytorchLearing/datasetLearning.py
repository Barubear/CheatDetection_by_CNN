from torch.utils.data import Dataset
from PIL import Image
import os
class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        #拼接路径
        #文件夹所在路径
        self.path = os.path.join(self.root_dir,self.label_dir)
        #所有图片路径
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idex):
        img_name = self.img_path[idex]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label =self.label_dir

        return img , label

    def __len__(self):
        return len(self.img_path)


root_dir = 'hymenoptera_data/train'
ants_label = 'ants'
bees_label = 'bees'

ants_dataset = MyData(root_dir,ants_label)
antimg01,antslabel01 = ants_dataset[0]
#antimg01.show()
print(antimg01.type)