from PIL import Image
from  torchvision import transforms

img_path = "E:\PythonWorkSpace\pytorchLearing\hymenoptera_data\\train\\ants\\0013035.jpg"
img = Image.open(img_path)

#ToTensor
tenso_trans = transforms.ToTensor()
tensor_img  =tenso_trans(img)

#Normalize归一化
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)


#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
print(img_resize.size)

#Compose - resize
#变为正方形
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,tenso_trans])#前者输出是后者输入
img_resize_2 = trans_compose(img)
