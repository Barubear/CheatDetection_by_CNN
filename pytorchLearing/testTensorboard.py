from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('logs')

#writer.add_image()
for i in range(-50,50):
    writer.add_scalar('y=x*x',i*i,i)
writer.close()

#tensorboard --logdir=E:\PythonWorkSpace\pytorchLearing\logs
