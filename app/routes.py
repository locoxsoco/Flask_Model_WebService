from app import app
from flask import send_file
import pandas as pd
import os.path
import torch
from torchvision import transforms, datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

PATH = '/home/ubuntu/Flask_Model_WebService/model_test_3.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = torch.hub.load('pytorch/vision:v0.5.0', 'inception_v3', pretrained=False, aux_logits=False)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  # net = nn.DataParallel(net)
net.to(device)
net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
net.eval()

@app.route('/')
@app.route('/index')
def index():
    return "Hello, BREIN!"

@app.route('/API/test2.csv', methods=['GET'])
def get_test_csv():
    # Parameters and DataLoaders
    batch_size = 1
    testset = ImageFolderWithPaths(root='/home/ubuntu/Flask_Model_WebService/test_img_100', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    idx_class = {0: 'aceite',
 1: 'agua',
 2: 'arroz',
 3: 'azucar',
 4: 'cafe',
 5: 'caramelo',
 6: 'cereal',
 7: 'chips',
 8: 'chocolate',
 9: 'especias',
 10: 'frijoles',
 11: 'gaseosa',
 12: 'harina',
 13: 'jamon',
 14: 'jugo',
 15: 'leche',
 16: 'maiz',
 17: 'miel',
 18: 'nueces',
 19: 'pasta',
 20: 'pescado',
 21: 'salsatomate',
 22: 'te',
 23: 'torta',
 24: 'vinagre'}

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)


    imagenes_clasificadas = {'image_id':[],'label':[]}
    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images, names = data[0].to(device), data[2]
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            imagenes_clasificadas['image_id'].append(names[0][54:])
            imagenes_clasificadas['label'].append(idx_class[predicted.item()])


    df = pd.DataFrame(imagenes_clasificadas,columns=['image_id','label'])
    df = df.sort_values(by=['image_id'])
    df.to_csv(r'/home/ubuntu/Flask_Model_WebService/test2.csv', index = False)
    try:
        return send_file('/home/ubuntu/Flask_Model_WebService/test2.csv', attachment_filename='test2.csv')
    except Exception as e:
        return str(e)

@app.route('/API/test.csv', methods=['GET'])
def get_csv():
    # Parameters and DataLoaders
    batch_size = 1
    testset = ImageFolderWithPaths(root='/home/ubuntu/Flask_Model_WebService/test_img_BREIN', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    idx_class = {0: 'aceite',
 1: 'agua',
 2: 'arroz',
 3: 'azucar',
 4: 'cafe',
 5: 'caramelo',
 6: 'cereal',
 7: 'chips',
 8: 'chocolate',
 9: 'especias',
 10: 'frijoles',
 11: 'gaseosa',
 12: 'harina',
 13: 'mermelada',
 14: 'jugo',
 15: 'leche',
 16: 'maiz',
 17: 'miel',
 18: 'nueces',
 19: 'pasta',
 20: 'pescado',
 21: 'salsatomate',
 22: 'te',
 23: 'torta',
 24: 'vinagre'}

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)


    imagenes_clasificadas = {'image_id':[],'label':[]}
    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images, names = data[0].to(device), data[2]
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            imagenes_clasificadas['image_id'].append(names[0][56:])
            imagenes_clasificadas['label'].append(idx_class[predicted.item()])


    df = pd.DataFrame(imagenes_clasificadas,columns=['image_id','label'])
    df = df.sort_values(by=['image_id'])
    df.to_csv(r'/home/ubuntu/Flask_Model_WebService/test.csv', index = False)
    try:
        return send_file('/home/ubuntu/Flask_Model_WebService/test.csv', attachment_filename='test.csv')
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run()

