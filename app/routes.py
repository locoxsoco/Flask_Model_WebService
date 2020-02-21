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

PATH = '/home/ubuntu/model_test_3.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = torch.hub.load('pytorch/vision:v0.5.0', 'inception_v3', pretrained=False, aux_logits=False)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  # net = nn.DataParallel(net)
net.to(device)
net.load_state_dict(torch.load(PATH))
net.eval()

@app.route('/')
@app.route('/index')
def index():
    return "Hello, Wooorld!"

@app.route('/API/test.csv', methods=['GET'])
def get_csv():
    # Parameters and DataLoaders
    batch_size = 1
    testset = datasets.ImageFolderWithPaths(root='/home/ubuntu/test_img_100', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    testset.idx_to_class = {v: k for k, v in testset.class_to_idx.items()}

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)


    imagenes_clasificadas = {'image_id':[],'label':[]}
    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images, names = data[0].to(device), data[2]
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            imagenes_clasificadas['image_id'].item().append(names)
            imagenes_clasificadas['label'].item().append(predicted.item())


    df = pd.DataFrame(imagenes_clasificadas,columns=['image_id','label'])
    df.to_csv(r'/home/ubuntu/test.csv')
    try:
        return send_file('/home/ubuntu/test.csv', attachment_filename='test.csv')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
    