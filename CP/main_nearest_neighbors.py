import torch
import argparse
import numpy as np
import torchvision
from torchvision import transforms

from model import AlexNetwork
from utils import str2bool, UnNormalize
from dataset import build_dataset, MyDataset

import matplotlib.pyplot as plt

# Setting Parameters
def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation-Nearst Neighbors', add_help=False)
    
    # Dataset parameters   
    parser.add_argument('--root', default='tiny-imagenet-200/', type=str, help='dataset path')
    parser.add_argument('--ckpt_path', default='model_64_2.pth', type=str, help='checkpoint path')
    parser.add_argument('--ratio', default=0.2, type=int, help='ratio ')
    parser.add_argument('--patch_dim', default=15, type=int, help='patch size')
    parser.add_argument('--gap', default=3, type=int, help='gap')
    # Dataloading parameters
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    # Training parameters
    parser.add_argument('--cuda', default=True, type=str2bool, help='to use CUDA')
    return parser

def main(args):
    # Set device:CPU/GPU
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # Un-normalizing
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    # Load model and weights
    model = AlexNetwork().to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # data transformations
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])])
    # Preparing test dataset
    df_test = build_dataset(args, 'val')
    
    test_set = MyDataset(args.patch_dim, args.gap, df_test['filename'], False, data_transforms)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers )


    data_iter_1 = iter(test_loader)
    data_iter_2 = iter(test_loader)

    example_batch = next(data_iter_1)
    vectors = []
    for j, data in enumerate(test_loader,0):
        img0, img1, label = data
        label = label.reshape([-1])
        img0, img1 , label = data[0].to(device), data[1].to(device), data[2].to(device)
        output ,output1,output2= model(img0,img1)
        img1 = img1.cpu().detach().numpy()
        output2 = output2.cpu().detach().numpy()
        for i in range(len(output2)):
            vectors.append([img1[i],output2[i]])


    img0 , img1 , label = example_batch
    label = label.reshape([-1])
    img0, img1 , label = data[0].to(device), data[1].to(device), data[2].to(device)
    output ,output1,output2= model(img0,img1)
    output2 = output2.cpu().detach().numpy()
    img1 = img1.cpu().detach().numpy()

    for i in range(20):
        vectors.sort(key=lambda tup: np.linalg.norm(tup[1]-output2[i]))
        npimg = img1[i]
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(1,10,1)
        plt.axis("off")
        ax1.imshow(np.transpose(unorm(torch.tensor(npimg)), (1, 2, 0)))
        for j in range(1,10):
            ax1 = fig.add_subplot(1,10,j+1)
            ax1.imshow(np.transpose(unorm(torch.tensor(vectors[j-1][0])), (1, 2, 0)))
            plt.axis("off")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation-Nearst Neighbors', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
