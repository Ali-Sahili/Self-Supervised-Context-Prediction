import time
import torch
import argparse
import numpy as np
import torchvision
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms

from model import AlexNetwork
from dataset import build_dataset
from utils import str2bool, UnNormalize
from Chromatic_aberration.model import ColorAbberationNetwork
from Chromatic_aberration.dataset import ChromaticAberrationDataset


# Setting Parameters
def get_args_parser():
    parser = argparse.ArgumentParser('Training/Evaluation of Chromatic Abberation', 
                                      add_help=False)   
    # Dataset parameters   
    parser.add_argument('--root', default='tiny-imagenet-200/', type=str, help='dataset path')
    parser.add_argument('--ckpt_path', default='model_64_2.pth', type=str, help='checkpoint')
    parser.add_argument('--ratio', default=0.2, type=int, help='ratio ')
    parser.add_argument('--patch_dim', default=15, type=int, help='patch size')
    parser.add_argument('--gap', default=3, type=int, help='gap')
    # Dataloading parameters
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--num_epochs', default=65, type=int, help='number of workers')
    # Training parameters
    parser.add_argument('--cuda', default=True, type=str2bool, help='to use CUDA')
    return parser

# Main function
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
    # Preparing training and validation datasets
    df_trn = build_dataset(args, 'train')
    df_val = build_dataset(args, 'val')

    train_set = ChromaticAberrationDataset(args.patch_dim, args.gap, df_trn['filename'], 
                                              False, data_transforms)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers )

    val_set = ChromaticAberrationDataset(args.patch_dim, args.gap, df_val['filename'], 
                                              True, data_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers )
    # Define CAN model
    modelCAN = ColorAbberationNetwork().to(device)

    # Freezing pretrained layers except the last fc layer
    # Run this if you are loading the pretrained weights of the mail model
    for param in modelCAN.cnn.parameters():
        param.requires_grad = False

    for param in modelCAN.fc6.parameters():
        param.requires_grad = False

    # Define optimizer
    optimizer = torch.optim.SGD(modelCAN.fc.parameters(), lr=0.01, momentum=0.9)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5,
                                                           factor=0.3, verbose=True)

    global_trn_loss = []
    global_val_loss = []

    print("Start training...")
    for epoch in range(args.num_epochs):
        train_running_loss = []
        val_running_loss = []
        start_time = time.time()
        modelCAN.train()
        # Training
        for idx, data in tqdm(enumerate(trainloader), desc='Training', total=int(len(train_set)/args.batch_size)):
            bs, ncrops, c, h, w = data[0].size()
            bs, v1, v2 = data[1].size()

            # Reshape ncrops into batch size
            data[0] = data[0].view(-1, c, h, w)
            data[1] = data[1].view(-1, v2)
            data[0], data[1] = data[0].to(device), data[1].to(device)
        
            optimizer.zero_grad()
            output = modelCAN(data[0])
            loss = torch.sqrt(criterion(output, data[1]))
            loss.backward()
            optimizer.step()
        
            train_running_loss.append(loss.item())

        # Evaluation
        modelCAN.eval()
        with torch.no_grad():
            for idx, data in tqdm(enumerate(val_loader), desc='Validation', total=int(len(val_set)/args.batch_size)):
                bs, ncrops, c, h, w = data[0].size()
                bs, v1, v2 = data[1].size()

                # Reshape ncrops into batch size
                data[0] = data[0].view(-1, c, h, w)
                data[1] = data[1].view(-1, v2)
                data[0], data[1] = data[0].to(device), data[1].to(device)

                output = modelCAN(data[0])
                loss = torch.sqrt(criterion(output, data[1]))
                val_running_loss.append(loss.item())

        global_trn_loss.append(sum(train_running_loss) / len(train_running_loss))
        global_val_loss.append(sum(val_running_loss) / len(val_running_loss))

        scheduler.step(global_val_loss[-1])

        print('Epoch [{}/{}], TRNLoss:{:.4f}, VALLoss:{:.4f}, Time:{:.2f}'.format(
                    epoch + 1, args.num_epochs, global_trn_loss[-1], global_val_loss[-1],
                                                        (time.time() - start_time) / 60))
    
        if epoch % 20 == 0:
            MODEL_SAVE_PATH = f'model_CA_base_bs{args.batch_size}_epochs{args.num_epochs}.pth'
            print(f'Model Saved at {MODEL_SAVE_PATH}')
            torch.save({
                        'epoch': args.num_epochs,
                        'model_state_dict': modelCAN.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'global_trnloss': global_trn_loss,
                        'global_valloss': global_val_loss
                       }, MODEL_SAVE_PATH)


    # plot losses
    plt.plot(range(len(global_trn_loss)), global_trn_loss, label='TRN Loss')
    plt.plot(range(len(global_val_loss)), global_val_loss, label='VAL Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Chromatic Aberration Training/Validation Loss plot')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
