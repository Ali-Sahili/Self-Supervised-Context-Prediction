import time
import argparse
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

from model import AlexNetwork
from dataset import build_dataset, MyDataset
from utils import UnNormalize, imshow, str2bool


# Setting Parameters
def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    
    # Dataset parameters   
    parser.add_argument('--root', default='tiny-imagenet-200/', type=str, help='dataset path')
    parser.add_argument('--ratio', default=0.2, type=int, help='ratio ')
    parser.add_argument('--patch_dim', default=15, type=int, help='patch size')
    parser.add_argument('--gap', default=3, type=int, help='gap')
    # Dataloading parameters
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    # Training parameters
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=65, type=int, help='number of epochs.')
    parser.add_argument('--update_freq', default=10, type=int, 
                        help='frequency of batches with which to update counter.')
    parser.add_argument('--cuda', default=True, type=str2bool, help='to use CUDA')
    parser.add_argument('--visualize', default=False, type=str2bool, help='to visualize dataset')
    return parser

# Main function
def main(args):
    # Set device: CPU/GPU
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Un-normalizing
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # data transformations
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225])])
    # Preparing training and validation datasets
    df_trn = build_dataset(args, 'train')
    df_val = build_dataset(args, 'val')
    
    train_set = MyDataset(args.patch_dim, args.gap, df_trn['filename'], False, data_transforms)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers )

    val_set = MyDataset(args.patch_dim, args.gap, df_val['filename'], True, data_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers )

    # Visualizing training dataset
    if args.visualize:
        example_batch = next(iter(trainloader))
        concatenated = torch.cat((unorm(example_batch[0]),unorm(example_batch[1])),0)
        imshow(torchvision.utils.make_grid(concatenated))
        print(f'Labels: {example_batch[2].numpy()}')

    # Visualizing validation dataset
    if args.visualize:
        example_batch_val = next(iter(val_loader))
        concatenated = torch.cat((unorm(example_batch_val[0]),unorm(example_batch_val[1])),0)
        imshow(torchvision.utils.make_grid(concatenated))
        print(f'Labels: {example_batch_val[2].numpy()}')
    
    # Define the model
    model = AlexNetwork().to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5,
                                                           factor=0.3, verbose=True)
    # Training / Evaluation 
    global_trn_loss = []
    global_val_loss = []
    
    print("Start training...")
    for epoch in range(args.num_epochs):
        train_running_loss = []
        val_running_loss = []
        start_time = time.time()
        model.train()
        
        # Training
        for idx,data in tqdm(enumerate(trainloader),total=int(len(train_set)/args.batch_size)):
            uniform_patch = data[0].to(device)
            random_patch = data[1].to(device)
            random_patch_label = data[2].to(device)
            
            output, output_fc6_uniform, output_fc6_random = model(uniform_patch, random_patch)
            loss = criterion(output, random_patch_label)
            
            # Back-prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_running_loss.append(loss.item())
        
        # Evaluation
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for idx,data in tqdm(enumerate(val_loader),total=int(len(val_set)/args.batch_size)):
                uniform_patch = data[0].to(device)
                random_patch = data[1].to(device)
                random_patch_label = data[2].to(device)
                
                output, out_fc6_uniform, out_fc6_random = model(uniform_patch, random_patch)
                loss = criterion(output, random_patch_label)
                
                val_running_loss.append(loss.item())
        
                _, predicted = torch.max(output.data, 1)
                total += random_patch_label.size(0)
                correct += (predicted == random_patch_label).sum()
                
            print('Val Progress --- total:{}, correct:{}'.format(total, correct.item()))
            print('Val Accuracy of the network on test set: {}%'.format(100*correct / total))

        # Recording losses
        global_trn_loss.append(sum(train_running_loss) / len(train_running_loss))
        global_val_loss.append(sum(val_running_loss) / len(val_running_loss))

        scheduler.step(global_val_loss[-1])

        print('Epoch [{}/{}], TRNLoss:{:.4f}, VALLoss:{:.4f}, Time:{:.2f}'.format(
              epoch + 1, args.num_epochs, global_trn_loss[-1], global_val_loss[-1],
                                                         (time.time() - start_time) / 60))
        # Saving checkpoints
        if epoch % 20 == 0:
            MODEL_SAVE_PATH = f'model_{args.batch_size}_{args.num_epochs}.pth'
      
            torch.save({
                         'epoch': args.num_epochs,
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'loss': loss,
                         'global_trnloss': global_trn_loss,
                         'global_valloss': global_val_loss
                       }, MODEL_SAVE_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
