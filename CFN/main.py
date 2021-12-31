import os
import torch
import argparse
import torch.nn as nn

from eval import validate
from train import train_one_epoch
from utils import adjust_learning_rate
from modules.network import Jigsaw_model

from dataset import DataLoader

# Settings parameters
parser = argparse.ArgumentParser(description='Train JigsawPuzzleSolver on Imagenet')
parser.add_argument('--data', type=str, default='tiny-imagenet-200', help='Imagenet folder')
parser.add_argument('--perm_path', type=str, default='permutations_1000.npy', help='permutation')
parser.add_argument('--model', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--classes', default=1000, type=int, help='Number of permutation to use')
parser.add_argument('--epochs', default=70, type=int, help='number of total epochs for training')
parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--cuda', default=True, type=bool, help='use CUDA')
args = parser.parse_args()


# Main Function
def main(args):
    
    ## DataLoader initialize ILSVRC2012_train_processed
    train_data = DataLoader(args.data, args.perm_path, mode='train', classes=args.classes)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    val_data = DataLoader(args.data, args.perm_path, mode='val', classes=args.classes)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)
    
    iter_per_epoch = train_data.N/args.batch_size
    print('Images: train %d, validation %d' % (train_data.N, val_data.N))
    
    # Define the model
    model = Jigsaw_model(args.classes)
    if args.cuda:
        model.cuda()
    
    # Load checkpoints
    if os.path.exists(args.checkpoint):
        files = [f for f in os.listdir(args.checkpoint) if 'pth' in f]
        if len(files)>0:
            files.sort()
            #print files
            ckp = files[-1]
            model.load_state_dict(torch.load(args.checkpoint+'/'+ckp))
            args.iter_start = int(ckp.split(".")[-3].split("_")[-1])
            print('Starting from: ',ckp)
        else:
            if args.model is not None:
                model.load(args.model)
    else:
        if args.model is not None:
            model.load(args.model)

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay = 5e-4)


    steps = args.iter_start
    
    # Training / Evaluation
    print('Start training...')
    for epoch in range(int(args.iter_start/iter_per_epoch),args.epochs):
    
        # Scheduling
        lr = adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=20, decay=0.1)
        
        # Training
        train_one_epoch(args, train_loader, model, criterion, optimizer, lr, steps, epoch)

        # Evaluation
        validate(args, model, criterion, val_loader, steps)




if __name__ == "__main__":
    main(args)
