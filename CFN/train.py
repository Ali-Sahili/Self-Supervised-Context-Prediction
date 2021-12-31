
import numpy as np
from torch.autograd import Variable
from utils import compute_accuracy



def train_one_epoch(args, train_loader, model, criterion, optimizer, lr, steps, epoch):
    model.train()

    for i, (images, labels, original) in enumerate(train_loader):

        images = Variable(images)
        labels = Variable(labels)
        if args.cuda:
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
            
        prec1, prec5 = compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))
        acc = prec1.item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss = float(loss.cpu().data.numpy())

        # Print results
        if steps % 20 ==0:
            print('[%2d/%2d] %5d), LR %.5f, Loss: % 1.3f, Accuracy % 2.2f%%' %(epoch+1, 
                                                            args.epochs, steps, lr, loss,acc))

            original = [im[0] for im in original]
            imgs = np.zeros([9,75,75,3])
            for ti, img in enumerate(original):
                img = img.numpy()
                imgs[ti] = np.stack([(im-im.min())/(im.max()-im.min()) for im in img],axis=2)

        steps += 1

        # Saving checkpoints
        if steps % 1000==0:
            filename = '%s/jps_%03i_%06d.pth.tar'%(args.checkpoint,epoch,steps)
            model.save(filename)
            print('Saved: ' + args.checkpoint)
