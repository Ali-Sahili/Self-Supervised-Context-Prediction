
import numpy as np
from utils import compute_accuracy
from torch.autograd import Variable



def validate(args, model, criterion, val_loader, steps):
    #print('Evaluating network.......')
    accuracy = []
    model.eval()
    for i, (images, labels, _) in enumerate(val_loader):
        images = Variable(images)
        if args.cuda:
            images = images.cuda()

        # Forward + Backward + Optimize
        outputs = model(images)
        outputs = outputs.cpu().data

        prec1, prec5 = compute_accuracy(outputs, labels, topk=(1, 5))
        accuracy.append(prec1.item())

    print('TESTING: %d), Accuracy %.2f%%' % (steps, np.mean(accuracy)))
