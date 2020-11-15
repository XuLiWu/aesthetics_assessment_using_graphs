import torch
import   PNASNet_Base, ResNet_Base, Graph_Models
from termcolor import colored
import torchvision

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def load(args, dset_classes):

    if args.id in ['AIAG', 'AIAG_Extraction']:

        if args.base_model == 'nasnetalarge':
            if args.id in ['AIAG_Extraction']:
                model = PNASNet_Base.pnasnet5large(num_classes=1000, pretrained='imagenet')
        elif args.base_model == 'resnet50':
            if args.id in ['AIAG_Extraction']:
                model = ResNet_Base.resnet50(True)
            else:
                model = Graph_Models.Net_GAT_Conv_SoA_1(args, dset_classes)
        else:
            print('Wrong model specified: %s' % (args.basemodel))
            exit(1)
    else:
        print ('Wrong id specified: %s'%(args.id))
        exit(1)
        

    if args.start_from != None:
        model.load_state_dict(torch.load(args.start_from)['model'])
        print(colored('PTM Loaded from: %s'%(args.start_from),'white'))
    if args.data_precision == 16:
        return model.half().cuda()
    elif args.data_precision == 32:
        return model.float().cuda()
    else:
        print ('Wrong precision specified: %s'%(args.data_precision))
        exit(1)

