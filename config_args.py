import os.path as path 
import os
import numpy as np
from pdb import set_trace as stop


def get_args(parser,eval=False):
    parser.add_argument('--dataroot', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, choices=['coco', 'voc','coco1000','nus','vg','news','cub'], default='coco')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--test_known', type=int, default=0)

    # Optimization
    parser.add_argument('--optim', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=-1)
    parser.add_argument('--grad_ac_steps', type=int, default=1)
    parser.add_argument('--scheduler_step', type=int, default=1000)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--int_loss', type=float, default=0.0)
    parser.add_argument('--aux_loss', type=float, default=0.0)
    parser.add_argument('--loss_type', type=str, choices=['bce', 'mixed','class_ce','soft_margin'], default='bce')
    parser.add_argument('--scheduler_type', type=str, choices=['plateau', 'step'], default='plateau')
    parser.add_argument('--loss_labels', type=str, choices=['all', 'unk'], default='all')
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--max_batches', type=int, default=-1)
    parser.add_argument('--warmup_scheduler', action='store_true',help='')

    # Model
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pos_emb', action='store_true',help='positional encoding') 
    parser.add_argument('--use_lmt', dest='use_lmt', action='store_true',help='label mask training') 
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--no_x_features', action='store_true')

    # CUB
    parser.add_argument('--attr_group_dict', type=str, default='')
    
    parser.add_argument('--n_groups', type=int, default=10,help='groups for CUB test time intervention')
    
    # Image Sizes
    parser.add_argument('--scale_size', type=int, default=640)
    parser.add_argument('--crop_size', type=int, default=576)

    # Testing Models
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--saved_model_name', type=str, default='')
    
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()

    model_name = args.dataset
    if args.dataset == 'voc':
        args.num_labels = 20
    elif args.dataset == 'nus':
        args.num_labels = 1000
    elif args.dataset == 'coco1000':
        args.num_labels = 1000
    elif args.dataset == 'coco':
        args.num_labels = 80
    elif args.dataset == 'vg':
        args.num_labels = 500
    elif args.dataset == 'news':
        args.num_labels = 500
    elif args.dataset == 'cub':
        args.num_labels = 112
    else:
        print('dataset not included')
        exit()
    

    model_name += '.'+str(args.layers)+'layer'
    model_name += '.bsz_{}'.format(int(args.batch_size * args.grad_ac_steps))
    model_name += '.'+args.optim+str(args.lr)#.split('.')[1]
    

    if args.use_lmt:
        model_name += '.lmt'
        args.loss_labels = 'unk'
        model_name += '.unk_loss'
        args.train_known_labels = 100
    else:
        args.train_known_labels = 0


    if args.pos_emb:
        model_name += '.pos_emb'

    if args.int_loss != 0.0:
        model_name += '.int_loss'+str(args.int_loss).split('.')[1]

    if args.aux_loss != 0.0:
        model_name += '.aux_loss'+str(args.aux_loss).replace('.','')

    if args.no_x_features:
        model_name += '.no_x_features'


    
    args.test_known_labels = int(args.test_known*0.01*args.num_labels)

    if args.dataset == 'cub':
        # reset the TOTAL number of labels to be concepts+classes
        model_name += '.step_{}'.format(args.scheduler_step)

        model_name += '.'+args.loss_type+'_loss'
        args.num_labels = 112+200

        args.attr_group_dict = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8, 9], 2: [10, 11, 12, 13, 14, 15], 3: [16, 17, 18, 19, 20, 21], 4: [22, 23, 24], 5: [25, 26, 27, 28, 29, 30], 6: [31], 7: [32, 33, 34, 35, 36], 8: [37, 38], 9: [39, 40, 41, 42, 43, 44], 10: [45, 46, 47, 48, 49], 11: [50], 12: [51, 52], 13: [53, 54, 55, 56, 57, 58], 14: [59, 60, 61, 62, 63], 15: [64, 65, 66, 67, 68, 69], 16: [70, 71, 72, 73, 74, 75], 17: [76, 77], 18: [78, 79, 80], 19: [81, 82], 20: [83, 84, 85], 21: [86, 87, 88], 22: [89], 23: [90, 91, 92, 93, 94, 95], 24: [96, 97, 98], 25: [99, 100, 101], 26: [102, 103, 104, 105, 106, 107], 27: [108, 109, 110, 111]}


    if args.name != '':
        model_name += '.'+args.name
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    model_name = os.path.join(args.results_dir,model_name)
    
    args.model_name = model_name


    if args.inference:
        args.epochs = 1

    
    if os.path.exists(args.model_name) and (not args.overwrite) and (not 'test' in args.name) and (not eval) and (not args.inference) and (not args.resume):
        print(args.model_name)
        overwrite_status = input('Already Exists. Overwrite?: ')
        if overwrite_status == 'rm':
            os.system('rm -rf '+args.model_name)
        elif not 'y' in overwrite_status:
            exit(0)
    elif not os.path.exists(args.model_name):
        os.makedirs(args.model_name)


    return args
