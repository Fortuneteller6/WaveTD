# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader


# Metric, loss .etc
from model.utils import *
from model.loss import *
from model.load_param_data import  load_dataset

# Model
from model.model_WaveTD import WaveTD

class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args  = args
        self.save_prefix = '_'.join([args.model, args.dataset])

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Load model
        if args.model   == 'WaveTD':
            model = WaveTD(n_channels=3, n_classes=1, heads=[1, 2, 4, 8])

        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Checkpoint
        checkpoint             = torch.load('result/' + args.model_dir)
        visulization_path      = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_result'
        visulization_fuse_path = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_fuse'

        make_visulization_dir(visulization_path, visulization_fuse_path)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        with torch.no_grad():
            num = 0
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                    
                save_Pred_GT(pred, labels,visulization_path, val_img_ids, num, args.suffix)
                num += 1

            total_visulization_generation(dataset_dir, args.mode, test_txt, args.suffix, visulization_path, visulization_fuse_path)

def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
