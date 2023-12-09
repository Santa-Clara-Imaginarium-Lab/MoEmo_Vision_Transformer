import os
import numpy as np
import torch
import argparse
import random
from torch.utils.data import Dataset
# from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import sys
from model import EmotionNetwork

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def configure_optimizers(net, args):

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-cd", "--contextDataset", type=str,
        default='D:/Tianma/dataset/Pre_process/train_tensor/',
        help="Training dataset"
    )

    parser.add_argument(
        "-cdt", "--contextDataset_test", type=str,
        default='D:/Tianma/dataset/Pre_process/test_tensor/',
        help="Training dataset"
    )

    parser.add_argument(
        "-sd", "--skeletonDataset", type=str,
        default='D:/Tianma/dataset/Pre_process/joints/',
        help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=1000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=12, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",type=int,default=2,help="Test batch size (default: %(default)s)",
    )
    parser.add_argument("--cuda",  default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save_path", type=str, default="./save/", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint",
     default="",  # ./train0008/18.ckpt
     type=str, help="Path to a checkpoint")

    args = parser.parse_args(argv)
    return args


class myDataset(Dataset):
    def __init__(self, root,skeletonPath, split="train"):
        self.skeletonPath = skeletonPath

        emotion_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

        # contexts is from ClIP pre-trained model
        self.contexts = []
        supported = ['.pt'] # save the output of last layer of CLIP
        for cla in emotion_class:
            cla_path = os.path.join(root, cla)
            self.contexts = self.contexts + [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                                    if os.path.splitext(i)[-1] in supported]

    def __getitem__(self, index):
        contextTensors = torch.load(self.contexts[index])
        info = self.contexts[index].split('/')[-1].split('\\')
        ID = info[1][:-3]
        className = info[0]
        jointsPath = os.path.join(self.skeletonPath,ID+'_f16.npy')

        skeleton = np.load(jointsPath)
        input1 = skeleton[:-1, :, :]
        input2 = skeleton[1:, :, :]

        skeletons = np.dstack((input1, input2))
        if className == 'angry':
            label = 0
        elif className == 'happy':
            label = 1
        elif className == 'sad':
            label = 2
        elif className == 'fear':
            label = 3
        elif className == 'surprise':
            label = 4
        elif className == 'disgust':
            label = 5
        else:
            print('error label')

        return contextTensors,skeletons,label

    def __len__(self):
        return len(self.contexts)


# class skeletonDataset(Dataset):
#     def __init__(self, root, split="train"):
#         splitdir = Path(root)
#
#         if not splitdir.is_dir():
#             raise RuntimeError(f'Invalid directory "{root}"')
#
#         self.samples = [f for f in splitdir.iterdir() if f.is_file()]
#
#     def __getitem__(self, index):
#         input = self.samples[index]
#
#         input1 = input[:-1,:,:]
#         input2 = input[1:,:,:]
#
#         output = np.dstack((input1, input2))
#         return output
#
#     def __len__(self):
#         return len(self.samples)


def train_one_epoch(model, train_dataloader, optimizer, epoch, clip_max_norm):
    model.train()
    device = next(model.parameters()).device
    start = time.time()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0

    for i, d in enumerate(train_dataloader):

        contextTensors, skeletons, label = d
        optimizer.zero_grad()
        sample_num += contextTensors.shape[0]

        out_net = model(contextTensors.to(device),skeletons.to(device))
        pred_classes = torch.max(out_net, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()

        loss_function = torch.nn.CrossEntropyLoss()
        out_criterion = loss_function(out_net, label.to(device))
        out_criterion.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 30 == 0:
            enc_time = time.time() - start
            start = time.time()
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion.item():.4f} |'
                f'\tacc: {accu_num.item() / sample_num:.4f} |'
                f"\ttime: {enc_time:.1f}"
            )

def test_epoch(epoch, test_dataloader, model):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0

    with torch.no_grad():
        for d in test_dataloader:
            contextTensors, skeletons, label = d
            sample_num += contextTensors.shape[0]
            out_net = model(contextTensors.to(device), skeletons.to(device))
            pred_classes = torch.max(out_net, dim=1)[1]
            accu_num += torch.eq(pred_classes, label.to(device)).sum()

            out_criterion = loss_function(out_net, label.to(device))

            loss.update(out_criterion)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f'\tacc: {accu_num.item() / sample_num:.4f} |'
    )
    return loss.avg

def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)


    train_dataset = myDataset(args.contextDataset,args.skeletonDataset)
    test_dataset = myDataset(args.contextDataset_test, args.skeletonDataset)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = EmotionNetwork()
    net = net.to(device)

    # print('GPU:',torch.cuda.device_count())
    #
    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=400)
    # criterion = RateDistortionLoss(lmbda=args.lmbda)
    #
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        new_state_dict = checkpoint["state_dict"]
        # new_state_dict = OrderedDict()

        # for k, v in checkpoint["state_dict"].items():
        #     # if 'gaussian_conditional' in k:
        #     #     new_state_dict[k]=v
        #     #     print(k)
        #     #     continue
        #     # if 'module' not in k:
        #     k = k[7:]
        #     # else:
        #     #     k = k.replace('features.module.', 'module.features.')
        #     new_state_dict[k]=v

        # net.load_state_dict(new_state_dict)


        optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(net,train_dataloader,optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                args.save_path +str(epoch)+'.ckpt'
            )

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])











