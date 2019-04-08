import torch
import config
import time
from config import args_setting
from dataset import RoadSequenceDataset, RoadSequenceDatasetList
from model import generate_model
from torchvision import transforms
from torch.optim import lr_scheduler

def train(args, epoch, model, train_loader, device, optimizer, criterion):
    since = time.time()
    model.train()
    for batch_idx,  sample_batched in enumerate(train_loader):
        data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device) # LongTensor
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    time_elapsed = time.time() - since
    print('Train Epoch: {} complete in {:.0f}m {:.0f}s'.format(epoch,
        time_elapsed // 60, time_elapsed % 60))

def val(args, model, val_loader, device, criterion, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample_batched in val_loader:
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            output,_ = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= (len(val_loader.dataset)/args.test_batch_size)
    val_acc = 100. * int(correct) / (len(val_loader.dataset) * config.label_height * config.label_width)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
        test_loss, int(correct), len(val_loader.dataset), val_acc))
    torch.save(model.state_dict(), '%s.pth'%val_acc)


def get_parameters(model, layer_name):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.UpsamplingBilinear2d
    )
    for name, module in model.named_children():
        if name in layer_name:
            for layer in module.children():
                if isinstance(layer, modules_skipped):
                    continue
                else:
                    for parma in layer.parameters():
                        yield parma


if __name__ == '__main__':
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load data for batches, num_workers for multiprocess
    if args.model == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
        train_loader = torch.utils.data.DataLoader(
            RoadSequenceDatasetList(file_path=config.train_path, transforms=op_tranforms),
            batch_size=args.batch_size,shuffle=True,num_workers=config.data_loader_numworkers)
        val_loader = torch.utils.data.DataLoader(
            RoadSequenceDatasetList(file_path=config.val_path, transforms=op_tranforms),
            batch_size=args.test_batch_size,shuffle=True,num_workers=config.data_loader_numworkers)
    else:
        train_loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=config.train_path, transforms=op_tranforms),
            batch_size=args.batch_size, shuffle=True, num_workers=config.data_loader_numworkers)
        val_loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=config.val_path, transforms=op_tranforms),
            batch_size=args.test_batch_size, shuffle=True, num_workers=config.data_loader_numworkers)

    #load model
    model = generate_model(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam([
    #     {'params': get_parameters(model, layer_name=["inc", "down1", "down2", "down3", "down4"]), 'lr': args.lr * 0.0},
    #     {'params': get_parameters(model, layer_name=["outc", "up1", "up2", "up3", "up4"]), 'lr': args.lr * 0.1},
    #     {'params': get_parameters(model, layer_name=["convlstm"]), 'lr': args.lr * 1},
    # ], lr=args.lr)
    # optimizer = torch.optim.SGD([
    #     {'params': get_parameters(model, layer_name=["conv1_block", "conv2_block", "conv3_block", "conv4_block", "conv5_block"]), 'lr': args.lr * 0.5},
    #     {'params': get_parameters(model, layer_name=["upconv5_block", "upconv4_block", "upconv3_block", "upconv2_block", "upconv1_block"]), 'lr': args.lr * 0.33},
    #     {'params': get_parameters(model, layer_name=["Conv3D_block"]), 'lr': args.lr * 0.5},
    # ], lr=args.lr,momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    class_weight = torch.Tensor(config.class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    best_acc = 0

    pretrained_dict = torch.load(config.pretrained_path)
    model_dict = model.state_dict()

    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)

    # train
    for epoch in range(1, args.epochs+1):
        scheduler.step()
        train(args, epoch, model, train_loader, device, optimizer, criterion)
        val(args, model, val_loader, device, criterion, best_acc)

