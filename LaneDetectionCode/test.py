import torch
import config
from config import args_setting
from dataset import RoadSequenceDataset, RoadSequenceDatasetList
from model import generate_model
from torchvision import transforms
from torch.optim import lr_scheduler
from PIL import Image
import numpy as np
import cv2

def output_result(model, test_loader, device):
    model.eval()
    k = 0
    feature_dic=[]
    with torch.no_grad():
        for sample_batched in test_loader:
            k+=1
            print(k)
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            output,feature = model(data)
            feature_dic.append(feature)
            pred = output.max(1, keepdim=True)[1]
            img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255
            img = Image.fromarray(img.astype(np.uint8))

            data = torch.squeeze(data).cpu().numpy()
            if args.model == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
                data = np.transpose(data[-1], [1, 2, 0]) * 255
            else:
                data = np.transpose(data, [1, 2, 0]) * 255
            data = Image.fromarray(data.astype(np.uint8))
            rows = img.size[0]
            cols = img.size[1]
            for i in range(0, rows):
                for j in range(0, cols):
                    img2 = (img.getpixel((i, j)))
                    if (img2[0] > 200 or img2[1] > 200 or img2[2] > 200):
                        data.putpixel((i, j), (234, 53, 57, 255))
            data = data.convert("RGB")
            data.save(config.save_path + "%s_data.jpg" % k)#red line on the original image
            img.save(config.save_path + "%s_pred.jpg" % k)#prediction result

def evaluate_model(model, test_loader, device, criterion):
    model.eval()
    i = 0
    precision = 0.0
    recall = 0.0
    test_loss = 0
    correct = 0
    error=0
    with torch.no_grad():
        for sample_batched in test_loader:
            i+=1
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            output, feature = model(data)
            pred = output.max(1, keepdim=True)[1]  # 返回两个，一个是最大值另一个是最大值索引
            img = torch.squeeze(pred).cpu().numpy()*255
            lab = torch.squeeze(target).cpu().numpy()*255
            img = img.astype(np.uint8)
            lab = lab.astype(np.uint8)
            kernel = np.uint8(np.ones((3, 3)))

            #accuracy
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            #precision,recall,f1
            label_precision = cv2.dilate(lab, kernel)
            pred_recall = cv2.dilate(img, kernel)
            img = img.astype(np.int32)
            lab = lab.astype(np.int32)
            label_precision = label_precision.astype(np.int32)
            pred_recall = pred_recall.astype(np.int32)
            a = len(np.nonzero(img*label_precision)[1])
            b = len(np.nonzero(img)[1])
            if b==0:
                error=error+1
                continue
            else:
                precision += float(a/b)
            c = len(np.nonzero(pred_recall*lab)[1])
            d = len(np.nonzero(lab)[1])
            if d==0:
                error = error + 1
                continue
            else:
                recall += float(c / d)
            F1_measure=(2*precision*recall)/(precision+recall)

    test_loss /= (len(test_loader.dataset) / args.test_batch_size)
    test_acc = 100. * int(correct) / (len(test_loader.dataset) * config.label_height * config.label_width)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)'.format(
        test_loss, int(correct), len(test_loader.dataset), test_acc))

    precision = precision / (len(test_loader.dataset) - error)
    recall = recall / (len(test_loader.dataset) - error)
    F1_measure = F1_measure / (len(test_loader.dataset) - error)
    print('Precision: {:.5f}, Recall: {:.5f}, F1_measure: {:.5f}\n'.format(precision,recall,F1_measure))


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

    #turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load data for batches, num_workers for multiprocess
    if args.model == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
        test_loader=torch.utils.data.DataLoader(
            RoadSequenceDatasetList(file_path=config.test_path, transforms=op_tranforms),
            batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    else:
        test_loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=config.test_path, transforms=op_tranforms),
            batch_size=args.test_batch_size, shuffle=False, num_workers=1)

    # load model and weights
    model = generate_model(args)
    class_weight = torch.Tensor(config.class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

    pretrained_dict = torch.load(config.pretrained_path)
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)

    # output the result pictures
    output_result(model, test_loader, device)
    # calculate the values of accuracy, precision, recall, f1_measure
    evaluate_model(model, test_loader, device, criterion)
