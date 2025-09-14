import os
import sys
sys.path.append('../..')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchsparse.utils.collate import sparse_collate_fn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import *

from core.pconlynet import FusionNet, get_loss
from core.dataset import SceneDataset


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        # torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        # torch.nn.init.constant_(m.bias.data, 0.0)


def main():
    classes = [
        'office building',
        'industrial area',
        'greenland',
        'residential',
        'transport',
        'sport field',
        'farmland',
    ]

    train_dataset = SceneDataset("/home/ExtraData/SceneClass/Data/train_data", 0.005)
    test_dataset = SceneDataset("/home/ExtraData/SceneClass/Data/test_data", 0.005)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=16,
        shuffle=True,
        drop_last=True,
        collate_fn=sparse_collate_fn # Access the sparse tensors in the input list and call sparse_collate.
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, num_workers=16, shuffle=True, drop_last=True, collate_fn=sparse_collate_fn
    )

    classifier = FusionNet(num_class=7).cuda()
    # # device_ids = [0, 1] 	# id为0和1的两块显卡
    # classifier = torch.nn.DataParallel(classifier)
    classifier.apply(inplace_relu).apply(weights_init)

    criterion = get_loss()
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    writer = SummaryWriter(
        "/home/ExtraData/SceneClass/results/ex10/metrics"
    )


    best_acc = 0.0
    best_ka = 0.0

    for epoch in range(100):
        classifier = classifier.train()
        for i, train_dict in enumerate(train_dataloader):
            pc = train_dict['point'].to(device='cuda')
            # img = train_dict['image'].to(device='cuda')
            label_tensor = train_dict['label'].to(device='cuda')


            pred = classifier(pc)
            loss = criterion(pred, label_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # len(train_dataloader)等于(训练样本数/BATCH_SIZE)，dataloader本质上是一次从训练样本中拿BATCH_SIZE个样本批处理
            writer.add_scalar(
                "loss", loss.cpu().item(), ((epoch) * len(train_dataloader)) + (i + 1)
            )
            if i % 5 == 0:
                print(
                    "Epoch ",
                    epoch,
                    "Iter ",
                    i,
                    "/",
                    len(train_dataloader),
                    "Loss : ",
                    loss.cpu().item(),
                )

        classifier = classifier.eval()
        pred_list = []
        label_list = []
        for j, test_dict in enumerate(train_dataloader):
            test_pc = test_dict['point'].to(device='cuda')
            # img = test_dict['image'].to(device='cuda')
            test_label = test_dict['label'].to(device='cuda')
            test_pred = classifier(test_pc)  # [B, N]
            # torch.argmax函数返回指定维度最大值的序号dim=0表示二维中的列，dim=1在二维矩阵中表示行
            test_pred = torch.argmax(test_pred, dim=1).cpu().numpy()
            test_label = test_label.cpu().numpy()
            pred_list.extend(test_pred)
            label_list.extend(test_label)

        # print(pred_list)
        # print(label_list)

        # OA
        accuracy = accuracy_score(label_list, pred_list)
        print("OA:", accuracy)
        writer.add_scalar("Accuracy", accuracy, epoch)

        # F1-macro，宏平均
        F1_macro = f1_score(
            label_list, pred_list, labels=[0, 1, 2, 3, 4, 5, 6], average="macro"
        )
        print("F1_macro:", F1_macro)
        writer.add_scalar("F1_macro", F1_macro, epoch)

        # F1-weighted，加权平均
        F1_weighted = f1_score(
            label_list, pred_list, labels=[0, 1, 2, 3, 4, 5, 6], average="weighted"
        )
        print("F1_weighted:", F1_weighted)
        writer.add_scalar("F1_weighted", F1_weighted, epoch)

        # kappa
        kappa = cohen_kappa_score(pred_list,label_list)
        print("kappa:", kappa)
        writer.add_scalar("kappa", kappa, epoch)

        # F1-score
        _, _, f_class, _ = precision_recall_fscore_support(
            y_true=label_list,
            y_pred=pred_list,
            labels=[0, 1, 2, 3, 4, 5, 6],
            average=None,
        )
        class_f1 = {}
        for ff in range(7):
            class_f1[classes[ff]] = f_class[ff]
        print("各类单独F1:", class_f1)
        writer.add_scalars(
            main_tag="F1-score",
            tag_scalar_dict=class_f1,
            global_step=epoch,
        )

        if accuracy > best_acc:
            best_acc = accuracy
            print("!!!!!!!!!!!-Record-breaking-!!!!!!!!!!!")
            print(
                "Epoch:"
                + str(epoch)
                + ";"
                + "Acc:"
                + str(best_acc)
            )
            print("!!!!!!!!!!!-Record-breaking-!!!!!!!!!!!")
            torch.save(
                classifier.state_dict(),
                "/home/ExtraData/SceneClass/results/ex10/weights/"
                + str(epoch)
                +"_OA"
                + ".path",
            )

        if kappa > best_ka:
            best_ka = kappa
            print("!!!!!!!!!!!-Record-breaking-!!!!!!!!!!!")
            print(
                "Epoch:"
                + str(epoch)
                + ";"
                + "kappa:"
                + str(best_acc)
            )
            print("!!!!!!!!!!!-Record-breaking-!!!!!!!!!!!")
            torch.save(
                classifier.state_dict(),
                "/home/ExtraData/SceneClass/results/ex10/weights/"
                + str(epoch)
                +"_kappa"
                + ".path",
            )        
        

    torch.save(
        classifier.state_dict(),
        "/home/ExtraData/SceneClass/results/ex10/weights/final.pth",
    )

if __name__ == "__main__":
    main()
