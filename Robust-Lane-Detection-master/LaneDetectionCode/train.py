import torch
from model import *
from config import *
import torchvision
from torch.utils.data import DataLoader
from data_set import *
import numpy as np

if __name__ == '__main__':
    
    device = torch.device("cuda")
    class_weights=torch.FloatTensor(class_weight).cuda()

    img_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) # convert image into pytorch tensor
    train_dataset = DataLoader(tvtDatasetList(file_path=TRAIN_PATH, transforms=img_to_tensor),batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    validation_dataset = DataLoader(tvtDatasetList(file_path=VALIDATION_PATH, transforms=img_to_tensor), batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    model = STRNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)   
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    for epoch in range(EPOCH):
        print("Train Epoch = {}".format(epoch))
        ## Training
        model.train()
        for i, mini_batch in enumerate(train_dataset):
            images = mini_batch['data'].to(device)
            truth = mini_batch['label'].type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, truth)
            pred = output.max(1, keepdim=True)[1]
            print(pred.size())
            pred = torch.tensor_split(pred, BATCH_SIZE, dim=0)
            for j in range(BATCH_SIZE):
                img = torch.squeeze(pred[j]).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255
                img = Image.fromarray(img.astype(np.uint8))
                img.save(SAVE_PATH + "%s_pred.jpg" % j)
            loss.backward()
            optimizer.step()
            print("Batch Index = {}".format(i))
            print("Loss = {}".format(loss))
        ## Validation
        model.eval()
        with torch.no_grad():
            loss = 0
            count = 0
            pixels = 0
            for mini_batch in validation_dataset:
                count += 1
                images = mini_batch['data'].to(device)
                truth = mini_batch['label'].type(torch.LongTensor).to(device)
                output = model(images)

                # pred = output.max(1, keepdim=True)[1]
                # img = torch.squeeze(truth).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255
                # img = Image.fromarray(img.astype(np.uint8))
                # img.save(SAVE_PATH + "%s_pred.jpg" % count)

                loss += loss_function(output, truth).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]
                pixels += pred.eq(truth.view_as(pred)).sum().item()
        loss /= count
        accuracy = 100. * int(pixels) / (count * 128 * 256)
        print("Loss = {}".format(loss))
        print("Accuracy = {}".format(accuracy))
        scheduler.step()
