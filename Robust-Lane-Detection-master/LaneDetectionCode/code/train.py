from model import *


if __name__ == '__main__':
    model = STRNN().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.008)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    

    pass
