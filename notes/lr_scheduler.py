import torch
import numpy as np
import matplotlib.pyplot as plt

'''
    不同学习率测试
'''

if __name__ == '__main__':
    net = torch.nn.Conv2d(3, 64, 3, 1)
    epoch = 100
    x = np.arange(1, 101)
    '''
        ExpLR
    '''
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    expLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    lrs = []
    for i in range(epoch):
        lr = optimizer.param_groups[0]['lr']
        optimizer.step() # in oreder to display warning information
        # print('lr:', lr)
        expLR.step()
        lrs.append(lr)
    lrs = np.array(lrs)
    plt.plot(x, lrs,'r--', label='expLR:gamma=0.9')

    '''
        StepLR
    '''
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    lrs = []
    for i in range(epoch):
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        # print('lr:', lr)
        stepLR.step()
        lrs.append(lr)
    lrs = np.array(lrs)
    plt.plot(x, lrs,'b--', label='stepLR:gamma=0.9,step_size=1')

    '''
        multiStepLR
    '''
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    multiStepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,20,40,80], gamma=0.6)
    lrs = []
    for i in range(epoch):
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        # print('lr:', lr)
        multiStepLR.step()
        lrs.append(lr)
    lrs = np.array(lrs)
    plt.plot(x, lrs,'g--', label='multiStepLR:gamma=0.6')
    

    '''
        CosineAnnealingLR
    '''
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    cosineStepLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
    lrs = []
    for i in range(epoch):
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        # print('lr:', lr)
        cosineStepLR.step()
        lrs.append(lr)
    lrs = np.array(lrs)
    plt.plot(x, lrs,'r+', label='cosineStepLR:T=30')
    plt.legend()
    plt.savefig('LR.png')

    '''
        customization:
            1、different lr for different module
            2、different lr scheduler for different module
    '''
    
    plt.cla()
    layer1 = torch.nn.Conv2d(3, 96, 3)
    layer2 = torch.nn.Conv2d(96, 96, 3)
    net = torch.nn.Sequential(
        layer1,
        layer2
    )
    # construct trainable params
    trainable_params = []
    trainable_params += [{'params': net[0].parameters(),
                            'lr': 0.01}]
    trainable_params += [{'params': net[1].parameters(),
                            'lr': 0.05}]
    # optimizer
    optimizer = torch.optim.SGD(trainable_params, momentum=0.99, weight_decay=0.0001)
    # lr scheduler
    lr_lambda1 = lambda epoch: 0.1 ** (epoch // 10)
    lr_lambda2 = lambda epoch: 0.9 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=[lr_lambda1, lr_lambda2]
    )

    layer1_lrs, layer2_lrs = [], []
    for epoch in range(epoch):
        optimizer.step()
        layer1_lrs.append(scheduler.get_last_lr()[0])
        layer2_lrs.append(scheduler.get_last_lr()[1])
        scheduler.step()

    # visualize

    plt.plot(x, layer1_lrs, 'r--', label='layer1:stepLr')
    plt.plot(x, layer2_lrs, 'b--', label='layer2:expLr')
    plt.legend()
    plt.savefig('lambdaLr.png')