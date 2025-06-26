import torch.nn as nn

class SynergyModel(nn.Module):
    def __init__(self, in_dim, arch='std', drop=0.3):
        super(SynergyModel, self).__init__()
        
        if arch == 'std':
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.BatchNorm1d(512),
                nn.Tanh(),
                nn.Dropout(drop),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(drop),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(drop),

                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(drop),

                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(drop),

                nn.Linear(512, 1)
            )
        elif arch == 'fold2':
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.BatchNorm1d(512),
                nn.Tanh(),
                nn.Dropout(drop),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(drop),
                
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(drop),
                
                nn.Linear(512, 1)
            )
        elif arch == 'fold3':
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.BatchNorm1d(512),
                nn.Tanh(),
                nn.Dropout(drop),
                
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(drop),
                
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(drop),
                
                nn.Linear(128, 1)
            )
    
    def forward(self, x):
        return self.net(x)