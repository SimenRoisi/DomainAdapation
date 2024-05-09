import torch.nn as nn
import torch.nn.functional as F

class DomainClassifier(nn.Module):
    def __init__(self, input_features):
        super(DomainClassifier, self).__init__()
        
        # Defining layers
        self.fc1 = nn.Linear(input_features, 256)
        self.ln1 = nn.LayerNorm(256)  
        self.fc2 = nn.Linear(256, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.02262634706287875)
        self.leaky_relu = nn.LeakyReLU(0.07237411984130354)

        # Initialize weights using He initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)  # No activation here since BCEWithLogitsLoss will be used
        return x
