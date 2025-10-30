
import torch.nn as nn
import torch.nn.functional as F
class ChessNet(nn.Module):
    #as far as i understand, nn.module need to have 2 methods: __init__ and forward
    #__init__ is for defining layers
    def __init__(self) -> None:
        super().__init__() #always need to call super().__init__()
        self.conv1 = nn.Conv2d(17, 20, 5) #will return (20, 4, 4)
        self.value_head = nn.Linear(320, 1) #so we need to flatten (20,4,4) to (320)
        self.policy_head = nn.Linear(320, 100)
        self.tan1 = nn.Tanh()

    #forward is for defining how data flows through the network
    def forward(self, x):
        x = self.conv1(x)
        common_features = x.view(-1, 320)  # (BatchSize, 320)
        value = self.value_head(common_features)  # (BatchSize, 1)
        self.tan1(value)
        policy = self.policy_head(common_features)  # (BatchSize, 100)
        return value, policy #IMPORTANT: return both value and policy. at first i forgot to return policy and it was a pain to debug
