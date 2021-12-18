from torch import nn


class MapToSequence(nn.Module):
    def __init__(self, in_features=7, out_features=10, batch_first=False):
        super(MapToSequence, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
        )
        
        self.batch_first = batch_first
        
    def forward(self, x):
        b, c, h, w = x.size()
        assert h == 1
        x = x.squeeze(dim=2)
        o = self.layers(x)
        
        if self.batch_first:
            o = o.permute(0, 2, 1)
        else:
            o = o.permute(2, 0, 1)
        
        return o