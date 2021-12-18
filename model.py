from map_to_sequence import MapToSequence
from torch import nn
import timm


class CRNN(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_output_width=7, encoder_output_channels=512, sequence_length=10, hidden_size=256, bidirectional=False, num_layers=3, dropout=0.1, batch_first=False, in_channels=3, num_classes=26):
        super(CRNN, self).__init__()
        self.feature_extractor = timm.create_model(encoder_name, pretrained=True, in_chans=in_channels, features_only=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 1), stride=1)
        self.map_to_sequence = MapToSequence(in_features=encoder_output_width, out_features=sequence_length, batch_first=batch_first)
        
        self.rnn = nn.LSTM(input_size=encoder_output_channels, 
                           hidden_size=hidden_size, 
                           bidirectional=bidirectional, 
                           num_layers=num_layers, 
                           dropout=dropout, 
                           batch_first=batch_first)
        
        self._weights_init(self.rnn)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=(self.rnn.hidden_size * (int(self.rnn.bidirectional)+1)), out_features=self.rnn.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.rnn.hidden_size, out_features=num_classes),
            nn.LogSoftmax(dim=2),
        )
        
        self.batch_first = batch_first
        
    def _weights_init(self, module):
        if isinstance(module, nn.LSTM):
            nn.init.xavier_normal_(module.weight_ih_l0)
            nn.init.xavier_normal_(module.weight_hh_l0)
            if module.bidirectional:
                nn.init.xavier_normal_(module.weight_ih_l0_reverse)
                nn.init.xavier_normal_(module.weight_hh_l0_reverse)
        
    def forward(self, x):        
        o = self.feature_extractor(x)[-1] 
        o = self.max_pool(o) 
        o = self.map_to_sequence(o)
        o, h_n = self.rnn(o)
        
        if not self.batch_first:
            o = o.permute(1, 0, 2)
            
        o = self.classifier(o)
        if not self.batch_first:
            o = o.permute(1, 0, 2)

        return o