import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.hidden_size = 32
        self.num_layers = 1

        # CNN
        self.conv1 = nn.Sequential(
                            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
                            nn.ReLU(),
                            nn.MaxPool1d(2, 2)
                        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2, 2)
        # )
        # attention 机制
        # self.Linear_k = nn.Linear(65, 64)
        # self.Linear_q = nn.Linear(65, 64)
        # self.Linear_v = nn.Linear(65, 64)
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=64, num_heads=2)

        # 全连接层
        self.fc = nn.Sequential(
                            nn.Linear(in_features=3008, out_features=128),
                            nn.Sigmoid(),
                            nn.Linear(in_features= 128, out_features=2)

        )

    def forward(self, x):
        # x:[64, 200] [batch_size, seq_len]
        x=x.float()
        # x=abs(x)
        # out = self.embed(x)  # [64, 200, 300] [batch, seq_len, hidden_size]


        # out, _ = self.lstm(x)
        out = torch.unsqueeze(x, dim=1)

        out = self.conv1(out)

        out = self.conv2(out)
        # out = self.conv3(out)

        # 开始 多头attention机制------------
        # att_k= self.Linear_k(out)
        # att_q= self.Linear_q(out)
        # att_v= self.Linear_v(out)
        #
        # out, _ = self.multihead_attn(att_k, att_q, att_v)

        out = out.flatten(start_dim=1)

        logits = self.fc(out)  # [64, 10] [batch, num_classes]
        return logits



if __name__ == "__main__":
    """ for test """
    data = torch.rand((16,198))

    model = MyModel()    ## vocab_size, embedding_dim, hidden_size, num_layers, num_classes
    output = model(data)
    print(f'output shape:{output.shape}')
