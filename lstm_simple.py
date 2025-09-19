import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # LSTM 셀: input_size → hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 출력 레이어: hidden_size → output_size
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # input: (batch, seq_len, input_size)
        # hidden: (h_0, c_0) 튜플
        output, hidden = self.lstm(input, hidden)  
        # output: (batch, seq_len, hidden_size)

        # 마지막 시점의 hidden state만 사용
        out = self.h2o(output[:, -1, :])  
        return out, hidden

    def get_hidden(self, batch_size=1, device="cpu"):
        # h_0, c_0 둘 다 초기화해야 함
        h_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)