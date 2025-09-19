import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 모든 게이트를 하나의 Linear로 묶음 (i, f, o, g)
        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

        # 출력 레이어
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        x: (batch, input_size)
        hidden: (h, c) 튜플
        """
        h, c = hidden

        # 게이트 계산
        gates = self.i2h(x) + self.h2h(h)
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)

        i = torch.sigmoid(i_gate)   # 입력 게이트
        f = torch.sigmoid(f_gate)   # 망각 게이트
        o = torch.sigmoid(o_gate)   # 출력 게이트
        g = torch.tanh(g_gate)      # 후보 셀 상태

        # 셀 및 hidden 업데이트
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        # 출력
        out = self.h2o(h_next)

        return out, (h_next, c_next)

    def get_hidden(self, batch_size=1, device="cpu"):
        h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h_0, c_0)

batch_size, input_size, hidden_size, output_size = 2, 3, 5, 2
model = LSTM(input_size, hidden_size, output_size)

x = torch.randn(batch_size, input_size)  # 입력
hidden = model.get_hidden(batch_size)    # 초기 hidden, cell state

out, hidden = model(x, hidden)

print("출력:", out.shape)      # (2, 2)
print("새로운 hidden:", hidden[0].shape) # (2, 5)
print("새로운 cell:", hidden[1].shape)   # (2, 5)


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
