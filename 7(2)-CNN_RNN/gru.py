import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # 입력 x,t-1의 h에서 3개 게이트를 계산(리셋, 업데이트, 후보)
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=True)


    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        x2h = self.x2h(x)       # (batch_size, 3*hidden_size)
        h2h = self.h2h(h)       # (batch_size, 3*hidden_size)
        x_r, x_z, x_n = x2h.chunk(3, dim=-1)
        h_r, h_z, h_n = h2h.chunk(3, dim=-1)
        
        # 리셋 게이트, 업데이트 게이트 계산
        r = torch.sigmoid(x_r + h_r)  
        z = torch.sigmoid(x_z + h_z)  
        
        # 후보 hidden state 계산
        n = torch.tanh(x_n + r * h_n)
        
        # 최종 hidden state
        h_new = (1 - z) * n + z * h
        return h_new



class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, seq_length, _ = inputs.size()
        # 초기 hidden state init
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        for t in range(seq_length):
            x_t = inputs[:, t, :]  # (batch_size, d_model)
            h = self.cell(x_t, h)
        return h