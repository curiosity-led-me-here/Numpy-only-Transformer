import torch
import torch.nn.functional as F
import math

class torch_GPT:
    def __init__(self, dim, head_size, vocab_size, X, Y_in, targets, lr=1e-3):
        if dim % head_size != 0:
            raise ValueError("Please keep dim size perfectly divisible by the head_size")
        
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y_in = torch.tensor(Y_in, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
        self.vocab_size = vocab_size
        self.inner_dim = dim
        self.head_size = head_size
        self.dk = dim // head_size
        self.learning_rate = lr

        scale = 1.0 / math.sqrt(dim)

        self.embed = (torch.randn(vocab_size, dim) * scale).requires_grad_()

        self.WxQ = (torch.randn(head_size, dim, self.dk) * scale).requires_grad_()
        self.WxK = (torch.randn(head_size, dim, self.dk) * scale).requires_grad_()
        self.WxV = (torch.randn(head_size, dim, self.dk) * scale).requires_grad_()
        self.WxZ = (torch.randn(dim, dim) * scale).requires_grad_()

        self.WyQ = (torch.randn(head_size, dim, self.dk) * scale).requires_grad_()
        self.WyK = (torch.randn(head_size, dim, self.dk) * scale).requires_grad_()
        self.WyV = (torch.randn(head_size, dim, self.dk) * scale).requires_grad_()
        self.WYZ = (torch.randn(dim, dim) * scale).requires_grad_()

        self.WcQ = (torch.randn(head_size, dim, self.dk) * scale).requires_grad_()
        self.WcK = (torch.randn(head_size, dim, self.dk) * scale).requires_grad_()
        self.WcV = (torch.randn(head_size, dim, self.dk) * scale).requires_grad_()
        self.WyZ = (torch.randn(dim, dim) * scale).requires_grad_()

        ff_dim = dim * 4
        self.WXf1 = (torch.randn(dim, ff_dim) * scale).requires_grad_()
        self.BXf1 = torch.zeros(1, ff_dim, requires_grad=True)
        self.WXf2 = (torch.randn(ff_dim, dim) * scale).requires_grad_()
        self.BXf2 = torch.zeros(1, dim, requires_grad=True)

        self.WYf1 = (torch.randn(dim, ff_dim) * scale).requires_grad_()
        self.BYf1 = torch.zeros(1, ff_dim, requires_grad=True)
        self.WYf2 = (torch.randn(ff_dim, dim) * scale).requires_grad_()
        self.BYf2 = torch.zeros(1, dim, requires_grad=True)

        def init_ln():
            return torch.ones(1, dim, requires_grad=True), torch.zeros(1, dim, requires_grad=True)

        self.MHA_alpha, self.MHA_beta = init_ln()             # Encoder Self-Attention
        self.Ot_alpha, self.Ot_beta = init_ln()               # Encoder FFN
        
        self.mha_alpha_y, self.mha_beta_y = init_ln()         # Decoder Self-Attention
        self.mha_cross_alpha, self.mha_cross_beta = init_ln() # Decoder Cross-Attention
        self.Of_alpha, self.Of_beta = init_ln()               # Decoder FFN

        self.W_out = (torch.randn(dim, vocab_size) * scale).requires_grad_()
        self.B_out = torch.zeros(1, vocab_size, requires_grad=True)

        self.all_params = [
            v for k, v in self.__dict__.items()
            if torch.is_tensor(v) and v.requires_grad
        ]

    def positional_encoding(self, seq_len, device):
        pos = torch.arange(seq_len, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.inner_dim, 2, device=device) * (-math.log(10000.0) / self.inner_dim))
        pe = torch.zeros(seq_len, self.inner_dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def layernorm(self, x, alpha, beta):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + 1e-5)
        return alpha * xhat + beta

    def ffn(self, x, W1, B1, W2, B2):
        # Upgraded to GELU activation!
        x = F.gelu(x @ W1 + B1)
        return x @ W2 + B2

    def mha(self, Q_in, K_in, V_in, WQ, WK, WV, WO, causal=False):
        # Fully unified attention block
        Q = torch.einsum("sd, hdk -> hsk", Q_in, WQ)
        K = torch.einsum("od, hdk -> hok", K_in, WK) # 'o' is Key/Value seq length
        V = torch.einsum("od, hdk -> hok", V_in, WV)

        scores = torch.einsum("hsk, hok -> hso", Q, K) / math.sqrt(self.dk)

        if causal:
            seq_len = Q_in.shape[0]
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("hso, hok -> hsk", attn, V)

        out = out.transpose(0, 1).reshape(Q_in.shape[0], self.inner_dim)
        return out @ WO

    def forward_X(self, x_tokens):
        # Embed and apply positional encoding (scaling embed preserves variance)
        x = self.embed[x_tokens] * math.sqrt(self.inner_dim)
        x = x + self.positional_encoding(len(x_tokens), x.device)

        # Self-Attention
        attn = self.mha(x, x, x, self.WxQ, self.WxK, self.WxV, self.WxZ, causal=False)
        x = self.layernorm(x + attn, self.MHA_alpha, self.MHA_beta)

        # FFN
        ff = self.ffn(x, self.WXf1, self.BXf1, self.WXf2, self.BXf2)
        x = self.layernorm(x + ff, self.Ot_alpha, self.Ot_beta)
        return x

    def forward_Y(self, y_tokens, enc):
        y = self.embed[y_tokens] * math.sqrt(self.inner_dim)
        y = y + self.positional_encoding(len(y_tokens), y.device)

        # Masked Self-Attention
        self_attn = self.mha(y, y, y, self.WyQ, self.WyK, self.WyV, self.WYZ, causal=True)
        y = self.layernorm(y + self_attn, self.mha_alpha_y, self.mha_beta_y)

        # Cross-Attention (Q from Decoder, K/V from Encoder)
        cross = self.mha(y, enc, enc, self.WcQ, self.WcK, self.WcV, self.WyZ, causal=False)
        y = self.layernorm(y + cross, self.mha_cross_alpha, self.mha_cross_beta)

        # FFN
        ff = self.ffn(y, self.WYf1, self.BYf1, self.WYf2, self.BYf2)
        y = self.layernorm(y + ff, self.Of_alpha, self.Of_beta)
        return y

    def full_forward(self):
        enc = self.forward_X(self.X)
        dec = self.forward_Y(self.Y_in, enc)

        self.logits = dec @ self.W_out + self.B_out
        
        return torch.softmax(self.logits, dim=-1)

    def backward(self):
        loss = F.cross_entropy(self.logits, self.targets)
        loss.backward()
        return loss.item()

    def learn(self):
        with torch.no_grad():
            for p in self.all_params:
                if p.grad is not None:
                    p -= self.learning_rate * p.grad
                    p.grad.zero_()