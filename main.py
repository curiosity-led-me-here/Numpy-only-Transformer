import numpy as np

class GPT():
    def __init__(self, dim, head_size, vocab_size, X, Y):
        if dim % head_size != 0:
            raise ValueError("Please keep dim size perfectly divisible by the head_size")
        #tokenized datasets (int, int....)
        self.X = X
        self.Y = Y
        # hyperparams
        self.vocab_size = vocab_size
        self.head_size = head_size  # number of heads
        self.inner_dim = dim
        # embeddings
        self.embed = np.random.randn(vocab_size, dim)
        # pos emb
        self.pos = np.random.randn(len(self.X), dim)
        # layernorm params
        self.MHA_alpha, self.MHA_beta, self.Ot_alpha, self.Ot_beta, self.mha_cross_alpha, self.mha_cross_beta, self.Of_alpha, self.Of_beta  = np.random.randn(8, 1, self.inner_dim)
        # X-head
        self.WxQ, self.WxK, self.WxV = np.random.randn(3, self.head_size, self.inner_dim, int(self.inner_dim / self.head_size))
        # Y-head
        self.WyQ, self.WyK, self.WyV = np.random.randn(3, self.head_size, self.inner_dim, int(self.inner_dim / self.head_size))
        
        '''
        Important concept: Initially I was slicing the input vector into heads and calculating separate heads with same static QKV weights. That is not the intent of attention. We need diversity.
        So intead, we initialize different weights for separate heads, BROADCAST ENTIRE INPUT VECTOR INTO SMALLER DIMENSION (nxd) @ (d/dk) = (nxdk) for every head then apply these separate weights onto every head and loop them and collect scores then concatenate eveything to reconstruct the entire input vector again
        Core concept is using ENTIRE INPUT VECTOR and projecting it into a lower dimensional manifold.
        '''
        
        # MHA
        self.WxZ, self.WyZ = np.random.randn(2, self.inner_dim, self.inner_dim)
        # FFN-X
        self.WXf1, self.WXf2 = np.random.randn(2, self.inner_dim, self.inner_dim)
        self.BXf1, self.BXf2 = np.random.randn(2, 1, self.inner_dim)
        # FFN-Y
        self.WYf1, self.WYf2 = np.random.randn(2, self.inner_dim, self.inner_dim)
        self.BYf1, self.BYf2 = np.random.randn(2, 1, self.inner_dim)

        # Logit projection
        self.W_out = np.random.randn(self.inner_dim, self.vocab_size)
        self.B_out = np.random.randn(1, self.vocab_size)

    def positional_encode(self, X):
        pos = np.array([[np.sin(i / 10000 ** ((2 * (Npos // 2)) / self.inner_dim)) if Npos % 2 == 0 else np.cos(i / 10000 ** ((2 * (Npos // 2)) / self.inner_dim)) for Npos in range(self.inner_dim)] for i in range(len(X))])
        return X*np.sqrt(self.inner_dim) + pos

    def attention(self, Z, WQ, WK, WV):
         Zr = []
         for i in range(self.head_size):
            Qx = Z @ WQ[i]
            Kx = Z @ WK[i]
            Vx = Z @ WV[i]
            score = self.softmax((Qx @ Kx.T) / np.sqrt(self.head_size)) @ Vx
            Zr.append(score)
         return np.concatenate(Zr, axis=1)
    
    def cross_attention(self, Z, O, WQ, WK, WV):
         Zr = []
         for i in range(self.head_size):
            Qx = Z @ WQ[i]
            Kx = O @ WK[i]
            Vx = O @ WV[i]
            score = self.softmax((Qx @ Kx.T) / np.sqrt(self.head_size)) @ Vx
            Zr.append(score)
         return np.concatenate(Zr, axis=1)

    def softmax(self, x):
        x_max = np.max(x, axis=-1, keepdims=True) 
        exp = np.exp(x - x_max) 
    
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def layernorm(self, residue, alpha, beta):
        avg = np.average(residue, axis=1, keepdims=True)
        std_dev = np.std(residue, axis=1, keepdims=True)
        norm = (residue - avg) / (std_dev + 1e-6)
        return alpha * norm + beta
    
    def ffn(self, X, W1, B1, W2, B2):
        return np.tanh(X @ W1 + B1) @ W2 + B2 

    def forward_X(self, X_input, embed, inner_dim, WQ, WK, WV, WZ, MHA_alpha, MHA_beta, Wf1, Wf2, Bf1, Bf2, Ot_alpha, Ot_beta):
        X = embed[X_input] / np.sqrt(inner_dim) # n x d
        # -- positional encoding
        Zx = self.positional_encode(X)
        # -- multi head attention
        MHAx = self.attention(Zx, WQ, WK, WV) @ WZ
        # -- add + layernorm
        mha_residue = MHAx + X
        LNx = self.layernorm(residue=mha_residue, alpha=MHA_alpha, beta=MHA_beta)
        # -- feedforward
        FFNx = self.ffn(X=LNx, W1=Wf1, W2=Wf2, B1=Bf1, B2=Bf2)
        # -- add + layernorm
        ffn_residue = LNx + FFNx
        Ot = self.layernorm(residue=ffn_residue, alpha=Ot_alpha, beta=Ot_beta)
        return Ot

    def forward_Y(self, Y_input, X_output, embed, inner_dim, WQ, WK, WV, WZ, MHA_alpha, MHA_beta):
        Y = embed[Y_input] / np.sqrt(inner_dim) # n x d
        Zy = self.positional_encode(Y)
        MHAy = self.attention(Zy, WQ, WK, WV) @ WZ
        mha_residue = MHAy + Y
        Y2 = self.layernorm(residue=mha_residue, alpha=MHA_alpha, beta=MHA_beta)

        # cross attention
        MHA_cross = self.cross_attention(Z=Y2, O=X_output, WQ=WQ, WK=WK, WV=WV) @ self.WyZ
        # add + layernorm
        OF_residue = MHA_cross + Y2
        Y3 = self.layernorm(residue=OF_residue, alpha=self.mha_cross_alpha, beta=self.mha_cross_beta)
        # feed-forward
        FFNy = self.ffn(Y3, W1=self.WYf1, W2=self.WYf2, B1=self.BYf1 , B2=self.BYf2)
        # add + layernorm
        final_residue = Y3 + FFNy
        OF = self.layernorm(residue=final_residue, alpha=self.Of_alpha, beta=self.Of_beta)
        return OF

    def full_forward(self):
        X_output = self.forward_X(X_input=self.X, embed=self.embed, inner_dim=self.inner_dim, WQ=self.WxQ, WK=self.WxK, WV=self.WxV, WZ=self.WxZ, MHA_alpha=self.MHA_alpha, MHA_beta=self.MHA_beta, Wf1=self.WXf1, Wf2=self.WXf2, Bf1=self.BXf1, Bf2=self.BXf2, Ot_alpha=self.Ot_alpha, Ot_beta=self.Ot_beta)
        Y_output = self.forward_Y(Y_input=self.Y, X_output=X_output, embed=self.embed, inner_dim=self.inner_dim, WQ=self.WyQ, WK=self.WyK, WV=self.WyV, WZ=self.WyZ, MHA_alpha=self.mha_cross_alpha, MHA_beta=self.mha_cross_beta)
        logits = Y_output @ self.W_out + self.B_out
        prediction = self.softmax(logits)
        return prediction

