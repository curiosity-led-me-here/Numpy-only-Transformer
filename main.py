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
        self.dk = dim // head_size
        self.avg_dim = 1 / self.inner_dim
        self.avg_kdim = 1/ self.dk
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
            score = self.softmax((Qx @ Kx.T) / np.sqrt(self.dk)) @ Vx
            Zr.append(score)
         return np.concatenate(Zr, axis=1)
    
    def cross_attention(self, Z, O, WQ, WK, WV):
         Zr = []
         for i in range(self.head_size):
            Qx = Z @ WQ[i]
            Kx = O @ WK[i]
            Vx = O @ WV[i]
            score = self.softmax((Qx @ Kx.T) / np.sqrt(self.dk)) @ Vx
            Zr.append(score)
         return np.concatenate(Zr, axis=1)

    def softmax(self, x):
        x_max = np.max(x, axis=-1, keepdims=True) 
        exp = np.exp(x - x_max) 
    
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def layernorm(self, residue, alpha, beta):
        avg = np.average(residue, axis=1, keepdims=True)
        std_dev = np.std(residue, axis=1, keepdims=True)
        gamma = (residue - avg) / (std_dev + 1e-6)
        return alpha * gamma + beta, std_dev, gamma, avg
    
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
        LNx, self.stdev_X, self.gamma_X, self.avg_X = self.layernorm(residue=mha_residue, alpha=MHA_alpha, beta=MHA_beta)
        # -- feedforward
        FFNx = self.ffn(X=LNx, W1=Wf1, W2=Wf2, B1=Bf1, B2=Bf2)
        # -- add + layernorm
        ffn_residue = LNx + FFNx
        Ot, self.stdev_Ot, self.gamma_Ot, self.avg_Ot = self.layernorm(residue=ffn_residue, alpha=Ot_alpha, beta=Ot_beta)
        return Ot

    def forward_Y(self, Y_input, X_output, embed, inner_dim, WQ, WK, WV, WZ, MHA_alpha, MHA_beta):
        Y = embed[Y_input] / np.sqrt(inner_dim) # n x d
        Zy = self.positional_encode(Y)
        MHAy = self.attention(Zy, WQ, WK, WV) @ WZ
        mha_residue = MHAy + Y
        Y2, self.stdev_Y2, self.gamma_Y2, self.avg_Y2 = self.layernorm(residue=mha_residue, alpha=MHA_alpha, beta=MHA_beta)

        # cross attention
        MHA_cross = self.cross_attention(Z=Y2, O=X_output, WQ=WQ, WK=WK, WV=WV) @ self.WyZ
        # add + layernorm
        self.OF_residue = MHA_cross + Y2
        self.Y3, self.stdev_Y3, self.gamma_Y3, self.avg_Y3 = self.layernorm(residue=self.OF_residue, alpha=self.mha_cross_alpha, beta=self.mha_cross_beta)
        # feed-forward
        FFNy = self.ffn(self.Y3, W1=self.WYf1, W2=self.WYf2, B1=self.BYf1 , B2=self.BYf2)
        # add + layernorm
        self.final_residue = self.Y3 + FFNy
        OF, self.stdev_OF, self.gamma_OF, self.avg_OF = self.layernorm(residue=self.final_residue, alpha=self.Of_alpha, beta=self.Of_beta)
        return OF

    def full_forward(self):
        self.X_output = self.forward_X(X_input=self.X, embed=self.embed, inner_dim=self.inner_dim, WQ=self.WxQ, WK=self.WxK, WV=self.WxV, WZ=self.WxZ, MHA_alpha=self.MHA_alpha, MHA_beta=self.MHA_beta, Wf1=self.WXf1, Wf2=self.WXf2, Bf1=self.BXf1, Bf2=self.BXf2, Ot_alpha=self.Ot_alpha, Ot_beta=self.Ot_beta)
        self.Y_output = self.forward_Y(Y_input=self.Y, X_output=self.X_output, embed=self.embed, inner_dim=self.inner_dim, WQ=self.WyQ, WK=self.WyK, WV=self.WyV, WZ=self.WyZ, MHA_alpha=self.mha_cross_alpha, MHA_beta=self.mha_cross_beta)
        self.logits = self.Y_output @ self.W_out + self.B_out
        self.prediction = self.softmax(self.logits)
        return self.prediction

    def deriv_layernorm(self, X, dY: np.ndarray, stdev, gamma: np.ndarray, alpha, mu, avg_d):
        stdev += 1e-4
        sum_dY = dY.sum(axis=1, keepdims=True)  # n, 1
        accum_gamma = np.sum((dY * gamma), axis=1, keepdims=True)   # n, 1
        common_factor = alpha / stdev   # 1,d / n,1 --> n,d
        main_gamma = (X - mu) / stdev   # n,d - n,1 / n,1 --> n,d
        factor1 = dY # n,d
        factor2 = avg_d * sum_dY # n,1
        factor3 = avg_d * main_gamma * accum_gamma # nxd
        return common_factor * (factor1 - factor2 - factor3) # n,d


    def deriv_FFN(self, dY, X, W1, b1, W2):
        Z2 = np.tanh(X @ W1 + b1)    # n, d_hid       
        deriv_Z2 = (1 - Z2**2)
        dW2 = Z2.T @ dY  # d_hidden,n x n,d_out --> d_hidden, d_out
        db2 = np.sum(dY, axis=0, keepdims=True)
        dh = (dY @ W2.T) * deriv_Z2       # (n,d_out x d_out,d_hid) --> n,d_hid * n,d_hid = n,d_hid
        db1 = np.sum(dh, axis=0, keepdims=True)
        dW1 = X.T @ dh    # d_in,n x n,d_hid --> d_in,d_hid
        dX = dh @ W1.T    # n,d_hid x d_in,d_hid --> n,d_in
        return dW2, db2, dW1, db1, dX
    
    def backward(self): 
        d_logits = self.avg_dim * (self.prediction - np.eye(self.vocab_size)[self.Y])   # n,v
        dY_out = d_logits @ self.W_out.T    # n,v * v,d --> n,d
        dW_out = self.Y_output.T @ d_logits     # d,n x n,v --> d,v
        dB_out = np.sum(d_logits, axis=0, keepdims=True)    # n,v --> n,1
        dzi_OF = self.deriv_layernorm(X=self.final_residue, dY=dY_out, stdev=self.stdev_OF, gamma=self.gamma_OF, alpha=self.Of_alpha, mu=self.avg_OF, avg_d=self.avg_dim)   # n,d
        dW2_y3, db2_y3, dW1_y3, db1_y3, dY3 = self.deriv_FFN(dY=dzi_OF, X=self.Y3, W1=self.WYf1, b1=self.BYf1, W2=self.BYf2)
        dY3 += dzi_OF # this is due to the residual connection (Y3 + FFN) whose derivative was 1 w.r.t Y3 itself. Therefore adding both paths together gives (1+dFFN)*dY = dY + dFFN where dY = dzi_OF
        
        return dW_out, dB_out, dzi_OF, dW1_y3, dW2_y3, db1_y3,db2_y3
        