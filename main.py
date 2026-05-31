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
        self.MHA_alpha, self.MHA_beta, self.Ot_alpha, self.Ot_beta, self.mha_cross_alpha, self.mha_cross_beta, self.Of_alpha, self.Of_beta, self.mha_alpha_y, self.mha_beta_y  = np.random.randn(10, 1, self.inner_dim)
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
        self.WxZ, self.WyZ, self.WYZ = np.random.randn(3, self.inner_dim, self.inner_dim)
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
        self.X_inpt = embed[X_input] / np.sqrt(inner_dim) # n x d
        # -- positional encoding
        self.Zx = self.positional_encode(self.X_inpt)
        # -- multi head attention
        self.MHAx_concat = self.attention(self.Zx, WQ, WK, WV) 
        self.MHAx = self.MHAx_concat @ WZ
        # -- add + layernorm
        self.mha_residue_x = self.MHAx + self.X_inpt
        self.LNx, self.stdev_X, self.gamma_X, self.avg_X = self.layernorm(residue=self.mha_residue_x, alpha=MHA_alpha, beta=MHA_beta)
        # -- feedforward
        FFNx = self.ffn(X=self.LNx, W1=Wf1, W2=Wf2, B1=Bf1, B2=Bf2)
        # -- add + layernorm
        self.ffn_residue = self.LNx + FFNx
        Ot, self.stdev_Ot, self.gamma_Ot, self.avg_Ot = self.layernorm(residue=self.ffn_residue, alpha=Ot_alpha, beta=Ot_beta)
        return Ot

    def forward_Y(self, Y_input, X_output, embed, inner_dim, WQ, WK, WV, WZ, MHA_alpha, MHA_beta):
        self.Y_inpt = embed[Y_input] / np.sqrt(inner_dim) # n x d
        self.Zy = self.positional_encode(self.Y_inpt)
        self.MHAy_concat = self.attention(self.Zy, WQ, WK, WV)
        self.MHAy = self.MHAy_concat @ WZ
        self.mha_residue_y = self.MHAy + self.Y_inpt
        self.Y2, self.stdev_Y2, self.gamma_Y2, self.avg_Y2 = self.layernorm(residue=self.mha_residue_y, alpha=MHA_alpha, beta=MHA_beta)

        # cross attention
        self.MHA_cross_concat = self.cross_attention(Z=self.Y2, O=X_output, WQ=WQ, WK=WK, WV=WV)
        self.MHA_cross = self.MHA_cross_concat @ self.WyZ
        # add + layernorm
        self.OF_residue = self.MHA_cross + self.Y2
        self.Y3, self.stdev_Y3, self.gamma_Y3, self.avg_Y3 = self.layernorm(residue=self.OF_residue, alpha=self.mha_cross_alpha, beta=self.mha_cross_beta)
        # feed-forward
        FFNy = self.ffn(self.Y3, W1=self.WYf1, W2=self.WYf2, B1=self.BYf1 , B2=self.BYf2)
        # add + layernorm
        self.final_residue = self.Y3 + FFNy
        OF, self.stdev_OF, self.gamma_OF, self.avg_OF = self.layernorm(residue=self.final_residue, alpha=self.Of_alpha, beta=self.Of_beta)
        return OF

    def full_forward(self):
        self.X_output = self.forward_X(X_input=self.X, embed=self.embed, inner_dim=self.inner_dim, WQ=self.WxQ, WK=self.WxK, WV=self.WxV, WZ=self.WxZ, MHA_alpha=self.MHA_alpha, MHA_beta=self.MHA_beta, Wf1=self.WXf1, Wf2=self.WXf2, Bf1=self.BXf1, Bf2=self.BXf2, Ot_alpha=self.Ot_alpha, Ot_beta=self.Ot_beta)
        self.Y_output = self.forward_Y(Y_input=self.Y, X_output=self.X_output, embed=self.embed, inner_dim=self.inner_dim, WQ=self.WyQ, WK=self.WyK, WV=self.WyV, WZ=self.WYZ, MHA_alpha=self.mha_alpha_y, MHA_beta=self.mha_beta_y)
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
        dalpha = np.sum(dY * gamma, axis=0, keepdims=True)
        dbeta = np.sum(dY, axis=0, keepdims=True)
        return common_factor * (factor1 - factor2 - factor3), dalpha, dbeta # n,d

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

    def deriv_softmax(self, dZ, Z):
        term = np.sum(dZ * Z, axis=1, keepdims=True)
        return Z * (dZ - term)
    
    def deriv_cross_mha(self, dY, X:np.ndarray, O:np.ndarray, Wq, Wk, Wv):
        dY = np.reshape(dY, (len(dY), self.head_size, self.dk))
        dY = np.transpose(dY, (1, 0, 2))
        dWq, dWk, dWv = [], [], []
        dX_total = np.zeros_like(X) # (n, d)
        dO_total = np.zeros_like(O) 
        for head in range(self.head_size):
            Qx = X @ Wq[head]
            Kx = O @ Wk[head]
            Vx = O @ Wv[head]
            Z = self.softmax((Qx @ Kx.T) / np.sqrt(self.dk))
            dWv_head = (Z @ O).T @ dY[head]
            dmha_dZ = dY[head] @ (Vx).T # n,d x d,dk --> n,dk x dk,n --> n,n
            dZ_dx = self.deriv_softmax(dZ=dmha_dZ, Z=Z) / np.sqrt(self.dk)  # n,n
            dWk_head = (dZ_dx @ O).T @ Qx
            dWq_head = X.T @ (dZ_dx @ Kx)
            dWq.append(dWq_head)
            dWk.append(dWk_head)
            dWv.append(dWv_head)
            dQ = dZ_dx @ Kx
            dK = dZ_dx.T @ Qx
            dV = Z.T @ dY[head]
            dX_total += dQ @ Wq[head].T                  # (n, dk) @ (dk, d) --> (n, d)
            dO_total += (dK @ Wk[head].T) + (dV @ Wv[head].T) # (n, d) + (n, d)
        return dWq, dWk, dWv, dX_total, dO_total
    
    def deriv_mha(self, dY, X, Wq, Wk, Wv):
        dY = np.reshape(dY, (len(dY), self.head_size, self.dk))
        dY = np.transpose(dY, (1, 0, 2))
        dWq, dWk, dWv = [], [], []
        dX_total = np.zeros_like(X) # (n, d)
        for head in range(self.head_size):
            Qx = X @ Wq[head]
            Kx = X @ Wk[head]
            Vx = X @ Wv[head]
            Z = self.softmax((Qx @ Kx.T) / np.sqrt(self.dk))
            dmha_dZ = dY[head] @ (Vx).T # n,d x d,dk --> n,dk x dk,n --> n,n
            dZ_dx = self.deriv_softmax(dZ=dmha_dZ, Z=Z) / np.sqrt(self.dk)  # n,n
            dQ = dZ_dx @ Kx
            dK = dZ_dx.T @ Qx
            dV = Z.T @ dY[head]
            dWq.append(X.T @ dQ)
            dWk.append(X.T @ dK)
            dWv.append(X.T @ dV)
            dX_total += (dQ @ Wq[head].T) + (dK @ Wk[head].T) + (dV @ Wv[head].T)                # (n, dk) @ (dk, d) --> (n, d)
        return dWq, dWk, dWv, dX_total


    def backward(self): 
        d_logits = self.avg_dim * (self.prediction - np.eye(self.vocab_size)[self.Y])   # n,v
        dY_out = d_logits @ self.W_out.T    # n,v * v,d --> n,d
        self.dW_out = self.Y_output.T @ d_logits     # d,n x n,v --> d,v
        self.dB_out = np.sum(d_logits, axis=0, keepdims=True)    # n,v --> n,1
        self.dzi_OF, self.dalpha_OF, self.dbeta_OF = self.deriv_layernorm(X=self.final_residue, dY=dY_out, stdev=self.stdev_OF, gamma=self.gamma_OF, alpha=self.Of_alpha, mu=self.avg_OF, avg_d=self.avg_dim)   # n,d
        self.dW2_y3, self.db2_y3, self.dW1_y3, self.db1_y3, dY3 = self.deriv_FFN(dY=self.dzi_OF, X=self.Y3, W1=self.WYf1, b1=self.BYf1, W2=self.WYf2)
        dY3 += self.dzi_OF # this is due to the residual connection (Y3 + FFN) whose derivative was 1 w.r.t Y3 itself. Therefore adding both paths together gives (1+dFFN)*dY = dY + dFFN where dY = dzi_OF
        dzi_Y3, self.dalpha_Y3, self.dbeta_Y3 = self.deriv_layernorm(X=self.OF_residue, dY=dY3, stdev=self.stdev_Y3, gamma=self.gamma_Y3, alpha=self.mha_cross_alpha, mu=self.avg_Y3, avg_d=self.avg_dim)   # n,d
        self.dWYz = self.MHA_cross_concat.T @ dzi_Y3
        dY_cross = dzi_Y3 @ self.WyZ.T
        self.dWq_cross, self.dWk_cross, self.dWv_cross, dY2, dX = self.deriv_cross_mha(dY=dY_cross, X=self.Y2, O=self.X_output, Wq=self.WyQ, Wk=self.WyK, Wv=self.WyV)
        dY2 += dzi_Y3

        # PATH 1: Y2 --> encoder inception
        dyi_Y1, self.dalpha_Y1, self.dbeta_Y1 = self.deriv_layernorm(X=self.mha_residue_y, dY=dY2, stdev=self.stdev_Y2, gamma=self.gamma_Y2, alpha=self.mha_alpha_y, mu=self.avg_Ot, avg_d=self.avg_dim)   # n,d 
        self.dWyZ = self.MHAy_concat.T @ dyi_Y1
        dY_mha = dyi_Y1 @ self.WYZ.T
        self.dWq_mhay, self.dWk_mhay, self.dWv_mhay ,dY1 = self.deriv_mha(dY=dY_mha, X=self.Zy, Wq=self.WyQ, Wk=self.WyK, Wv=self.WyV)
        dY1 += dyi_Y1
        self.dembed_y = np.zeros_like(self.embed)
        np.add.at(self.dembed_y, self.Y, dY1 / np.sqrt(self.inner_dim))

        # PATH 2: dX --> decoder inception
        dzi_Ot, self.dalpha_Ot, self.dbeta_Ot = self.deriv_layernorm(X=self.ffn_residue, dY=dX, stdev=self.stdev_Ot, gamma=self.gamma_Ot, alpha=self.Ot_alpha, mu=self.avg_Ot, avg_d=self.avg_dim)   # n,d 
        self.dW2_x3, self.db2_x3, self.dW1_x3, self.db1_x3, dX3 = self.deriv_FFN(dY=dzi_Ot, X=self.LNx, W1=self.WXf1, b1=self.BXf1, W2=self.WXf2)
        dX3 += dzi_Ot
        dX2, self.dalpha_X2, self.dbeta_X2 = self.deriv_layernorm(X=self.mha_residue_x, dY=dX3, stdev=self.stdev_X, gamma=self.gamma_X, alpha=self.MHA_alpha, beta=self.MHA_beta, mu=self.avg_X, avg_d=self.avg_dim)
        self.dWXz = self.MHAx_concat.T @ dX2
        dX_mha = dX2 @ self.WxZ
        self.dWq_mhax, self.dWk_mhax, self.dWv_mhax, dX1 = self.deriv_mha(dY=dX_mha, X=self.Zx, Wq=self.WxQ, Wk=self.WxK, Wv=self.WxV)
        dX1 += dX2
        self.dembed_x = np.zeros_like(self.embed)
        np.add.at(self.dembed_x, self.X, dX1 / np.sqrt(self.inner_dim))
        return self.dW_out, self.dB_out, self.dzi_OF, self.dW1_y3, self.dW2_y3, self.db1_y3, self.db2_y3, self.dWq_cross, self.dWk_cross, self.dWv_cross, self.dWYz, self.dalpha_OF, self.dbeta_OF, self.dalpha_Y3, self.dbeta_Y3, self.dalpha_Ot, self.dbeta_Ot, self.dW2_x3, self.db2_x3, self.dW1_x3, self.db1_x3, self.dalpha_X2, self.dbeta_X2, self.dWXz, self.dalpha_Y1, self.dbeta_Y1, self.dWyZ, self.dembed_x, self.dembed_y, self.dWq_mhax, self.dWk_mhax, self.dWv_mhax, self.dWq_mhay, self.dWk_mhay, self.dWv_mhay
        
    def backward(self): 
        d_logits = self.avg_dim * (self.prediction - np.eye(self.vocab_size)[self.Y])   # n,v
        dY_out = d_logits @ self.W_out.T    # n,v * v,d --> n,d
        self.dW_out = self.Y_output.T @ d_logits     # d,n x n,v --> d,v
        self.dB_out = np.sum(d_logits, axis=0, keepdims=True)    # n,v --> n,1
        self.dzi_OF, self.dalpha_OF, self.dbeta_OF = self.deriv_layernorm(X=self.final_residue, dY=dY_out, stdev=self.stdev_OF, gamma=self.gamma_OF, alpha=self.Of_alpha, mu=self.avg_OF, avg_d=self.avg_dim)   # n,d
        self.dW2_y3, self.db2_y3, self.dW1_y3, self.db1_y3, dY3 = self.deriv_FFN(dY=self.dzi_OF, X=self.Y3, W1=self.WYf1, b1=self.BYf1, W2=self.WYf2)
        dY3 += self.dzi_OF # this is due to the residual connection (Y3 + FFN) whose derivative was 1 w.r.t Y3 itself. Therefore adding both paths together gives (1+dFFN)*dY = dY + dFFN where dY = dzi_OF
        self.dzi_Y3, self.dalpha_Y3, self.dbeta_Y3 = self.deriv_layernorm(X=self.OF_residue, dY=dY3, stdev=self.stdev_Y3, gamma=self.gamma_Y3, alpha=self.mha_cross_alpha, mu=self.avg_Y3, avg_d=self.avg_dim)   # n,d
        self.dWYz = self.MHA_cross_concat.T @ self.dzi_Y3
        dY_cross = self.dzi_Y3 @ self.WyZ.T
        self.dWq_cross, self.dWk_cross, self.dWv_cross, dY2, dX = self.deriv_cross_mha(dY=dY_cross, X=self.Y2, O=self.X_output, Wq=self.WyQ, Wk=self.WyK, Wv=self.WyV)
        dY2 += self.dzi_Y3

        # PATH 1: Y2 --> encoder inception
        dyi_Y1, self.dalpha_Y1, self.dbeta_Y1 = self.deriv_layernorm(X=self.mha_residue_y, dY=dY2, stdev=self.stdev_Y2, gamma=self.gamma_Y2, alpha=self.mha_alpha_y, mu=self.avg_Y2, avg_d=self.avg_dim)   # n,d 
        self.dWyZ = self.MHAy_concat.T @ dyi_Y1
        dY_mha = dyi_Y1 @ self.WYZ.T
        self.dWq_mhay, self.dWk_mhay, self.dWv_mhay ,dY1 = self.deriv_mha(dY=dY_mha, X=self.Zy, Wq=self.WyQ, Wk=self.WyK, Wv=self.WyV)
        dY1 += dyi_Y1
        self.dembed_y = np.zeros_like(self.embed)
        np.add.at(self.dembed_y, self.Y, dY1 / np.sqrt(self.inner_dim))

        # PATH 2: dX --> decoder inception
        self.dzi_Ot, self.dalpha_Ot, self.dbeta_Ot = self.deriv_layernorm(X=self.ffn_residue, dY=dX, stdev=self.stdev_Ot, gamma=self.gamma_Ot, alpha=self.Ot_alpha, mu=self.avg_Ot, avg_d=self.avg_dim)   # n,d 
        self.dW2_x3, self.db2_x3, self.dW1_x3, self.db1_x3, dX3 = self.deriv_FFN(dY=self.dzi_Ot, X=self.LNx, W1=self.WXf1, b1=self.BXf1, W2=self.WXf2)
        dX3 += self.dzi_Ot
        self.dX2, self.dalpha_X2, self.dbeta_X2 = self.deriv_layernorm(X=self.mha_residue_x, dY=dX3, stdev=self.stdev_X, gamma=self.gamma_X, alpha=self.MHA_alpha, mu=self.avg_X, avg_d=self.avg_dim)
        self.dWXz = self.MHAx_concat.T @ self.dX2
        dX_mha = self.dX2 @ self.WxZ.T
        self.dWq_mhax, self.dWk_mhax, self.dWv_mhax, dX1 = self.deriv_mha(dY=dX_mha, X=self.Zx, Wq=self.WxQ, Wk=self.WxK, Wv=self.WxV)
        dX1 += self.dX2
        self.dembed_x = np.zeros_like(self.embed)
        np.add.at(self.dembed_x, self.X, dX1 / np.sqrt(self.inner_dim))
        return self
    
    def learn(self, lr=1e-3):
        # Output Layer updates
        self.W_out -= lr * self.dW_out
        self.B_out -= lr * self.dB_out
        # Decoder FFN updates
        self.WYf1 -= lr * self.dW1_y3
        self.WYf2 -= lr * self.dW2_y3
        self.BYf1 -= lr * self.db1_y3
        self.BYf2 -= lr * self.db2_y3
        # Encoder FFN updates
        self.WXf1 -= lr * self.dW1_x3
        self.WXf2 -= lr * self.dW2_x3
        self.BXf1 -= lr * self.db1_x3
        self.BXf2 -= lr * self.db2_x3
        # Attention Output Projection updates
        self.WyZ -= lr * self.dWYz
        self.WxZ -= lr * self.dWXz
        self.WYZ -= lr * self.dWyZ
        # Combined Input Embedding Updates
        self.embed -= lr * (self.dembed_x + self.dembed_y)
        # Loop and Update Multi-Head Weights
        for head in range(self.head_size):
            # Cross Attention Block
            self.WyQ[head] -= lr * self.dWq_cross[head]
            self.WyK[head] -= lr * self.dWk_cross[head]
            self.WyV[head] -= lr * self.dWv_cross[head]
            # Decoder Self Attention Block
            self.WyQ[head] -= lr * self.dWq_mhay[head]
            self.WyK[head] -= lr * self.dWk_mhay[head]
            self.WyV[head] -= lr * self.dWv_mhay[head]
            # Encoder Self Attention Block
            self.WxQ[head] -= lr * self.dWq_mhax[head]
            self.WxK[head] -= lr * self.dWk_mhax[head]
            self.WxV[head] -= lr * self.dWv_mhax[head]
        # LayerNorm Parameter updates
        self.Of_alpha -= lr * self.dalpha_OF
        self.Of_beta -= lr * self.dbeta_OF
        self.mha_cross_alpha -= lr * self.dalpha_Y3
        self.mha_cross_beta -= lr * self.dbeta_Y3
        self.mha_alpha_y -= lr * self.dalpha_Y1
        self.mha_beta_y -= lr * self.dbeta_Y1
        self.Ot_alpha -= lr * self.dalpha_Ot
        self.Ot_beta -= lr * self.dbeta_Ot
        self.MHA_alpha -= lr * self.dalpha_X2
        self.MHA_beta -= lr * self.dbeta_X2