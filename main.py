import numpy as np

class GPT():
    def __init__(self, dim, head_size, vocab_size, X, Y):
        #datasets
        self.X = X
        self.Y = Y

        # hyperparams
        self.vocab_size = vocab_size
        self.head_size = head_size
        self.inner_dim = dim
        
        # embeddings
        self.embed = np.random.rand(vocab_size, dim)
        
        # X-head
        self.WxQ, self.WxK, self.WxV = (
            np.random.rand(self.inner_dim, self.head_size),
            np.random.rand(self.inner_dim, self.head_size),
            np.random.rand(self.inner_dim, self.head_size)
        )
        
        # Y-head
        self.WxQ, self.WxK, self.WxV = (
            np.random.rand(self.inner_dim, self.head_size),
            np.random.rand(self.inner_dim, self.head_size),
            np.random.rand(self.inner_dim, self.head_size)
        )
        
        # MHA
        self.WxZ = np.random.rand(self.inner_dim, self.inner_dim)
        
        # FFN-X
        self.WXf1 = np.random.rand(self.inner_dim, self.inner_dim)
        self.WXf2 = np.random.rand(self.inner_dim, self.inner_dim)
        self.BXf1 = np.random.rand(1, self.inner_dim)
        self.BXf2 = np.random.rand(1, self.inner_dim)

    def softmax(self, x):
        exp = np.exp(x)
        return exp / np.sum(exp, axis=-1)

    def attention(self, x, wq, wk, wv):
        Q = x @ wq
        K = x @ wk
        V = x @ wv
        Z1 = Q @ K.T
        Z1 = Z1 / np.sqrt(Z1)

    def forward(self):
         X = self.embed[self.X] / np.sqrt(self.inner_dim) # n x d
         # -- positional encoding
         Zx = 
