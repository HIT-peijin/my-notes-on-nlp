    
class PositionalEncoder(nn.Module):
    #固定的三角函数位置编码
    #与输入的embedding做元素加
    def __init__(self, d_model, max_seq_len, norm = 1):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        self.pe = torch.zeros(max_seq_len, d_model)
        

        
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
                self.pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        self.pe = self.pe.unsqueeze(0)
        self.pe.requires_grad = False
        self.pe = self.pe * math.sqrt(norm)   #乘一个常数 增大位置嵌入的影响

    
    def forward(self, x):
        get_cuda_device = x.get_device()

        x = x + self.pe.to(get_cuda_device)

        return x
    
class PositionalEncoder_Concat(nn.Module):
    #固定的三角函数位置编码
    #与输入的embedding做concat 那么 dim of model 就会翻倍
    def __init__(self, d_model, max_seq_len, batch_size, norm):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        self.pe = torch.zeros(max_seq_len, d_model)
        

        
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
                self.pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        # self.pe = self.pe.unsqueeze(0)
        self.pe.requires_grad = False
        self.pe = self.pe.repeat(batch_size, 1, 1)

    
    def forward(self, x):
        get_cuda_device = x.get_device()

        self.pe = self.pe.to(get_cuda_device)
        out =  torch.cat((x,self.pe), 2)

        return out

class PositionalEncoder_Learnable(nn.Module):
    #可学习的位置编码
    #与原embedding 元素相加
    def __init__(self, d_model, max_seq_len, batch_size, norm):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        self.pe = nn.Parameters(torch.randn(1, max_seq_len, d_model))
        


        self.pe = self.pe.repeat(batch_size, 1, 1)
        # self.pe = self.pe * math.sqrt(norm)   #乘一个常数 增大位置嵌入的影响

    
    def forward(self, x):
        get_cuda_device = x.get_device()

        self.pe = self.pe.to(get_cuda_device)
        out =  x + self.pe

        return out

class PositionalEncoder_Learnable_Concat(nn.Module):
    #可学习的位置编码
    #与原embedding 连接
    def __init__(self, d_model, max_seq_len, batch_size, norm):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 

        self.pe = nn.Parameters(torch.randn(1, max_seq_len, d_model))
        

        

        self.pe = self.pe.repeat(batch_size, 1, 1)
        # self.pe = self.pe * math.sqrt(norm)   #乘一个常数 增大位置嵌入的影响

    
    def forward(self, x):
        get_cuda_device = x.get_device()

        self.pe = self.pe.to(get_cuda_device)
        out =  torch.cat((x,self.pe), 2)

        return out
    

