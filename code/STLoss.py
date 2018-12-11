import torch

class STLoss(torch.nn.Module):
    def __init__(self):
        super(STLoss,self).__init__()
        self.nllloss = torch.nn.NLLLoss()
    
    def forward(self,domain_weight,output,target):
        #domain_weight.shape = [batch_size,seq_len,num_split]
        nll = self.nllloss(output,target)
        return nll