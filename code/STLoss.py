import torch

class STLoss(torch.nn.Module):
    def __init__(self, hyper):
        super(STLoss,self).__init__()
        self.nllloss = torch.nn.NLLLoss()
        self.hyper = hyper
    
    def forward(self,domain_weight,output,target):
        #domain_weight.shape = [batch_size,seq_len,num_split]
        nll = self.nllloss(output,target)
        output_loss = nll + (torch.dot(domain_weight, domain_weight) * self.hyper)


        return output_loss
