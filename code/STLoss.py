import torch

class STLoss(torch.nn.Module):
    def __init__(self, hyper):
        super(STLoss,self).__init__()
        self.nllloss = torch.nn.NLLLoss()
        self.hyper = hyper
    
    def forward(self,domain_weight,output,target):
        #print(domain_weight, domain_weight.size())
        #domain_weight.shape = [batch_size,seq_len,num_split]
        nll = self.nllloss(output,target)
        dot_sum = 0;
        domain_size = list(domain_weight.size)
        for i in range(domain_size[0]):
            for j in range(domain_size[1]):
                dot_sum += torch.dot(domain_weight[i][j], domain_weight[i][j])
        output_loss = nll + (torch.dot(domain_weight, domain_weight) * self.hyper)

        return output_loss
