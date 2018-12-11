import torch
# Custom loss function to control output weight of control RNN
class DCMLoss(torch.nn.Module):
    def __init__(self, hyper1, hyper2):
        super(DCMLoss,self).__init__()
        self.nllloss = torch.nn.NLLLoss()
        # Hyperparameters
        self.hyper1 = hyper1
        self.hyper2 = hyper2
    
    def forward(self,domain_weight,output,target):

        nll = self.nllloss(output,target)
        # reverse of the sum of dot products to itself
        # drives the weight for one of them to be significant
        dot_sum = torch.sum(domain_weight*domain_weight, dim=2)
        dot_sum = torch.sum(dot_sum, dim =1)
        dot_sum = torch.sum(dot_sum, dim = 0)
        dot_sum = (domain_weight.size(0)*domain_weight.size(1))/dot_sum

        # sum of the mean of weight squared
        # to normalize cell usage distribution
        dot_col = torch.sum(domain_weight, dim=1)/domain_weight.size(1)
        dot_col = torch.sum(dot_col, dim=0)/domain_weight.size(0)
        dot_col = dot_col*dot_col
        dot_col = torch.sum(dot_col, dim=0)

        output_loss = nll + ((dot_sum-1) * self.hyper1) + ((dot_col-0.25) * self.hyper2)

        return output_loss
