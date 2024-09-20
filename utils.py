import random
import numpy as np
import torch
import csv
import os
import torch.nn as nn
from collections import defaultdict
from torch.optim import *
from torch.optim.lr_scheduler import LambdaLR
    
def linear_warmup(step, warmup_steps):
    if step < warmup_steps:
        return float(step) / float(max(1.0, warmup_steps))
    return 1.0

def lr_lambda_inverse(step, d_model=768, warmup_steps=4000, initial_lr=1.0):
    scale_factor = initial_lr * d_model ** -0.5
    if step == 0:
        return scale_factor * warmup_steps ** -0.5
    return scale_factor * min(step ** -0.5, step * warmup_steps ** -1.5)

class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_warm_min, lr_cos_min, lr_max, warm_up=0, T_max=10, start_ratio=0.1):
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min

        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()

        """
        self.lr_warm_min = lr_warm_min
        self.lr_cos_min = lr_cos_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_warm_min + (self.lr_max - self.lr_warm_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_warm_min + (self.lr_max - self.lr_warm_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:
            # this works fine
            lr = self.lr_cos_min + (self.lr_max - self.lr_cos_min) * 0.5 * \
                 (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)

        self.cur += 1

        return [lr for base_lr in self.base_lrs]
    
    
class CustomLRScheduler:
    def __init__(self, optimizer, warmup_steps, lr_min, lr_max, warmup_steps_cos, T_max):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_scheduler =  LambdaLR(optimizer, lr_lambda=lambda step: linear_warmup(step, warmup_steps))
        self.cosine_scheduler = WarmupCosineLR(optimizer, lr_min, lr_max, warm_up=warmup_steps_cos, T_max=T_max, start_ratio=0.0)
        self.step_count = 0

    def step(self):
        self.step_count += 1
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i == 0:  # other_params
                if self.step_count < self.warmup_steps:
                    lr = param_group['initial_lr'] * linear_warmup(self.step_count, self.warmup_steps)
                else:
                    lr = param_group['initial_lr']
            elif i == 1:  # decoder_params
                lr = self.cosine_scheduler.get_lr()[i]
            param_group['lr'] = lr

        self.warmup_scheduler.step()
        self.cosine_scheduler.step()


def set_seed(seed):
    # fix all random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_accuracy_per_class(labels, preds):
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for label, pred in zip(labels, preds):
        if label == pred:
            correct_counts[label] += 1
        total_counts[label] += 1

    accuracy_per_class = {}
    for label in total_counts:
        accuracy_per_class[label] = correct_counts[label] / total_counts[label] * 100

    return accuracy_per_class

def read_plabel_csv(csv_file):
    plabel_dict = {}
    with open(csv_file, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin)
        header = next(reader) 
        assert header == ['Phoneme', 'ID'], "CSV header must be ['Phoneme', 'ID']"
        
        for row in reader:
            assert len(row) == 2, "Each row must have exactly 2 columns"
            phoneme = row[0]
            phoneme_id = int(row[1])
            if phoneme_id == 4:
                plabel_dict[phoneme_id] = '_'
            else:
                plabel_dict[phoneme_id] = phoneme
    return plabel_dict

def plabel2strlist(plabels: torch.tensor, plabel_dict: dict):
    
    phoneme_lists = []
    plabel_lists = plabels.tolist()
    for plabel_list in plabel_lists:
        phoneme_list = []
        for plable in plabel_list:
            phoneme_list.append(plabel_dict[plable])
            if plabel_dict[plable] == '<eos>':
                break
        phoneme_str = ''.join(phoneme_list)
        phoneme_lists.append(phoneme_str) 
    return phoneme_lists

def get_padding_mask(max_len, batch_size, lengths):
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.

    Returns:
        (Tensor): The padding mask.
    """
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths[:, None]
    return mask

class GradNorm:
    def __init__(self, model, num_tasks, alpha=1.5):
        self.model = model
        self.num_tasks = num_tasks
        self.alpha = alpha

        self.task_weights = nn.Parameter(torch.ones(num_tasks, device='cuda'))
        self.initial_task_losses = None  # This should be set during the first forward pass
        self.optimizer = torch.optim.SGD([self.task_weights], lr=5e-5)
        self.layer = self.model.pretrain.pretrain.encoder.layers[-1]
        self.cnt = 0
        
        torch.autograd.set_detect_anomaly(True)
    
    def update_weights(self, losses, grad_norms):
        self.cnt += 1
        
        if self.initial_task_losses is None:
            self.initial_task_losses = [l.detach() for l in losses]
            print('init_loss: ', self.initial_task_losses)

        loss_ratios = torch.stack([l.detach() / init_l for l, init_l in zip(losses, self.initial_task_losses)])
        
        avg_loss_ratio = torch.mean(loss_ratios)
        rt = loss_ratios / avg_loss_ratio
        adjusted_grad_norms = grad_norms.clone()
        adjusted_grad_norms = adjusted_grad_norms * self.task_weights
        gw_avg = grad_norms.mean().detach()
        constant = (gw_avg * rt ** self.alpha).detach()
        
        if self.cnt % 20 == 0:
            print('\n\n')
            print('rt: ', rt)
            print('constant: ', constant, ' grad_norms: ', grad_norms)
        
        grad_norm_loss = torch.sum(torch.abs(adjusted_grad_norms - constant))
        
        self.optimizer.zero_grad()
        grad_norm_loss.backward()
        
        if self.cnt % 20 == 0:
            print('task_weight_grad: ', self.task_weights.grad)
            print()
            print('task_weights: ', self.task_weights)
            
        self.optimizer.step()
        if self.cnt % 20 == 0:
            print('task_weights_afterstep: ', self.task_weights)
            
        self.optimizer.zero_grad()
            
        with torch.no_grad():
            self.task_weights.data = (self.task_weights.data / self.task_weights.data.sum()) * self.num_tasks
        

class DynamicWeightAveraging:
    def __init__(self, num_tasks, temperature=2.0):
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.previous_losses = torch.ones(num_tasks).cuda()
        self.task_weights = torch.ones(num_tasks).cuda()
    
    def get_weights(self, current_losses):
        """
        Get dynamic weights based on the current and previous losses.
        
        Args:
            current_losses (torch.Tensor): The current losses for each task.
        
        Returns:
            torch.Tensor: The dynamic weights for each task.
        """
        # Compute the rate of change of losses
        delta_losses = current_losses / self.previous_losses
        delta_losses = torch.clamp(delta_losses, min=0.3, max=3.0)
        # Update the previous losses
        self.previous_losses = current_losses.clone()
        # Compute the weights
        weights = torch.softmax(delta_losses / self.temperature, dim=0) * self.num_tasks
        self.task_weights = weights
        
        return weights

class UncertaintyWeights(nn.Module):
    r"""
    Uncertainty Weights (UW).

    This method implements the uncertainty weighting strategy proposed in the paper:
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (CVPR 2018). The weights for each task's loss are learned during training based on their uncertainties.
    """

    def __init__(self, num_tasks: int, device: str = 'cuda'):
        super(UncertaintyWeights, self).__init__()
        self.num_tasks = num_tasks
        self.device = device
        self.cnt = 0
        # Initialize learnable parameters for scaling the loss of each task.
        self.loss_scales = nn.Parameter(
            torch.full((self.num_tasks,), -0.5, device=self.device),
            requires_grad=True,
        )
        self.optimizer = torch.optim.SGD([self.loss_scales], lr=5e-5)

    def forward(self, task_losses: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass to compute the weighted sum of task losses using uncertainty weighting.

        Args:
            task_losses (torch.Tensor): A tensor containing the losses for each task.

        Returns:
            torch.Tensor: The weighted sum of all task losses.
        """
        # Compute the weights for each task based on the inverse of the exponential of loss scales.
        self.cnt += 1

        task_weights = 1 / (2 * torch.exp(self.loss_scales))

        if self.cnt % 20 == 0:
            print('UW task weights: ', task_weights)

        # Compute the weighted loss for each task.
        weighted_losses = task_losses / (2 * torch.exp(self.loss_scales))

        # Add half the log of the scale to the weighted losses.
        # This term is derived from the uncertainty weighting formula in the paper.
        augmented_losses = weighted_losses + self.loss_scales / 2

        # Compute the total loss as the sum of all augmented losses.
        total_loss = augmented_losses.sum()

        # Backward pass is implicitly triggered when total_loss is used in an optimization step.
        return total_loss, task_weights.detach().cpu().numpy()


class DynamicTaskPrioritization(nn.Module):
    def __init__(self, num_tasks, alpha=0.8, temperature=1.0):
        super(DynamicTaskPrioritization, self).__init__()
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.alpha = alpha
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        self.last_Kt = torch.zeros(num_tasks, requires_grad=False)  

        # the value of gamma is smaller, the higher the priority is
        self.gammas = nn.Parameter(torch.tensor([1.0, 0.5]))

    def forward(self, Kt):
        cur_Kt = self.alpha * Kt + (1 - self.alpha) * self.last_Kt
        self.last_Kt = cur_Kt.detach()

        # compute the difficulty of each task
        difficulties = torch.pow(1 - cur_Kt, self.gammas)

        # compute the dynamic weights, avoid log(0) error
        weights = nn.functional.softmax(- difficulties * torch.log(cur_Kt + 1e-6) / self.temperature, dim=0)
        with torch.no_grad():
            self.task_weights.data.copy_(weights)

        print('DTP task weights: ', self.task_weights)
