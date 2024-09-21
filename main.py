"""
This code is based on the implementation from the following GitHub repository:
https://github.com/ECNU-Cross-Innovation-Lab/ENT

Original author: [Siyuan Shen]
Original repository license: [Apache-2.0 license]

This code has been modified and extended from the original implementation.
"""

import os
import torch
import argparse
import numpy as np
import torch.optim as optim
from dataset.build_loader import build_loader
from models.build_model import build_model
from metrics import IEMOCAP_Meter
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils import set_seed, calculate_accuracy_per_class, linear_warmup, read_plabel_csv, plabel2strlist, \
get_padding_mask, WarmupCosineLR, CustomLRScheduler, lr_lambda_inverse, \
GradNorm, DynamicWeightAveraging, UncertaintyWeights, DynamicTaskPrioritization
from models.phoneme_decoder_cache import predict_batch, PhonemeDecoder
from timm.scheduler.cosine_lr import CosineLRScheduler
from jiwer.measures import wer, cer

emo_dict = {0:'h', 1:'a', 2:'s', 3:'n'}
gender_dict = {'M': 0, 'F': 1}

# guarantee CUBLAS library calculation determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def parse_option():
    parser = argparse.ArgumentParser('Emo_Cls', add_help=False)
    parser.add_argument('--out_path', type=str, help='path to out')
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--finetune', action='store_true', help='whether to finetune or feature extraction')
    parser.add_argument('--gpu', type=int, help='gpu rank to use')
    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--five_fold', action='store_true', help='whether to 5 fold valid')
    parser.add_argument('--n_epoch', type=int, help='num of epoch', default=50)
    parser.add_argument('--mode', choices=['mean', 'att', 'att_fusion', 't_att', 'multi_trans'], 
                        type=str, help='pooling mode')
    parser.add_argument('--weight_phoneme', type=float, help='weight phoneme')
    parser.add_argument('--fold_id', default=1, type=int, help='id for single-fold training')
    parser.add_argument('--schd_mode', default='fixed', type=str, help='mode of scheduler')
    parser.add_argument('--spk_weight', default=0.1, type=float, help='weight of speaker_loss')
    parser.add_argument('--fuse_mode', choices=['concat', 'project', 'cross', 'gate', 'cross_simple', 'bi_linear'], default='concat',
                    type=str, help='the way of fusion features')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='warmup ratio of total step')
    parser.add_argument('--use_sampler', action='store_true', help='whether to use a custom sampler')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--other_head', default=0.0, type=float, help='weight of other head')
    parser.add_argument('--ablation_level', default=-1, type=int, help='level of ablation_features')
    parser.add_argument('--fixcos', default=0, type=int, help='whether to use two scheduler')
    parser.add_argument('--train_strategy', default=0, type=int, help='which train strategy to use')
    parser.add_argument('--freeze_all', default=0, type=int, help='whether to freeze all parameters')

    args = parser.parse_args()
    return args


def train_one_epoch(args, model, train_loader, optimizer, epoch, scheduler, optimizer_spk=None, scheduler_spk=None,\
                    optimizer_other=None, optimizer_decoder=None, scheduler_other=None, 
                    scheduler_decoder=None, grad_norm=None, DWA=None, UW=None, DTP=None):
    model.train()
    total_loss = 0.0
    cls_loss = 0.0
    phoneme_loss = 0.0
    domain_loss = 0.0
    spk_loss = 0.0
    gender_loss = 0.0
    current_loss_sum = torch.tensor([0.0, 0.0]).cuda()
    
    if args.fixcos == 0:
        optimizer.zero_grad()
    else:
        optimizer_other.zero_grad()
        optimizer_decoder.zero_grad()
    
    if optimizer_spk is not None:
        optimizer_spk.zero_grad()
        
    sum_step = len(train_loader) * args.n_epoch
    now_step = epoch * len(train_loader)
    for i, batch in enumerate(train_loader):

        audio, audio_length, tlabel, tlabel_length, label, text, filename, \
                                                    plabel, plabel_length = batch

        audio = audio.cuda()
        audio_length = audio_length.cuda()
        tlabel = tlabel.cuda()
        tlabel_length = tlabel_length.cuda()
        label = label.cuda()
        plabel = plabel.cuda()
        plabel_length = plabel_length.cuda()

        now_step += len(batch)
                
        if optimizer_spk is not None:
            outputs = model(audio, audio_length, label, plabel, plabel_length, 
                filename, now_step, sum_step)
            loss_spk = outputs.loss_spk
            optimizer_spk.zero_grad()
            loss_spk.backward()
            optimizer_spk.step()
            spk_loss += loss_spk.item()
            if scheduler_spk is not None:
                scheduler_spk.step()
    
        outputs = model(audio, audio_length, label, plabel, plabel_length, 
                        filename, now_step, sum_step)
        
        loss = outputs.loss
        cls_loss += outputs.loss_emo.item()
        phoneme_loss += outputs.loss_phoneme.item()
        domain_loss += outputs.loss_domain.item()
        gender_loss += outputs.loss_gender.item()
 
        if args.fixcos == 0:
            optimizer.zero_grad()
        else:
            optimizer_other.zero_grad()
            optimizer_decoder.zero_grad()
        
        if args.train_strategy == 1: 
            losses = outputs.losses
            grads = {}
            grad_norms = []
            for idx in range(grad_norm.num_tasks):
                if args.fixcos == 0:
                    optimizer.zero_grad()
                else:
                    optimizer_other.zero_grad()
                    optimizer_decoder.zero_grad()
                    
                outputs = model(audio, audio_length, label, plabel, plabel_length, 
                        filename, now_step, sum_step)

                task_loss=outputs.losses
                task_grads = torch.autograd.grad(task_loss[idx], model.phoneme_dec.transformer_decoder.parameters())
                task_grads = [g for g, p in zip(task_grads, grad_norm.layer.parameters())]
                grads[idx] = torch.cat([g.flatten() for g in task_grads])
                grad_norms.append(torch.norm(grads[idx]))
            
            grad_norms = torch.stack(grad_norms)
            # apply GradNorm
            loss = sum([w * l for w, l in zip(grad_norm.task_weights, losses)])
            loss.backward(retain_graph=True)
            grad_norm.update_weights(losses, grad_norms)
        elif args.train_strategy == 2:
            
            if epoch <= 1:
                DWA_weights = torch.ones(DWA.num_tasks).cuda()
            else:
                DWA_weights = DWA.task_weights
            current_losses = torch.tensor([loss.item() for loss in outputs.losses]).cuda()
            # update weights every step
            weights = DWA.get_weights(current_losses)
            current_loss_sum += current_losses
            loss = sum([w * l for w, l in zip(DWA_weights, outputs.losses)])
            loss.backward()
        elif args.train_strategy == 3:
            losses = outputs.losses
            loss, task_weights = UW(torch.stack(losses))
            loss.backward()
            UW.optimizer.step()
        elif args.train_strategy == 4:
            loss = sum([w * l for w, l in zip(DTP.task_weights, outputs.losses)])
            loss.backward()
        else:
            loss.backward()
        
        if args.fixcos == 0:
            optimizer.step()
        else:
            optimizer_other.step()
            optimizer_decoder.step()
            
        total_loss += loss.item()

        if args.fixcos == 0:
            scheduler.step()
        else:
            scheduler_other.step()
            scheduler_decoder.step()

    if args.fixcos == 0:
        for param_group in optimizer.param_groups:
            print(f"  Learning Rate: {param_group['lr']}")
    else:
        current_linear_lr = scheduler_other.get_last_lr()[0]
        current_cos_lr = scheduler_decoder.get_last_lr()[0]
        print('Linear LR:', current_linear_lr, ' CosorInverse LR:', current_cos_lr)

    return total_loss, cls_loss, phoneme_loss, domain_loss, spk_loss, gender_loss, outputs

@torch.no_grad()
def validate(args, test_loader, model, epoch, cache=None):
    model.eval()
    total_loss = 0.0
    cls_loss = 0.0
    phoneme_loss = 0.0

    emo_pred_list = []
    emo_label_list = []
    male_emo_pred_list = []
    male_emo_label_list = []
    female_emo_pred_list = []
    female_emo_label_list = []

    gender_pred_list = []
    gender_label_list = []
    phoneme_pred_list = []
    phoneme_label_list = []
    sum_step = len(test_loader) * args.n_epoch
    now_step = epoch * len(test_loader)
    my_dict = read_plabel_csv('./dataset/IEMOCAP/phoneme_dictcmu.csv')
    
    # use phoneme decoder with kvcache to inference to speed up
    phoneme_decoder_cache = PhonemeDecoder(phoneme_vocab_size=75, d_model=768, num_decoder_layers=2,
                            nhead=12, dim_feedforward=768, dropout=0.1)
    phoneme_decoder_cache.cuda().eval()
    phoneme_decoder_cache.load_state_dict(model.phoneme_dec.state_dict())
    
    # use half precision inference to speed up
    phoneme_decoder_cache.half()
    for i, batch in enumerate(test_loader):
        audio, audio_length, tlabel, tlabel_length, label, text, filename, \
                                                    plabel, plabel_length = batch
        
        no_predicted_label = 0
        if args.freeze_all == 1:
            no_predicted_label = any(key not in cache for key in filename)
        
        audio = audio.cuda()
        audio_length = audio_length.cuda()
        label = label.cuda()
        plabel = plabel.half().cuda()
        tlabel = tlabel.half().cuda()
        plabel_length = plabel_length.cuda()
        tlabel_length = tlabel_length.cuda()
        audio_states= model.pretrain(audio)
        batch_size = audio_states.shape[0]
        
        # output transform half precision
        audio_states.half()
        B_audio, T_audio, E_audio = audio_states.shape
        audio_padding_mask = get_padding_mask(T_audio, B_audio, audio_length)
                    
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                if args.ablation_level == -1 or args.ablation_level == 2:
                    if args.freeze_all == 0 or (args.freeze_all == 1 and no_predicted_label == 1):
                        pred_plabel, pred_plabel_length = predict_batch(phoneme_decoder_cache, audio_states, batch_size=batch_size, start_token=2, end_token=3, 
                                            max_length=400, vocab_size=75, memory_mask=~audio_padding_mask)
                        if args.freeze_all == 1:
                            if (epoch >= 1):
                                print('Bad Cache!')
                            for i, file in enumerate(filename):
                                cache[file] = [pred_plabel[i], pred_plabel_length[i]]
                    else:
                        pred_plabel, pred_plabel_length = [], []
                        for i, file in enumerate(filename):
                            pred_plabel.append(cache[file][0])
                            pred_plabel_length.append(cache[file][1])
                        pred_plabel = torch.stack(pred_plabel)
                        pred_plabel_length = torch.stack(pred_plabel_length)
                
                if args.ablation_level == 0 or args.ablation_level == 1:
                    # ablation study, construct the label for input
                    pred_plabel = torch.full(size=(B_audio, 11), fill_value=2)
                    pred_plabel_length = torch.full(size=(B_audio, ), fill_value=1)
            
            pred_list = plabel2strlist(pred_plabel, my_dict)
            target_list = plabel2strlist(plabel, my_dict)
            phoneme_pred_list.extend(pred_list)
            phoneme_label_list.extend(target_list)
            now_step += 1
            outputs = model(audio, audio_length, label, pred_plabel.cuda(), pred_plabel_length.cuda(), 
                            filename, now_step, sum_step)

        loss = outputs.loss
        cls_loss += outputs.loss_emo.item()
        phoneme_loss += outputs.loss_phoneme.item()
        total_loss += loss.item()

        logits = outputs.head_logits
        gender_logits = outputs.gender_logits

        emo_pred = list(torch.argmax(logits, dim=1).cpu().numpy())
        emo_label = list(label.cpu().numpy())
        emo_pred_list.extend(emo_pred)
        emo_label_list.extend(emo_label)

        gender_label = []
        for file in filename:
            gender_label.append(gender_dict[file.split('_')[0][-1]])

        indices_of_male = [i for i, x in enumerate(gender_label) if x == 0]
        indices_of_female = [i for i, x in enumerate(gender_label) if x == 1]
        male_emo_label_list.extend([emo_label[i] for i in indices_of_male])
        female_emo_label_list.extend([emo_label[i] for i in indices_of_female])

        indices_of_male_tensor = torch.LongTensor(indices_of_male)
        indices_of_female_tensor = torch.LongTensor(indices_of_female)
        male_emo_pred_list.extend(list(torch.argmax(logits, dim=1)[indices_of_male_tensor].cpu().numpy()))
        female_emo_pred_list.extend(list(torch.argmax(logits, dim=1)[indices_of_female_tensor].cpu().numpy()))

        gender_pred = list(torch.argmax(gender_logits, dim=1).cpu().numpy())
        gender_pred_list.extend(gender_pred)
        gender_label_list.extend((gender_label))

    if len(emo_label_list):
        WA = accuracy_score(emo_label_list, emo_pred_list) * 100
        UA = balanced_accuracy_score(emo_label_list, emo_pred_list) * 100
        WA_male = accuracy_score(male_emo_label_list, male_emo_pred_list) * 100
        UA_male = balanced_accuracy_score(male_emo_label_list, male_emo_pred_list) * 100
        WA_female = accuracy_score(female_emo_label_list, female_emo_pred_list) * 100
        UA_female = balanced_accuracy_score(female_emo_label_list, female_emo_pred_list) * 100

    if len(gender_label_list):
        gender_acc = accuracy_score(gender_label_list, gender_pred_list) * 100

    acc_pre_label = calculate_accuracy_per_class(emo_label_list, emo_pred_list)
    e0_acc, e1_acc, e2_acc, e3_acc = acc_pre_label[0], acc_pre_label[1], acc_pre_label[2], acc_pre_label[3]
    CER = cer(phoneme_label_list, phoneme_pred_list) * 100.0

    return (total_loss, cls_loss, phoneme_loss, WA, UA, WA_male, UA_male, WA_female, UA_female,
            e0_acc, e1_acc, e2_acc, e3_acc, gender_acc, CER, cache)

def solve(args, id_fold=1):

    os.makedirs(args.out_path, exist_ok=True)
    train_loader, test_loader = build_loader(batch_size=args.batch_size, id_fold=id_fold, use_sampler=args.use_sampler)
    model = build_model(finetune=args.finetune, mode=args.mode, fuse_mode=args.fuse_mode, 
                        phoneme_weight=args.weight_phoneme, spk_weight=args.spk_weight, otherhead_weight=args.other_head, all_args=args)
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    pred_cache = {}
    if args.freeze_all:
        checkpoint = torch.load('./output/checkpoint2/cls_B16_mean_SE_LR2e-5_LR2e-4_1.0PH_join_muti-ln_0.0hidgrl_lenbatch_kvcache_0.2warm_fp16_0.0w2v2mask/fold2/Emo_Cls_2.pth')
        model.load_state_dict(checkpoint, strict=False)
        print('Model Load Successfully!')
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.emo_head.se_audio.parameters():
            param.requires_grad = True
        for param in model.emo_head.se_phoneme.parameters():
            param.requires_grad = True
        
        for param in model.emo_head.cross1.parameters():
            param.requires_grad = True
        for param in model.emo_head.cross2.parameters():
            param.requires_grad = True
    
    # emotion recognition and phoneme recognition
    num_tasks = 2
    grad_norm = GradNorm(model, num_tasks)
    DWA = DynamicWeightAveraging(num_tasks)
    UW = UncertaintyWeights(num_tasks)
    DTP = DynamicTaskPrioritization(num_tasks)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                            eps=1e-8)
    optimizer_spk = None
    scheduler_spk = None
    warmup_ratio = args.warmup_ratio
    total_steps = len(train_loader) * args.n_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    
    decoder_params = list(model.phoneme_dec.parameters())
    other_params = [param for name, param in model.named_parameters() if 'phoneme_dec' not in name]
    
    # due to the encoder is pretrained model, and the decoder is new trained model
    # so set a large learning rate for decoder
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': args.learning_rate},
        {'params': decoder_params, 'lr': args.learning_rate * 10}
    ], betas=(0.9, 0.999), eps=1e-8)
    
    optimizer_other = optim.AdamW([{'params': other_params, 'lr': args.learning_rate}], betas=(0.9, 0.999), eps=1e-8)
    optimizer_decoder = optim.AdamW([{'params': decoder_params, 'lr': args.learning_rate * 10}], betas=(0.9, 0.999), eps=1e-8)

    if args.spk_weight > 0.0 and args.fuse_mode == 'concat':
        all_parameters = set(model.parameters())
        spk_head_parameters = set(model.emo_head.spk_head.parameters())
        parameters_without_spk_head = all_parameters - spk_head_parameters
        
        optimizer_spk = optim.AdamW(model.emo_head.spk_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                eps=1e-8)
        optimizer = optim.AdamW(parameters_without_spk_head, lr=1e-5, betas=(0.9, 0.999), 
                                eps=1e-8)
        if args.schd_mode == 'fixed':
            scheduler_spk = LambdaLR(optimizer_spk, lr_lambda=lambda step: linear_warmup(step, warmup_steps))
        elif args.schd_mode == 'cos':
            scheduler_spk = WarmupCosineLR(optimizer_spk, lr_min=1e-9, lr_max=1e-5, 
                               warm_up=warmup_steps, T_max=total_steps)
    
    if args.schd_mode == 'fixed':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: linear_warmup(step, warmup_steps))
        
    elif args.schd_mode == 'cos':
        scheduler = WarmupCosineLR(optimizer, lr_min=1e-9, lr_max=1e-5, 
                               warm_up=warmup_steps, T_max=total_steps)
        
    
    scheduler_other = LambdaLR(optimizer_other, lr_lambda=lambda step: linear_warmup(step, warmup_steps))
    scheduler_decoder = LambdaLR(optimizer_decoder, lr_lambda=lambda step: linear_warmup(step, warmup_steps))

    if id_fold == args.fold_id:
        print(str(model))
        print(f"number of params: {n_parameters}")
        print("Optimizer Parameters:")
        
        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']}")
            print(f"Weight Decay: {param_group['weight_decay']}")
        if optimizer_spk is not None:
            print('\n')
            print("Optimizer_Spk Parameters:")
            for param_group in optimizer_spk.param_groups:
                print(f"Learning Rate: {param_group['lr']}")
                print(f"Weight Decay: {param_group['weight_decay']}")

        print("\nScheduler Parameters:")
        print(scheduler.state_dict())

    print('#' * 30 + '  Start Training  ' + '#' * 30)

    Meter = IEMOCAP_Meter()
    
    for epoch in range(args.n_epoch):
        print(f'>> epoch {epoch}')
        train_loss, train_cls_loss, train_phoneme_loss, train_domain_loss, train_spk_loss, train_gender_loss, outputs = \
        train_one_epoch(args, model, train_loader, optimizer, epoch, scheduler, optimizer_spk=optimizer_spk, scheduler_spk=scheduler_spk,\
                            optimizer_other=optimizer_other, optimizer_decoder=optimizer_decoder,\
                            scheduler_other=scheduler_other, scheduler_decoder=scheduler_decoder, grad_norm=grad_norm, DWA=DWA, UW=UW, DTP=DTP)
        print('Validing... ')
        test_loss, cls_loss, phoneme_loss, WA, UA, WA_male, UA_male, WA_female, UA_female, e0_acc, e1_acc, e2_acc, e3_acc, gender_acc, CER, pred_cache = validate(args, test_loader, model, epoch, cache=pred_cache)
        print(
            f'train loss: {train_loss:.2f}, test loss: {test_loss:.2f}, WA: {WA:.2f}, ({Meter.WA:.2f}), UA: {UA:.2f}, ({Meter.UA:.2f}), CER: {CER:.2f}, Gender_Acc: {gender_acc:.2f},\n'
            f'WA_male: {WA_male:.2f}, ({Meter.WA_male:.2f}), UA_male: {UA_male:.2f}, ({Meter.UA_male:.2f}), WA_female: {WA_female:.2f}, ({Meter.WA_female:.2f}), UA_female: {UA_female:.2f}, ({Meter.UA_female:.2f}),\n'
            f'train_cls_loss: {train_cls_loss:.2f}, train_phoneme_loss: {train_phoneme_loss:.2f}, \n'
            f'train_domain_loss: {train_domain_loss:.2f}, train_spk_loss: {train_spk_loss:.2f}, train_gender_loss: {train_gender_loss:.2f} \n'
            f'hap_acc: {e0_acc:.2f}, ang_acc: {e1_acc:.2f}, sad_acc: {e2_acc:.2f}, ner_acc: {e3_acc:.2f} \n')

        if args.train_strategy == 4:
            KT = torch.tensor([WA * 0.01, 1.0 - CER * 0.01])
            KT = torch.clamp(KT, min=0.01)
            DTP(KT)

        if Meter.UA < UA and args.out_path:
            torch.save(model.state_dict(), f'{args.out_path}/Emo_Cls_{id_fold}.pth')
        Meter.update(WA, UA, WA_male, UA_male, WA_female, UA_female)

    print('#' * 30 + f'  Summary fold{id_fold}  ' + '#' * 30)
    print(f'MAX_WA: {Meter.WA:.2f}')
    print(f'MAX_UA: {Meter.UA:.2f}')
    print(f'MAX_WA_male: {Meter.WA_male:.2f}')
    print(f'MAX_UA_male: {Meter.UA_male:.2f}')
    print(f'MAX_WA_female: {Meter.WA_female:.2f}')
    print(f'MAX_UA_female: {Meter.UA_female:.2f}')
    
    test_UA = (Meter.test_UA_male + Meter.test_UA_female) / 2
    test_WA = (Meter.test_WA_male + Meter.test_WA_female) / 2
    print("TEST RESULTS: ")
    print(f'TEST_WA: {test_WA:.2f}')
    print(f'TEST_UA: {test_UA:.2f}')
    return test_WA, test_UA


def main():
    args = parse_option()
    print('#' * 30 + '  Train Args  ' + '#' * 30)
    print(args)
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    if args.five_fold:
        results = []
        for i in range(1, 6):
            WA, UA = solve(args, i)
            results.append([WA, UA])

        if results:
            print('#' * 30 + f'  Summary  ' + '#' * 30)
            mean_WA = np.mean([acc[0] for acc in results])
            mean_UA = np.mean([acc[1] for acc in results])
            print(f'MEAN_UA: {mean_UA:.2f}, MEAN_WA: {mean_WA:.2f}')
    else:
        solve(args, args.fold_id)

if __name__ == '__main__':
    main()
