import os
import copy
import argparse
from sympy import arg
import torch.optim
from meta import *
from model import *
from NLT_CIFAR import *
import torch.nn.functional as F
from loguru import logger
import time

parser = argparse.ArgumentParser(description='Meta-GLA')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--meta_net_hidden_size', type=int, default=200)
parser.add_argument('--meta_net_num_layers', type=int, default=2)

parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--dampening', type=float, default=0.)
parser.add_argument('--nesterov', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--meta_lr', type=float, default=0.01)   
parser.add_argument('--meta_batch', type=int, default=100)   
parser.add_argument('--meta_weight_decay', type=float, default=5e-4)

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--imbalanced_factor', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=200) 
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--meta_interval', type=int, default=20)
parser.add_argument('--paint_interval', type=int, default=20)
parser.add_argument('--nk', type=int, default=40)
parser.add_argument('--tau1', type=float, default=1.5)
parser.add_argument('--tau2', type=float, default=1.0)
parser.add_argument('--epoch1', type=int, default=160)
parser.add_argument('--epoch2', type=int, default=180)
args = parser.parse_args()
logger.info(args)

# cosine similarity
class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x1,x2):
        x2 = x2.t()
        x = x1.mm(x2)
        x1_ = x1.norm(dim=1).unsqueeze(0).t()
        x2_ = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_.mm(x2_)
        dist = x.mul(1/x_frobenins)
        return dist


def weight_feature(logits_vector,loss_vector,labels,index,class_rate, neibor_index,loss_tensor_last, margin_tensor_last, grad_tensor_last, logit_tensor_last, pred_tensor_last,  entropy_tensor_last, diff_loss_n_last, diff_margin_n_last, diff_grad_n_last, diff_logit_n_last, diff_pred_n_last, diff_entropy_n_last, 
                  loss_n_mean_last, margin_n_mean_last,grad_n_mean_last,  logit_n_mean_last, pred_n_mean_last, entropy_n_mean_last, current_class, other_class, neighbor_feature_distance, current_class_last, other_class_last, neighbor_feature_distance_last):

    # basic characteristics
    labels_one_hot = F.one_hot(labels,num_classes=(args.num_classes)).float() 
    class_rate = torch.mm(labels_one_hot,class_rate.unsqueeze(1)) 
    logits_labels = torch.sum(F.softmax(logits_vector,dim=1) * labels_one_hot,dim=1)
    logits = torch.sum(logits_vector*labels_one_hot, dim=1)
    logits_vector_grad = torch.norm(labels_one_hot- F.softmax(logits_vector,dim=1),dim=1)   
    logits_others_max =(F.softmax(logits_vector,dim=1)[labels_one_hot!=1].reshape(F.softmax(logits_vector,dim=1).size(0),-1)).max(dim=1).values 
    logits_margin =  logits_labels - logits_others_max
    entropy =  torch.sum(F.softmax(logits_vector,dim=1)*F.log_softmax(logits_vector,dim=1),dim=1) 


    # neighborhood extension
    neighbor_loss = loss_vector[neibor_index] 
    neighbor_logit = logits[neibor_index]
    neighbor_grad = logits_vector_grad[neibor_index]
    neighbor_margin  = logits_margin[neibor_index]
    neighbor_entropy = entropy[neibor_index]
    neighbor_prediction = logits_labels[neibor_index]
    n_loss_mean = torch.mean(neighbor_loss,1)

    n_margin_mean = torch.mean(neighbor_margin,1)
    n_grad_mean = torch.mean(neighbor_grad,1)
    n_logit_mean = torch.mean(neighbor_logit,1)
    n_pred_mean = torch.mean(neighbor_prediction,1)
    n_entropy_mean = torch.mean(neighbor_entropy,1)
    diff_loss_n = loss_vector - n_loss_mean
    diff_logit_n = logits - n_logit_mean
    diff_grad_n = logits_vector_grad - n_grad_mean
    diff_pred_n = logits_labels - n_pred_mean
    diff_entropy_n = entropy - n_entropy_mean
    diff_margin_n = logits_margin - n_margin_mean

    # sequence extension
    loss_last = torch.sub(loss_vector, loss_tensor_last[index])
    margin_last = torch.sub(logits_margin,margin_tensor_last[index])
    grad_last = torch.sub(logits_vector_grad,grad_tensor_last[index])
    logit_last = torch.sub(logits, logit_tensor_last[index])
    pred_last = torch.sub(logits_labels,pred_tensor_last[index])
    entropy_last = torch.sub(entropy,entropy_tensor_last[index])
    loss_n_last = torch.sub(diff_loss_n,diff_loss_n_last[index])
    margin_n_last = torch.sub(diff_margin_n,diff_margin_n_last[index])
    grad_n_last = torch.sub(diff_grad_n, diff_grad_n_last[index])
    logit_n_last = torch.sub(diff_logit_n,diff_logit_n_last[index])
    pred_n_last = torch.sub(diff_pred_n, diff_pred_n_last[index])
    entropy_n_last = torch.sub(diff_entropy_n, diff_entropy_n_last[index])
    loss_n_mean_last1 = torch.sub(n_loss_mean,loss_n_mean_last[index])
    margin_n_mean_last1 = torch.sub(n_margin_mean,margin_n_mean_last[index])
    grad_n_mean_last1 = torch.sub(n_grad_mean,grad_n_mean_last[index])
    logit_n_mean_last1 = torch.sub(n_logit_mean,logit_n_mean_last[index])
    pred_n_mean_last1 = torch.sub(n_pred_mean,pred_n_mean_last[index])
    entropy_n_mean_last1 = torch.sub(n_entropy_mean,entropy_n_mean_last[index])

    # other characteristics
    current_class_diff = torch.sub(current_class.squeeze(),current_class_last[index])
    other_class_diff = torch.sub(other_class.squeeze(),other_class_last[index])
    neighbor_feature_distance_diff = torch.sub(neighbor_feature_distance.squeeze(),neighbor_feature_distance_last[index])

    feature = torch.cat([loss_vector.unsqueeze(1),
                         logits.unsqueeze(1),
                         logits_labels.unsqueeze(1),
                        logits_vector_grad.unsqueeze(1),
                        logits_margin.unsqueeze(1),
                        entropy.unsqueeze(1),
      
                        n_loss_mean.unsqueeze(1),
                        n_margin_mean.unsqueeze(1),
                        n_grad_mean.unsqueeze(1),
                        n_logit_mean.unsqueeze(1),
                        n_pred_mean.unsqueeze(1),
                        n_entropy_mean.unsqueeze(1),
              
                        diff_loss_n.unsqueeze(1),
                        diff_margin_n.unsqueeze(1),
                        diff_grad_n.unsqueeze(1),
                        diff_logit_n.unsqueeze(1),
                        diff_pred_n.unsqueeze(1),
                        diff_entropy_n.unsqueeze(1),
                  
                        current_class.unsqueeze(1),
                        other_class.unsqueeze(1),
                        neighbor_feature_distance,
                       
                        current_class_diff.unsqueeze(1),
                        other_class_diff.unsqueeze(1),
                        neighbor_feature_distance_diff.unsqueeze(1),
                 
                        loss_last.unsqueeze(1),
                        margin_last.unsqueeze(1),
                        grad_last.unsqueeze(1),
                        logit_last.unsqueeze(1),
                        pred_last.unsqueeze(1),
                        entropy_last.unsqueeze(1),
                        loss_n_last.unsqueeze(1),
                        margin_n_last.unsqueeze(1),
                        grad_n_last.unsqueeze(1),
                        logit_n_last.unsqueeze(1),
                        pred_n_last.unsqueeze(1),
                        entropy_n_last.unsqueeze(1),
                        loss_n_mean_last1.unsqueeze(1),
                        margin_n_mean_last1.unsqueeze(1),
                        grad_n_mean_last1.unsqueeze(1),
                        logit_n_mean_last1.unsqueeze(1),
                        pred_n_mean_last1.unsqueeze(1),
                        entropy_n_mean_last1.unsqueeze(1),
                        ],dim=1)
    return feature


def compute_loss_accuracy(net, data_loader, criterion, device):
    net.eval()
    correct = 0
    total_loss = 0.
    

    with torch.no_grad():
        for batch_idx, (_,inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            features, outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()

    return total_loss / (batch_idx + 1), correct / len(data_loader.dataset)


def meta_GLA():
    meta_net = MLP(in_size = 42,hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers).to(device=args.device)
    time_begin = time.time()
    print("time_begin")
    net = ResNet32(args.dataset == 'cifar10' and 10 or 100).to(device=args.device)
    criterion = nn.CrossEntropyLoss().to(device=args.device)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
    lr = args.lr

    train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list = build_dataloader(
        seed=args.seed,
        dataset=args.dataset,
        num_meta_total=args.num_meta,
        imbalanced_factor=args.imbalanced_factor,
        batch_size=args.batch_size,
        meta_batch=args.meta_batch,
    )

    label_freq_array = np.array(imbalanced_num_list) / np.sum(np.array(imbalanced_num_list))
    adjustments = np.log(label_freq_array ** 1.0 + 1e-12)
    adjustments = torch.FloatTensor(adjustments).cuda()

    class_rate = torch.from_numpy(np.array(imbalanced_num_list)/sum(imbalanced_num_list)).cuda().float()
    meta_dataloader_iter = iter(meta_dataloader)
    iteration = 0
    best_acc = 0.
    
    loss_tensor_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    loss_tensor_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    logit_tensor_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    logit_tensor_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    pred_tensor_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    pred_tensor_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    grad_tensor_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    grad_tensor_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    margin_tensor_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    margin_tensor_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    entropy_tensor_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    entropy_tensor_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_loss_n_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_loss_n_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_margin_n_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_margin_n_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_grad_n_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_grad_n_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_logit_n_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_logit_n_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_pred_n_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_pred_n_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_entropy_n_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    diff_entropy_n_current = torch.ones([sum(imbalanced_num_list)]).cuda()
    
    loss_n_mean_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    loss_n_mean_current = torch.ones([sum(imbalanced_num_list)]).cuda()

    margin_n_mean_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    margin_n_mean_current = torch.ones([sum(imbalanced_num_list)]).cuda()

    grad_n_mean_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    grad_n_mean_current = torch.ones([sum(imbalanced_num_list)]).cuda()

    logit_n_mean_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    logit_n_mean_current = torch.ones([sum(imbalanced_num_list)]).cuda()

    pred_n_mean_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    pred_n_mean_current = torch.ones([sum(imbalanced_num_list)]).cuda()

    entropy_n_mean_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    entropy_n_mean_current = torch.ones([sum(imbalanced_num_list)]).cuda()

    current_class_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    current_class_current = torch.ones([sum(imbalanced_num_list)]).cuda()

    other_class_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    other_class_current = torch.ones([sum(imbalanced_num_list)]).cuda()

    neighbor_feature_distance_last = torch.ones([sum(imbalanced_num_list)]).cuda()
    neighbor_feature_distance_current = torch.ones([sum(imbalanced_num_list)]).cuda()

    for epoch in range(args.max_epoch):
        lr = args.lr * ((0.01 ** int(epoch >= args.epoch1)) * (0.01 ** int(epoch >= args.epoch2)))
        for group in optimizer.param_groups:
            group['lr'] = lr

        logger.info(f'Training epoch {epoch}')
        for iteration, (index, inputs, labels) in enumerate(train_dataloader):
            net.train()
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            if (iteration + 1) % args.meta_interval == 0:
                pseudo_net = ResNet32(args.dataset == 'cifar10' and 10 or 100).to(args.device)
                pseudo_net.load_state_dict(net.state_dict())
                pseudo_net.train()
                
                pseudo_feature, pseudo_outputs = pseudo_net(inputs)
                pseudo_sim_model = CosineSimilarity().cuda()
                pseudo_dist = pseudo_sim_model(pseudo_feature, pseudo_feature) 
                pseudo_neighbor_value, pseudo_neighbor_index = torch.topk(pseudo_dist, args.nk, dim=0) 
                pseudo_neighbor_index_t = pseudo_neighbor_index.t() 
                pseudo_neighbor_labels = torch.gather(labels.unsqueeze(0).repeat(128, 1), 1, pseudo_neighbor_index_t) 
            
                pseudo_one_hot = torch.nn.functional.one_hot(pseudo_neighbor_labels, num_classes=args.num_classes)
                pseudo_neighbor_classes = torch.sum(pseudo_one_hot, dim=1) 

                pseudo_current_classes = torch.gather(pseudo_neighbor_classes, 1, labels.unsqueeze(1))
                pseudo_current_classes = pseudo_current_classes.squeeze()/args.nk
            
                pseudo_mask = torch.arange(args.num_classes).unsqueeze(0).cuda() != labels.unsqueeze(1)
                pseudo_other_classes = torch.max(torch.where(pseudo_mask, pseudo_neighbor_classes, torch.full_like(pseudo_neighbor_classes, -float('inf'), dtype=torch.float32)), dim=1).values
                pseudo_other_classes = pseudo_other_classes/args.nk

                neighbor_features = pseudo_feature[pseudo_neighbor_index_t, :]

                mean_neighbor_features = torch.mean(neighbor_features, dim=1)

                feature_norms = torch.norm(pseudo_feature, dim=1, keepdim=True)
                mean_neighbor_norms = torch.norm(mean_neighbor_features, dim=1, keepdim=True)
                cosine_similarities = torch.mm(pseudo_feature, mean_neighbor_features.t()) / torch.mm(feature_norms, mean_neighbor_norms.t())

                sample_similarities = torch.diagonal(cosine_similarities)

                pseudo_sim = sample_similarities.unsqueeze(1)


                pseudo_loss_vector = F.cross_entropy(pseudo_outputs, labels.long(), reduction='none')
                pseudo_feature = weight_feature(pseudo_outputs,pseudo_loss_vector,labels,index,class_rate, pseudo_neighbor_index_t,
                                                loss_tensor_last, margin_tensor_last, grad_tensor_last, logit_tensor_last, pred_tensor_last,  entropy_tensor_last, diff_loss_n_last, diff_margin_n_last, diff_grad_n_last, diff_logit_n_last, diff_pred_n_last, diff_entropy_n_last, 
                                                loss_n_mean_last, margin_n_mean_last,
                                                grad_n_mean_last, logit_n_mean_last, pred_n_mean_last, entropy_n_mean_last, pseudo_current_classes, pseudo_other_classes, pseudo_sim, current_class_last, other_class_last, neighbor_feature_distance_last)
                
                pseudo_perturb = meta_net(pseudo_feature)
                pseudo_loss = F.cross_entropy(pseudo_outputs+args.tau1 * adjustments+args.tau2 * pseudo_perturb, labels.long())
                pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)
                pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)

                del pseudo_grads

                try:
                    _, meta_inputs, meta_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_dataloader)
                    _, meta_inputs, meta_labels = next(meta_dataloader_iter)

                meta_inputs, meta_labels = meta_inputs.to(args.device), meta_labels.to(args.device)
                meta_features, meta_outputs = pseudo_net(meta_inputs)
                meta_loss = criterion(meta_outputs, meta_labels.long())

                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

            feature, outputs = net(inputs)
            sim_model = CosineSimilarity().cuda()
            dist = sim_model(feature, feature) # 
            neighbor_value, neighbor_index = torch.topk(dist, args.nk, dim=0) 
            neighbor_index_t = neighbor_index.t() 

            neighbor_labels = torch.gather(labels.unsqueeze(0).repeat(128, 1), 1, neighbor_index_t) 
            
            one_hot = torch.nn.functional.one_hot(neighbor_labels, num_classes=args.num_classes)
            neighbor_classes = torch.sum(one_hot, dim=1) 

            current_classes = torch.gather(neighbor_classes, 1, labels.unsqueeze(1))
            current_classes = current_classes.squeeze()
            current_classes = current_classes/args.nk
            
            mask = torch.arange(args.num_classes).unsqueeze(0).cuda() != labels.unsqueeze(1)
            other_classes = torch.max(torch.where(mask, neighbor_classes, torch.full_like(neighbor_classes, -float('inf'), dtype=torch.float32)), dim=1).values
            other_classes = other_classes/args.nk
          
            
            neighbor_features1 = feature[neighbor_index_t, :]
            mean_neighbor_features1 = torch.mean(neighbor_features1, dim=1)
            feature_norms1 = torch.norm(feature, dim=1, keepdim=True)
            mean_neighbor_norms1 = torch.norm(mean_neighbor_features1, dim=1, keepdim=True)
            cosine_similarities1 = torch.mm(feature, mean_neighbor_features1.t()) / torch.mm(feature_norms1, mean_neighbor_norms1.t())

            sample_similarities1 = torch.diagonal(cosine_similarities1)
            sim = sample_similarities1.unsqueeze(1)  



            loss_vector = F.cross_entropy(outputs, labels.long(), reduction='none')
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))

            with torch.no_grad():
                feature = weight_feature(outputs,loss_vector,labels,index, class_rate,neighbor_index_t,
                                                loss_tensor_last, margin_tensor_last, grad_tensor_last, logit_tensor_last, pred_tensor_last,  entropy_tensor_last, diff_loss_n_last, diff_margin_n_last, diff_grad_n_last, diff_logit_n_last, diff_pred_n_last, diff_entropy_n_last,
                                               loss_n_mean_last, margin_n_mean_last,
                                                grad_n_mean_last, logit_n_mean_last, pred_n_mean_last, entropy_n_mean_last, current_classes, other_classes, sim, current_class_last, other_class_last, neighbor_feature_distance_last)
                perturb = meta_net(feature)
            
            loss = F.cross_entropy(outputs+args.tau1 * adjustments+args.tau2 * perturb, labels.long())
            
            loss_tensor_current.scatter_(0, index.cuda(), feature[:,0])
            logit_tensor_current.scatter_(0, index.cuda(), feature[:,1])
            pred_tensor_current.scatter_(0, index.cuda(), feature[:,2])
            grad_tensor_current.scatter_(0, index.cuda(), feature[:,3])
            margin_tensor_current.scatter_(0, index.cuda(), feature[:,4])
            entropy_tensor_current.scatter_(0, index.cuda(), feature[:,5])

            loss_n_mean_current.scatter_(0, index.cuda(), feature[:,6])
            margin_n_mean_current.scatter_(0, index.cuda(), feature[:,7])
            grad_n_mean_current.scatter_(0, index.cuda(), feature[:,8])
            logit_n_mean_current.scatter_(0, index.cuda(), feature[:,9])
            pred_n_mean_current.scatter_(0, index.cuda(), feature[:,10])
            entropy_n_mean_current.scatter_(0, index.cuda(), feature[:,11])
            diff_loss_n_current.scatter_(0, index.cuda(), feature[:,12])
            diff_margin_n_current.scatter_(0, index.cuda(), feature[:,13])
            diff_grad_n_current.scatter_(0, index.cuda(), feature[:,14])
            diff_logit_n_current.scatter_(0, index.cuda(), feature[:,15])
            diff_pred_n_current.scatter_(0, index.cuda(), feature[:,16])
            diff_entropy_n_current.scatter_(0, index.cuda(), feature[:,17])

            current_class_current.scatter_(0, index.cuda(), feature[:,18])
            other_class_current.scatter_(0, index.cuda(), feature[:,19])
            neighbor_feature_distance_current.scatter_(0, index.cuda(), feature[:,20])


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_tensor_last = copy.deepcopy(loss_tensor_current)
        logit_tensor_last = copy.deepcopy(logit_tensor_current)
        pred_tensor_last = copy.deepcopy(pred_tensor_current)
        margin_tensor_last = copy.deepcopy(margin_tensor_current)
        entropy_tensor_last = copy.deepcopy(entropy_tensor_current)
        grad_tensor_last = copy.deepcopy(grad_tensor_current)
        diff_loss_n_last = copy.deepcopy(diff_loss_n_current)
        diff_margin_n_last = copy.deepcopy(diff_margin_n_current)
        diff_grad_n_last = copy.deepcopy(diff_grad_n_current)
        diff_entropy_n_last = copy.deepcopy(diff_entropy_n_current)
        diff_logit_n_last = copy.deepcopy(diff_logit_n_current)
        diff_pred_n_last = copy.deepcopy(diff_pred_n_current)

        loss_n_mean_last = copy.deepcopy(loss_n_mean_current)
        margin_n_mean_last = copy.deepcopy(margin_n_mean_current)
        grad_n_mean_last = copy.deepcopy(grad_n_mean_current)
        logit_n_mean_last = copy.deepcopy(logit_n_mean_current)
        pred_n_mean_last = copy.deepcopy(pred_n_mean_current)
        entropy_n_mean_last = copy.deepcopy(entropy_n_mean_current)
        current_class_last = copy.deepcopy(current_class_current)
        other_class_last = copy.deepcopy(other_class_current)
        neighbor_feature_distance_last = copy.deepcopy(neighbor_feature_distance_current)
        
        logger.info('Computing Test Result...')

        test_loss, test_accuracy = compute_loss_accuracy(
            net=net,
            data_loader=test_dataloader,
            criterion=criterion,
            device=args.device,
        )
        
        if test_accuracy>best_acc:
            best_acc = test_accuracy
            print(best_acc)
            torch.save(net.state_dict(),"best_net.pt")
            torch.save(meta_net.state_dict(),"best_meta.pt")
            
        logger.info('Epoch: {}, (Loss, Accuracy) Test: ({:.4f}, {:.2%}) LR: {}'.format(
            epoch,
            test_loss,
            test_accuracy,
            lr,
        ))
    time_end = time.time()
    time_consume = time_end-time_begin
    print("time_consume")
    print(time_consume)
    logger.info(f'best_accuracy: {best_acc}')   

if __name__ == '__main__':

    meta_GLA()
