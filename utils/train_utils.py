import os
import time
import torch
import torch.nn.functional as F
import numpy as np

PATH_to_log_dir = './log_dir'

from torch.utils.tensorboard import SummaryWriter

def adjust_learning_rate(lr_deacy, optimizer, epoch, step=1):
    if optimizer is None:
        return
    if epoch>0 and epoch % step ==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_deacy
    return


def adjust_temperature(optimizer,args):
    decay_factor  = (args.temp_min/args.temperature)**(1.0/args.epochs)  # learning rate decay factor
    for param_group in optimizer.param_groups:
        param_group['temperature'] = np.maximum(param_group['temperature'] * decay_factor, 1e-20) # param_group['temperature'] * tp_deacy


def softmax_predictive_accuracy(logits_list, y, criterion, ret_loss=False):
    # this function used to compute the accuracy of average predictions
    probs_list = [logits for logits in logits_list]
    probs_tensor = torch.stack(probs_list, dim = 2)
    probs = torch.mean(probs_tensor, dim=2) # the prediction is the average of all sampling predictions
    if ret_loss:
        loss = criterion(probs, y).item()
        #loss = criterion(probs, y, reduction='sum').item()
    _, pred_class = torch.max(probs, 1)
    correct = pred_class.eq(y.view_as(pred_class)).sum().item()
    if ret_loss:
        return correct, loss
    return correct

def train_model(args, model, dataloaders, criterion, optimizer, bn_optimizer=None):
    """
    Performs Training and Validation on train/val set on the given model using the specified optimizer
    :param model: (nn.Module) Model to be trained
    :param dataloaders: (list) train, val and test dataloaders
    :param criterion: Loss Function
    :param optimizer: Optimizer to be used for training
    :param bn_optimizer: Optimizer for the float point real-valued BN layer parameters
    :return: trained model, val and train metric history
    """
    train_loss_hist = []
    train_accuracy_hist = []
    val_accuracy_hist = []
    val_loss_hist = []
    test_accuracy_hist = []
    test_loss_hist = []
    trainloader, valloader, testloader = dataloaders

    best_acc = 0


    PATH_to_log_dir = os.path.join(args.out_dir, 'log_dir_{}'.format(args.experiment_id))
    writer = SummaryWriter(PATH_to_log_dir)
    
    if args.lrschedular == 'Mstep':
        opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//2, args.epochs//4*3,150,250,350,450], gamma = 0.1)
        if bn_optimizer is not None:
            bn_scheduler = torch.optim.lr_scheduler.MultiStepLR(bn_optimizer, milestones=[args.epochs//2, args.epochs//4*3,150,250,350,450], gamma = 0.1)
    elif args.lrschedular == 'Expo':
        gamma = (1e-6)**(1.0/args.epochs)
        opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma, last_epoch=-1)
        if bn_optimizer is not None:
            bn_scheduler = torch.optim.lr_scheduler.ExponentialLR(bn_optimizer, gamma = gamma, last_epoch=-1)

    elif args.lrschedular == 'Cosine':
        opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-16, last_epoch=-1)
        if bn_optimizer is not None:
            bn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(bn_optimizer,T_max = args.epochs, eta_min = 1e-16, last_epoch=-1)

    else:
        raise ValueError('Wrong LR schedule!!')
        


    #################################
    # Training and Evaluation Part
    global_step = 0

    for epoch in range(args.epochs):
        model.train(True)
        print('Epoch[%d]:' % epoch)

        # learning rate decaly
        opt_scheduler.step()
        if bn_optimizer is not None:
            bn_scheduler.step()

        if args.optim == 'BayesBiNN':
            adjust_temperature(optimizer,args)

        running_train_loss = 0.
        running_train_correct = 0.
        running_train_samples = 0.
        for i, data in enumerate(trainloader):

            global_step += i


            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            if args.optim == 'BayesBiNN' and bn_optimizer is not None: #here we only perform optimization for BayesBiNN optimizer
                bn_optimizer.zero_grad()
                logits = model.forward(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                bn_optimizer.step()

            if args.optim == 'STE':
                optimizer.zero_grad()
                output = model.forward(inputs)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()

                for p in list(model.parameters()):
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)

                optimizer.step()

                for p in list(model.parameters()):
                    if hasattr(p, 'org'):
                        p.org.copy_(p.data.clamp_(-1, 1))

            else:
                if args.optim == 'BayesBiNN':
                    def closure():
                        optimizer.zero_grad()
                        output = model.forward(inputs)
                        loss = criterion(output, labels) #
                        return loss, output
                else:
                    def closure():
                        optimizer.zero_grad()
                        logits = model.forward(inputs)
                        loss = criterion(logits, labels)
                        loss.backward()
                        return loss, logits

                loss, output = optimizer.step(closure)

            if isinstance(output, list):
                output = output[0]

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(labels.view_as(pred)).sum().item()
            running_train_loss += loss.detach().item() * inputs.shape[0]
            running_train_correct += correct
            running_train_samples += inputs.shape[0]

            # Print Training Progress
            if i % args.log_interval == 0 and i > 0:
                if args.optim == 'BayesBiNN':
                    for param_group in optimizer.param_groups:
                        print('Current temperature is {}!'.format(param_group['temperature']))

                train_accuracy = running_train_correct / running_train_samples
                train_loss = running_train_loss / running_train_samples
                print('Iteration[%d]: Train Loss: %f   Train Accuracy: %f ' % (i+1, train_loss, train_accuracy))

        train_accuracy = 100 * running_train_correct / len(trainloader.sampler)
        train_loss = running_train_loss / len(trainloader.sampler)
        train_accuracy_hist.append(train_accuracy)
        train_loss_hist.append(train_loss)
        print('## Epoch[%d], Train Loss: %f   &   Train Accuracy: %f' % (epoch, train_loss, train_accuracy))



        ################## Evaluation ###################
        if valloader is not None:
            val_loss, val_accuracy = test_model(args, model, valloader, criterion, optimizer, bn_optimizer)
            val_accuracy_hist.append(val_accuracy)
            val_loss_hist.append(val_loss)
            print('## Epoch[%d], Val Loss:   %f   &   Val Accuracy:   %f' % (epoch, val_loss, val_accuracy))

            # remember best acc@1 and save checkpoint
            is_best = val_accuracy > best_acc
            best_acc = max(val_accuracy, best_acc)


            if is_best and args.save_model:
                state = {
                    'epoch': epoch+1,
                    'best_acc1': best_acc,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_path = os.path.join(args.out_dir, 'saved_models', 'model_{}_checkpoint_best.ckpt'.format(args.experiment_id))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(state, save_path)

        if testloader is not None:
            test_loss, test_accuracy = test_model(args, model, testloader, criterion, optimizer, bn_optimizer)
            test_accuracy_hist.append(test_accuracy)
            test_loss_hist.append(test_loss)
            print('## Epoch[%d], Test Loss:  %f   &   Test Accuracy:  %f' % (epoch, test_loss, test_accuracy))
        print('')


    writer.close()

    return model, train_loss_hist, train_accuracy_hist, val_loss_hist, val_accuracy_hist, test_loss_hist, test_accuracy_hist

def test_model(args, model, test_loader, criterion, optimizer, bn_optimizer):
    model.eval()
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)

            if optimizer is not None and args.optim == 'BayesBiNN':
                raw_noises = []
               
                if args.test_samples <= 0:
                    raw_noises = None
                else:
                    for mc_sample in range(args.test_samples):
                        raw_noises.append(torch.bernoulli(torch.sigmoid(2*optimizer.state['lamda'])))
                outputs = optimizer.get_mc_predictions(model, data, raw_noises=raw_noises)
                correct, loss = softmax_predictive_accuracy(outputs, target, criterion, ret_loss=True)
                total_correct += correct
                test_loss += loss
            else:
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.sampler)
    test_accuracy = 100. * total_correct / len(test_loader.sampler)

    return test_loss, test_accuracy
