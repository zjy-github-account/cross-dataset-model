from torch.utils.data import DataLoader
from utils import *
import torch.nn as nn
import torch.nn.functional as F

CUDA = torch.cuda.is_available()

def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        out = net(x_batch)

        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train

def train_one_epoch_EEGARNN(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        out = net(x_batch)

        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())

        A = net.A
        with torch.no_grad():
            W_grad = A.grad
            A = (1 - 0.001) * A - 0.001 * W_grad
        A = nn.Parameter(A, requires_grad=False)
        optimizer.step()  # Updating Parameters
        A = nn.Parameter(A)
        net.A = A

    return tl.item(), pred_train, act_train

def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        ADJ = []
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            out = net(x_batch)
            # ADJ.append(adj)

            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

def train(args, data_train, data_val, sub, val_best=None, model_F = False, kfold = None):
    seed_all(args.random_seed)
    set_up(args)

    train_loader = DataLoader(data_train, args.batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(data_val, args.batch_size, shuffle=False, pin_memory=True)

    if model_F:
        model = model_F
    else:
        model = get_model(args)
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['min_loss'] = 0.0
    trlog['F1'] = 0.0

    # timer = Timer()
    if kfold != None:
        model_path = osp.join(args.save_path, str(args.model))+'/sub'+str(sub)+'_'+str(kfold)+'.pth'
    else:
        model_path = osp.join(args.save_path,str(args.model),'{}.pth'.format('sub'+str(sub)))

    ensure_path(osp.join(args.save_path, str(args.model)))
    # model_path = osp.join(args.save_path, '{}.pth'.format('candidate_foldlanse_test' + str(fold)))
    early_stopping = EarlyStopping(patience=args.patient, checkpoint_path=model_path, best_score=val_best)
    for epoch in range(1, args.max_epoch + 1):
        if args.model == 'EEGARNN':
            loss_train, pred_train, act_train = train_one_epoch_EEGARNN(
                data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)
        else:
            loss_train, pred_train, act_train = train_one_epoch(
                data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        # print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch), fold))

        mode = 'acc'
        if mode == 'acc':
            early_stopping(acc_val, model, mode, f1_val)
        elif mode == 'loss':
            early_stopping(loss_val, model, mode, f1_val)
        if early_stopping.early_stop:
            break
        trlog['max_acc'] = max(trlog['val_acc'])
        trlog['min_loss'] = min(trlog['val_loss'])
        trlog['F1'] = early_stopping.f1score

    if mode == 'acc':
        return trlog['max_acc'], trlog['F1'], model, pred_val, act_val
    elif mode == 'loss':
        return trlog['val_acc'][np.argmin(trlog['val_loss'])], trlog['F1'], model, pred_val, act_val

def test(args, test_loader,model_path):
    set_up(args)
    seed_all(args.random_seed)

    model = get_model(args)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(model_path))
    test_loader = DataLoader(test_loader, args.batch_size, shuffle=False, pin_memory=True)
    loss, pred, act = predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn
    )
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act, f1

def combine_train(args, train_stu, val_stu, train_tea, fold, val_best=None):
    seed_all(args.random_seed)
    set_up(args)

    train_loader_stu = DataLoader(train_stu, args.batch_size, shuffle=False, pin_memory=True)
    val_loader_stu = DataLoader(val_stu, args.batch_size, shuffle=False, pin_memory=True)
    train_loader_tea = DataLoader(train_tea, args.batch_size, shuffle=False, pin_memory=True)

    # teacher model building
    args.model = 'IGNNNet'
    model_tea = get_model(args)
    # model_tea.load_state_dict(torch.load('./save_cross/candidate_foldKU_test' + str(fold) + '.pth'))
    model_tea.load_state_dict(torch.load('./save_cross/48channeltest' + str(fold) + '.pth'))
    args.model = 'IGNNNet_stu'
    args.Input_shape = (1, 8, 1000)
    model_stu = get_model(args)

    if CUDA:
        model_tea = model_tea.cuda()
        model_stu = model_stu.cuda()

    optimizer = torch.optim.Adam(model_stu.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['min_loss'] = 0.0
    trlog['F1'] = 0.0

    # model_path = osp.join(args.save_path, '{}.pth'.format('candidate_stu_test'))
    model_path = osp.join(args.save_path, '{}.pth'.format('48channeltest_stu_test'))
    early_stopping = EarlyStopping(patience=args.patient, checkpoint_path=model_path, best_score=val_best)
    for epoch in range(1, args.max_epoch + 1):

        loss_train, pred_train, act_train = train_one_epoch_stu(
                                            data_stu=train_loader_stu, data_tea=train_loader_tea,
                                            model_tea=model_tea, model_stu=model_stu,
                                            loss_fn=loss_fn, optimizer=optimizer)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader_stu, net=model_stu, loss_fn=loss_fn
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        # print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch), fold))

        mode = 'acc'
        if mode == 'acc':
            early_stopping(acc_val, model_stu, mode, f1_val)
        elif mode == 'loss':
            early_stopping(loss_val, model_stu, mode, f1_val)
        if early_stopping.early_stop:
            break
        trlog['max_acc'] = max(trlog['val_acc'])
        trlog['min_loss'] = min(trlog['val_loss'])
        trlog['F1'] = early_stopping.f1score
    # save the training log file
    # save_name = 'trlog' + save_name
    # experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    # save_path = osp.join(args.save_path, experiment_setting, 'log_train')
    # ensure_path(save_path)
    # torch.save(trlog, osp.join(save_path, save_name))

    if mode == 'acc':
        return trlog['max_acc'], trlog['F1']
    elif mode == 'loss':
        return trlog['val_acc'][np.argmin(trlog['val_loss'])], trlog['F1']

def train_one_epoch_stu(data_stu,data_tea, model_tea, model_stu, loss_fn, optimizer):
    model_stu.train()
    tl = Averager()
    pred_train = []
    act_train = []
    epoch_losses1 = []
    epoch_losses2 = []
    epoch_losses = []
    for (x_batch_stu, y_batch_stu), (x_batch_tea, y_batch_tea) in zip(data_stu, data_tea):
        if CUDA:
            x_batch_stu, y_batch_stu = x_batch_stu.cuda(), y_batch_stu.cuda()
            x_batch_tea = x_batch_tea.cuda()
        # teacher model
        with torch.no_grad():
            x1 = model_tea.Tception1(x_batch_tea)
            x2 = model_tea.Tception2(x_batch_tea)
            x3 = model_tea.Tception3(x_batch_tea)
            out_tea = torch.cat((x1, x2, x3), dim=1)
            out_tea = model_tea.BN_t(out_tea)
            out_tea = model_tea.D_Block1(out_tea)
            out_tea = model_tea.D_Block2(out_tea)

            adj_tea = model_tea.get_adj(out_tea)
            out1_tea = model_tea.GCN1(out_tea, adj_tea)
            out1_tea = model_tea.BN_1(out1_tea)
            out1_tea = model_tea.Dropout1(out1_tea)

            channel_8 = [10, 14, 4, 12, 7, 20, 13, 17]
            out1_tea = out1_tea[:, :, np.array(channel_8), :]
            out_tea = out_tea[:, :, np.array(channel_8), :]
            GCN_TEA = out1_tea
            out1_tea = model_tea.DWConv1(out1_tea)
            out1_tea = model_tea.midblock1(out1_tea)
            out1_tea = model_tea.block1(out1_tea)
            out_tea = model_tea.DWConv2(out_tea)
            out_tea = model_tea.midblock2(out_tea)
            out_tea = model_tea.block2(out_tea)
            # out_tea = torch.cat([out_tea, out1_tea], dim=1)
            # out = model_tea.classifier(out)
        # student
        z1 = model_stu.Tception1(x_batch_stu)
        z2 = model_stu.Tception2(x_batch_stu)
        z3 = model_stu.Tception3(x_batch_stu)
        out_stu = torch.cat((z1, z2, z3), dim=1)
        out_stu = model_stu.BN_t(out_stu)
        out_stu = model_stu.D_Block1(out_stu)
        out_stu = model_stu.D_Block2(out_stu)

        adj_stu = model_stu.get_adj(out_stu)
        out1_stu = model_stu.GCN1(out_stu, adj_stu)
        out1_stu = model_stu.BN_1(out1_stu)
        out1_stu = model_stu.Dropout1(out1_stu)
        GCN_STU = out1_stu
        out1_stu = model_stu.DWConv1(out1_stu)
        out1_stu = model_stu.midblock1(out1_stu)
        out1_stu = model_stu.block1(out1_stu)

        out_stu = model_stu.DWConv2(out_stu)
        out_stu = model_stu.midblock2(out_stu)
        out_stu = model_stu.block2(out_stu)
        out_stu = torch.cat([out_stu, out1_stu], dim=1)

        out_stu = model_stu.classifier(out_stu)

        loss1 = loss_fn(out_stu, y_batch_stu)
        loss2 = F.mse_loss(GCN_STU, GCN_TEA)
        loss = loss1 + loss2

        _, pred = torch.max(out_stu, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch_stu.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())

        epoch_losses1.append(loss1.item())
        epoch_losses2.append(loss2.item())
        epoch_losses.append(loss.item())
    avg_loss1 = sum(epoch_losses1) / len(epoch_losses1)
    avg_loss2 = sum(epoch_losses2) / len(epoch_losses2)
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f'Average Loss1: {avg_loss1}, Average Loss2: {avg_loss2}, Average Loss: {avg_loss}')

    return tl.item(), pred_train, act_train

def train_one_epoch_stu2(data_stu,data_tea, model_tea, model_stu, loss_fn, optimizer):
    model_stu.train()
    tl = Averager()
    pred_train = []
    act_train = []
    epoch_losses1 = []
    epoch_losses2 = []
    epoch_losses3 = []
    epoch_losses = []
    for (x_batch_stu, y_batch_stu), (x_batch_tea, y_batch_tea) in zip(data_stu, data_tea):
        if CUDA:
            x_batch_stu, y_batch_stu = x_batch_stu.cuda(), y_batch_stu.cuda()
            x_batch_tea = x_batch_tea.cuda()
        # teacher model
        with torch.no_grad():
            x1 = model_tea.Tception1(x_batch_tea)
            x2 = model_tea.Tception2(x_batch_tea)
            x3 = model_tea.Tception3(x_batch_tea)
            out_tea = torch.cat((x1, x2, x3), dim=1)
            out_tea = model_tea.BN_t(out_tea)
            out_tea = model_tea.D_Block1(out_tea)
            out_tea = model_tea.D_Block2(out_tea)

            adj_tea = model_tea.get_adj(out_tea)
            out1_tea = model_tea.GCN1(out_tea, adj_tea)
            out1_tea = model_tea.BN_1(out1_tea)
            out1_tea = model_tea.Dropout1(out1_tea)

            channel_8 = [10, 14, 4, 12, 7, 20, 13, 17]
            out1_tea = out1_tea[:, :, np.array(channel_8), :]
            out_tea = out_tea[:, :, np.array(channel_8), :]
            GCN_TEA = out1_tea
            CNN_TEA = out_tea
            out1_tea = model_tea.DWConv1(out1_tea)
            out1_tea = model_tea.midblock1(out1_tea)
            out1_tea = model_tea.block1(out1_tea)
            out_tea = model_tea.DWConv2(out_tea)
            out_tea = model_tea.midblock2(out_tea)
            out_tea = model_tea.block2(out_tea)
            # out_tea = torch.cat([out_tea, out1_tea], dim=1)
            # out = model_tea.classifier(out)
        # student
        z1 = model_stu.Tception1(x_batch_stu)
        z2 = model_stu.Tception2(x_batch_stu)
        z3 = model_stu.Tception3(x_batch_stu)
        out_stu = torch.cat((z1, z2, z3), dim=1)
        out_stu = model_stu.BN_t(out_stu)
        out_stu = model_stu.D_Block1(out_stu)
        out_stu = model_stu.D_Block2(out_stu)

        adj_stu = model_stu.get_adj(out_stu)
        out1_stu = model_stu.GCN1(out_stu, adj_stu)
        out1_stu = model_stu.BN_1(out1_stu)
        out1_stu = model_stu.Dropout1(out1_stu)
        GCN_STU = out1_stu
        CNN_STU = out_stu
        out1_stu = model_stu.DWConv1(out1_stu)
        out1_stu = model_stu.midblock1(out1_stu)
        out1_stu = model_stu.block1(out1_stu)

        out_stu = model_stu.DWConv2(out_stu)
        out_stu = model_stu.midblock2(out_stu)
        out_stu = model_stu.block2(out_stu)
        out_stu = torch.cat([out_stu, out1_stu], dim=1)

        out_stu = model_stu.classifier(out_stu)

        loss1 = loss_fn(out_stu, y_batch_stu)
        loss2 = F.mse_loss(GCN_STU, GCN_TEA)
        loss3 = F.mse_loss(CNN_STU, CNN_TEA)
        loss = loss1 + loss2 + loss3

        _, pred = torch.max(out_stu, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch_stu.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())

        epoch_losses1.append(loss1.item())
        epoch_losses2.append(loss2.item())
        epoch_losses3.append(loss3.item())
        epoch_losses.append(loss.item())
    avg_loss1 = sum(epoch_losses1) / len(epoch_losses1)
    avg_loss2 = sum(epoch_losses2) / len(epoch_losses2)
    avg_loss3 = sum(epoch_losses3) / len(epoch_losses3)
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f'Average Loss1: {avg_loss1}, Average Loss2: {avg_loss2}, Average Loss3: {avg_loss3}, Average Loss: {avg_loss}')

    return tl.item(), pred_train, act_train

def train_one_epoch_finetune(data_loader, net, loss_fn, optimizer):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001)
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        for param in net.parameters():
            param.requires_grad = True
        for param in net.Tception1.parameters():
            param.requires_grad = False
        for param in net.Tception2.parameters():
            param.requires_grad = False
        for param in net.Tception3.parameters():
            param.requires_grad = False
        # for param in net.BN_t.parameters():
        #     param.requires_grad = False
        # for param in net.D_Block1.parameters():
        #     param.requires_grad = False
        # for param in net.D_Block2.parameters():
        #     param.requires_grad = False
        # for param in net.DWConv1.parameters():
        #     param.requires_grad = True
        # for param in net.DWConv2.parameters():
        #     param.requires_grad = True
        # for param in net.midblock1.parameters():
        #     param.requires_grad = True
        # for param in net.midblock2.parameters():
        #     param.requires_grad = True
        # for param in net.block1.parameters():
        #     param.requires_grad = True
        # for param in net.block2.parameters():
        #     param.requires_grad = True

        out = net(x_batch)

        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train