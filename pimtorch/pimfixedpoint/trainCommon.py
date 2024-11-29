import torch
import logging
import time
from matplotlib import pyplot as plt
from torch import Generator
from torch.utils.data import DataLoader, random_split
from fixedPoint.nn.earlystopping import EarlyStopping
from fixedPoint.nn import fixedPointArithmetic as fpA
from fixedPoint.nn.commonConst import variations, phyArrParams


def split_data_loader(train_datasets, test_datasets, train_kwargs, test_kwargs, alpha=0.2, seed=1,
                        extra_train_datasets_for_valid=None):
    if extra_train_datasets_for_valid is None:
        train_loader = DataLoader(train_datasets, **train_kwargs)
        valid_loader = None
    else:
        full_train_size = len(train_datasets)
        valid_size = int(full_train_size * alpha)
        train_size = full_train_size - valid_size

        sub_train_datasets, _ = random_split(
            dataset=train_datasets,
            lengths=[train_size, valid_size],
            generator=Generator().manual_seed(seed)
        )

        _, sub_valid_datasets = random_split(
            dataset=extra_train_datasets_for_valid,
            lengths=[train_size, valid_size],
            generator=Generator().manual_seed(seed)
        )

        train_loader = DataLoader(sub_train_datasets, **train_kwargs)
        valid_loader = DataLoader(sub_valid_datasets, **test_kwargs)

    test_loader = DataLoader(test_datasets, **test_kwargs)

    return train_loader, test_loader, valid_loader


def train_model(model, device, train_loader, valid_loader, criterion, optimizer, n_epochs, patience=20,
                model_filename='model/network_checkpoint.pt', verbose=True, score_type='loss', scheduler=None,
                half=False, save_trace=False, trace_filename="trace/other_network/", single_target = True):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    valid_acc_list = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(filename=model_filename, patience=patience, verbose=verbose, score_type=score_type)

    mid_batch_of_epoch = train_loader.__len__() // 2

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            if half:
                data = data.half()
            output = model(data)

            # logging.debug("input = {}".format(data))
            # logging.debug("predict output = {}".format(output))
            # logging.debug("true output = {}".format(target))

            # calculate the loss
            a = "{}".format(criterion)
            if a=="MSELoss()" and single_target:
                target_new = output.new_zeros(output.size(0), output.size(1))
                for i in range(data.size(0)):
                    target_new[target[i]] = 1.0
                loss = criterion(output, target_new)
            else:
                loss = criterion(output, target)
            
            if phyArrParams.useBNN:
                #! something added here
                if epoch%40==0:
                    optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            if phyArrParams.useBNN:
                #! something added here
                for p in list(model.parameters()):
                    if hasattr(p,'org'):
                        p.data.copy_(p.org)

            # perform a single optimization step (parameter update)
            optimizer.step()

            if phyArrParams.useBNN:
                #! something added here
                for p in list(model.parameters()):
                    if hasattr(p,'org'):
                        p.org.copy_(p.data.clamp_(-1,1))

            # record training loss
            train_losses.append(loss.item())

            if save_trace:
                if batch == mid_batch_of_epoch:
                    torch.save(model.state_dict(), trace_filename + "epoch_" + str(epoch) + "_before_mid.pt")
                elif batch == mid_batch_of_epoch + 1:
                    torch.save(model.state_dict(), trace_filename + "epoch_" + str(epoch) + "_after_mid.pt")

        ######################
        # validate the model #
        ######################
        correct = 0
        # model.eval()  # prep model for evaluation
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                if half:
                    data = data.half()
                output = model(data)
                if single_target:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                # calculate the loss
                a = "{}".format(criterion)
                if a=="MSELoss()" and single_target:
                    target_new = output.new_zeros(output.size(0), output.size(1))    
                    for i in range(data.size(0)):
                        target_new[target[i]] = 1.0
                    loss = criterion(output, target_new)
                else:
                    loss = criterion(output, target)
                # record validation loss
                valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = sum(train_losses)/len(train_losses)
        valid_loss = sum(valid_losses)/len(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        valid_acc = correct / len(valid_loader.dataset)

        valid_acc_list.append(valid_acc)

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                    f'train_loss: {train_loss:.8f} ' +
                    f'valid_loss: {valid_loss:.8f} ' +
                    f'valid acc: {100. * valid_acc:.2f}%')

        print(print_msg, flush=True)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        if scheduler is not None:
            scheduler.step()

        # if save_trace is True:
        #     torch.save(model.state_dict(), "epoch" + str(epoch) + filename)

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        if score_type == 'accuracy':
            early_stopping(valid_acc, model)
        else:
            early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(model_filename))

    return avg_train_losses, avg_valid_losses, valid_acc_list


def train_full_data(model, device, train_loader, criterion, optimizer, n_epochs, filename='full_data.pt'):
    # to track the training loss as the model trains
    train_losses = []

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = sum(train_losses)/len(train_losses)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                    f'train_loss: {train_loss:.4f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []

    # save the model
    torch.save(model.state_dict(), filename)


def test_model(model, device, test_loader, criterion, half=False, single_target=True):
    # initialize lists to monitor test loss and accuracy
    correct = 0

    model.eval()  # prep model for evaluation

    item_size = 0
    T1 = time.time()
    test_loss = []
    with torch.no_grad():
        for data, target in test_loader:
            torch.manual_seed(0)

            #! reset torch seed for eNVM cell conductance variation at the beginning of each batch
            #! reset for the entire network and mirrored
            variations.network_gen_reset()
            variations.mirrored_gen_reset()

            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            if half:
                data = data.half()
            output = model(data)
            logging.debug("input = {}".format(data))
            logging.debug("predict output = {}".format(output))
            logging.debug("true output = {}".format(target))
            T2 = time.time()
            # calculate the loss
            current_loss = criterion(output, target).item()
            test_loss.append(current_loss)
            if single_target:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                item_size += data.size(0)
                print("now corrent item = {}, all item = {}, total use time = {:.4f} s".format(correct, item_size, T2-T1), flush=True)

    # calculate and print avg test loss
    test_loss = sum(test_loss)/len(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def draw_loss_figure(train_loss, valid_loss, model_name):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    min_pos = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(min_pos, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    figure_name = model_name + 'loss_plot.png'
    fig.savefig(figure_name, bbox_inches='tight')


def show_data_img(labels_map, data, figure_size=(32, 32), cols=3, rows=3, random=False, is_gray=False):
    figure = plt.figure(figsize=figure_size)
    for i in range(1, cols * rows + 1):
        if random:
            sample_idx = torch.randint(len(data), size=(1,)).item()
        else:
            sample_idx = i - 1
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")

        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)

        if is_gray:
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(img)

    plt.show()


# # only for sgd
# def get_optimal_learning_rate(model, device, train_loader, valid_loader, criterion, filename='model/temp.pt',
#                               initial_lr=0.0009765625, epoch=5, n=10, score_type='loss'):
#     lr = initial_lr
#     global_valid_loss_min = float('inf')
#     lr_optimal = None
#     i = 1
#
#     while i <= n:
#         optimizer = optim.SGD(model.parameters(), lr=lr)
#         print(f"now lr is: {lr}, we try train {epoch} epochs")
#         _, valid_loss, _ = train_model(model, device, train_loader, valid_loader, criterion, optimizer, epoch,
#                                        patience=epoch+1, filename=filename + '_' + str(lr), verbose=False)
#         local_valid_loss_min = min(valid_loss)
#         if local_valid_loss_min < global_valid_loss_min:
#             lr_optimal = lr
#             global_valid_loss_min = local_valid_loss_min
#         print(f"lr: {lr}, valid loss {local_valid_loss_min}")
#         lr *= 2
#         i += 1
#         net_reset_parameters(model)
#
#     return lr_optimal


def net_reset_parameters(model):
    # for layer in model.modules():
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def create_layer_weight_bit_width_list(model):
    # for layer in model.modules():
    bit_width_list = []
    for layer in model.modules():
        if hasattr(layer, 'weightBits'):
            bit_width_list.append(layer.weightBits)

    return bit_width_list


def load_float_weight_for_fixed_point(model_load_filename, model):
    param_dict = torch.load(model_load_filename)
    param_list = []
    for name, param in param_dict.items():
        param_list.append(param)

    for layer in model.modules():
        if hasattr(layer, 'weightBits'):
            weight = param_list.pop(0)
            bias = None
            if layer.hasBias:
                bias = param_list.pop(0)

            layer.reset_parameters_from_float_parameters(weight, bias)


def differ_two_model(model1, model2):
    layer_update_times = []
    for layer1, layer2 in zip(model1.modules(), model2.modules()):
        if hasattr(layer1, 'weightBits') and hasattr(layer1, 'weightBits'):
            weight_bits = layer1.weightBits
            bit_update_times = []
            weight_before = fpA.to_int(layer1.fp_weight)
            weight_after = fpA.to_int(layer2.fp_weight)
            for i in range(weight_bits):
                bit_slice_s = weight_before.bitwise_right_shift(i).bitwise_and(1)
                bit_slice_t = weight_after.bitwise_right_shift(i).bitwise_and(1)
                diff_bit_slice = bit_slice_s.not_equal(bit_slice_t)
                bit_update_times.append(diff_bit_slice.sum().item())

            layer_update_times.append(bit_update_times)

    return layer_update_times


def draw_data_graph(x_list, y_lists, x_label, y_label, title):

    for y_list, para in y_lists:
        plt.plot(x_list, y_list, **para)

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
