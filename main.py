import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Manager, Process, set_start_method
from tictactoe_gpu import TicTacToe
from alphazero import AZAgent, selfplay, arena_training
from replay_buffer import ExperienceReplay
from net import Net
import datetime

if __name__ == '__main__':
    width = 6
    height = 6
    win_count = 4
    repaly_buffer_size = 1000000
    batch_size = 15000
    actions = width * height
    iteration_count = 1000
    weight_decay = 0.0001
    lr = 0.01
    lrs = [lr, 0.001, 0.001, 0.0001, 0.0001]
    dirichlet_alphas = [0.3, 0.3, 0.15, 0.15, 0.03]

    cross_entropy = nn.CrossEntropyLoss()
    cross_entropy_moves = nn.CrossEntropyLoss()

    batches_in_iteration = 400
    cpus = 6
    name = 'azt_a6_6_7'

    torch.cuda.set_device(1)

    set_start_method('spawn')

    env = TicTacToe(width = width, height = height, win_count = win_count)

    current_model = Net(2, 7, 64, actions)

    #net.share_memory()
    optimizer = optim.Adam(current_model.parameters(), lr=lrs[0], weight_decay = weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    current_model.to(device)
    best_model = copy.deepcopy(current_model)
    best_model.to(device)

    agent = AZAgent(env, device = device, games_in_iteration = 100, simulation_count = 200, name = name)
    replay_buffer = ExperienceReplay(repaly_buffer_size, height, width, batch_size)

    model_index = 0
    for iteration in range(iteration_count):
        start = datetime.datetime.now()
        if iteration % 100 == 0:
            i = iteration // 100
            if i < (len(lrs)):
                for g in optimizer.param_groups:
                    g['lr'] = lrs[i]
                agent.dirichlet_alpha = dirichlet_alphas[i]

        print('selfplay', iteration)

        with Manager() as manager:
            output_list = manager.list()
            processes = []
            for i in range(cpus):
                p = Process(target=selfplay, args=(agent, best_model, output_list, i % 2 == 1))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            for i in range(len(output_list)):
                length, states, policies, values, moves_left = output_list[i]
                replay_buffer.add(length, states, policies, values, moves_left)

        print('index', replay_buffer.index)

        avg_loss_policy, avg_loss_value, avg_loss_moves = 0, 0, 0
        for b in range(batches_in_iteration):
            optimizer.zero_grad()
            states, target_policies, target_values, target_moves = replay_buffer.sample()
            model_policies, model_values, model_left_moves = current_model(states.to(device).float())
            loss_policy = - (target_policies.to(device) * F.log_softmax(model_policies, 1)).mean()
            loss_value = cross_entropy(model_values, target_values.view(-1).to(device))
            loss_left_moves = cross_entropy_moves(model_left_moves, target_moves.view(-1).to(device) )
            loss = loss_value + loss_policy + loss_left_moves
            avg_loss_policy += loss_policy.detach().item()
            avg_loss_value += loss_value.detach().item()
            avg_loss_moves += loss_left_moves.detach().item()
            loss.backward()
            optimizer.step()

        print('policy loss: ', (avg_loss_policy / batches_in_iteration), 'value loss: ', (avg_loss_value / batches_in_iteration), 'left moves loss: ', (avg_loss_moves / batches_in_iteration))
        print('update', (datetime.datetime.now() - start))

        torch.cuda.empty_cache()

        print('arena')
        win, loss, draw = 0, 0, 0
        with Manager() as manager:
            output_list = manager.list()
            processes = []
            for i in range(cpus // 2):
                p = Process(target=arena_training, args=(agent, current_model, best_model, output_list, 10, i % 2 == 0))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            for i in range(len(output_list)):
                w, l, d = output_list[i]
                win += w
                loss += l
                draw += d

        a_sum = win + loss
        if a_sum > 0 and win / float(a_sum) > 0.55:
            torch.save(current_model.state_dict(), 'models/' + name + '_' + str(model_index) + '.pt')
            best_model = copy.deepcopy(current_model)
            best_model.to(device)

            print('model is updated ', model_index)
            model_index += 1

        print('win: ', win, 'loss: ', loss, 'draw: ', draw)

        torch.cuda.empty_cache()
        print(datetime.datetime.now() - start)
