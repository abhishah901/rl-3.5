from agent import Agent
from dqn_model import DQN


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from itertools import count
import os
import math
import random
import numpy as np
from collections import deque

"""
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
"""

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.env = env
        self.args = args
        self.episode=0
        self.n_actions = self.env.action_space.n
        self.epsilon_start = 1.0
        self.epsilon_final = 0.025
        self.epsilon_decay = 3000
        self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)
        self.epsilon = 0
        self.eval_net = DQN().cuda()
        self.target_net = DQN().cuda()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.criterion = nn.MSELoss()
        #self._model = Net(self.env.observation_space.shape, self.env.action_space.n)
        self._use_cuda = torch.cuda.is_available()
        self.optim = torch.optim.Adam(self.eval_net.parameters(), lr=self.args.learning_rate)
        
        if self._use_cuda:
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
            self.criterion = self.criterion.cuda()
        
#       self.replaybuffer = ReplayBuffer(args.buffer_size)
        self.buffer=deque(maxlen=10000)
        if args.test_dqn:
            #you can load your model here   
            print('loading trained model')
            self.eval_net.load_state_dict(torch.load(args.model_dqn))
            self.target_net.load_state_dict(self.eval_net.state_dict()) 
            if self._use_cuda:
                self.eval_net = self.eval_net.cuda()
                self.target_net = self.target_net.cuda()

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def replay_buffer(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################


        print('begin train...')

#        if self.args.log_file is not None:
#        fp_log = open(self.args.log_file, 'w', buffering=1)
        fout=open('dqn_score.log','w')
        if os.path.exists('model') == False:
            os.makedirs('model')

        losses = []
        all_rewards = []
        episode_reward = 0
        best_mean_reward = 0
        state = self.env.reset()
        for i_step in range(self.args.max_steps):
            self.epsilon = self.epsilon_by_frame(i_step)
            action = self.make_action(state)
            next_state, reward, done, _ = self.env.step(action)

            self.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                self.episode+=1
                print('{},{}'.format(self.episode, episode_reward))
                fout.write('Episode{},episode_reward{}\n'.format(self.episode, episode_reward))
                episode_reward = 0

            if len(self.buffer) == self.args.buffer_size:
                if i_step % self.args.eval_net_update_step == 0:
                    loss = self.optimize_model()
                    losses.append(loss)

                if i_step % self.args.target_net_update_step == 0:
                    self.target_net.load_state_dict(self.eval_net.state_dict())

            if i_step % self.args.save_freq == 0:
                mean_reward = \
                    sum(all_rewards[-100:]) / 100
                if best_mean_reward < mean_reward:
                    print('save best model with mean reward = %f'
                          % mean_reward)
                    best_mean_reward = mean_reward
                    torch.save(self.eval_net.state_dict(), self.args.model_dqn)

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation=torch.cuda.FloatTensor(observation.reshape((1,84,84,4))).transpose(1,3).transpose(2,3)
#        print(type(observation))
        Q_value = self.eval_net.forward(observation).data.cpu().numpy()
        if random.random() > self.epsilon:
           action  = np.argmax(Q_value)
        else:
            action = self.env.get_random_action()
        return action

    def optimize_model(self):

        state, action, reward, next_state, done = self.replay_buffer(self.args.batch_size)

        state = torch.FloatTensor(np.float32(state)).permute(0,3,1,2)
        next_state = torch.FloatTensor(np.float32(next_state)).permute(0,3,1,2)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.ByteTensor(done)

        if self._use_cuda:
            state = state.cuda()
            next_state = next_state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            done = done.cuda()

        q_values = self.eval_net(state)

        # next_q_values = self.target_net(next_state).detach()

        q_value  = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_net(next_state).detach()
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward + self.args.gamma * next_q_value * (1 - done)

        loss = self.criterion(q_value, expected_q_value.data)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
        return loss
