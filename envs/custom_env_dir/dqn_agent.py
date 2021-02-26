from envs.custom_env_dir.custom_dqn import DeepQNetwork
from envs.custom_env_dir.replay_memory import ReplayBuffer
import numpy as np
import torch as T
from datetime import datetime
import os

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, optimizer, fc1_dims, fc2_dims,
                 mem_size, batch_size, chkpt_dir, replace, eps_min, eps_dec,
                 algo, env_name):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        cwd = os.getcwd()
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0  
        now = datetime.now().strftime('%Y%m%d_%H%M')

        # Replay memory
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
        # Q-network / online network
        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=optimizer+'_gamma'+str(gamma)+'_lr'+\
                                       (('%.15f' % lr).rstrip('0').rstrip('.'))+\
                                       '_replace'+str(replace)+'_HL['+str(fc1_dims)+\
                                       ' ,'+str(fc2_dims)+']_q_eval.pt',chkpt_dir=self.chkpt_dir,\
                                       fc1_dims=fc1_dims, fc2_dims=fc2_dims, seed=1, optimizer=optimizer)
        
        # Target network
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=optimizer+'_gamma'+str(gamma)+'_lr'+\
                                       (('%.15f' % lr).rstrip('0').rstrip('.'))+\
                                       '_replace'+str(replace)+'_HL['+str(fc1_dims)+\
                                       ' ,'+str(fc2_dims)+']_q_next.pt',chkpt_dir=self.chkpt_dir,\
                                       fc1_dims=fc1_dims, fc2_dims=fc2_dims, seed=1, optimizer=optimizer)

    # Function for action selection
    def choose_action(self, observation):
        # Apply e-greedy action selection
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    # Store state transitions in replay memory
    def store_transition(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)
    
    # Function samples mini-batch of transitions from replay memory
    def sample_memory(self):
        state, action, reward, new_state = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_

    # Function updates weights of the target network every C steps
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    # Function decrements epsilon during training process
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    # Function stores current model parameters
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    # Function stores final agent at the end of each training run
    def save_models_final(self):
        self.q_eval.save_checkpoint_final()
        self.q_next.save_checkpoint_final()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
        
    def load_models_final(self):
        self.q_eval.load_checkpoint_final()
        self.q_next.load_checkpoint_final()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # sets gradient of optimizer object to zero
        self.q_eval.optimizer.zero_grad() 

        self.replace_target_network()

        states, actions, rewards, states_ = self.sample_memory()
        indices = np.arange(self.batch_size)
        
        # q_eval ist the main DQN (online network)
        q_pred = self.q_eval.forward(states)[indices, actions]
        
        # q_next is the target network
        q_next = self.q_next.forward(states_).max(dim=1)[0]
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()