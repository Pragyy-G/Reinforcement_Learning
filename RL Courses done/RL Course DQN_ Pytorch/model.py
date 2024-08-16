import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self,ALPHA):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1,32,8,stride=4, padding=1)
        self.conv2 = nn.Conv2d(32,64,4,stride=2)
        self.conv3 = nn.Conv2d(64,128,3)
        self.fc1 = nn.Linear(128*19*8, 512)
        self.fc2 = nn.Linear(512, 6)

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
        self.to(self.device)

    def forward(self,observation):
        observation = T.Tensor(observation).to(self.device)
        observation = observation.view(-1,1,185,95)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1,128*19*8)
        observation = F.relu(self.fc1(observation))

        actions = self.fc2(observation)

        return actions
        
class Agent(object):
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, epsEnd=0.05, replace=10000,
                 actionSpace=[0,1,2,3,4,5]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = [] #Associative cost of storing in numpy array is very huge and computationally expensive
        self.memCntr = 0
        self.replace_target_cnt = replace
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)

    def storeTransition(self,state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr +=1

    def chooseAction(self,observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        

