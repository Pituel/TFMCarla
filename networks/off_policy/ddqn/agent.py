import torch
import numpy as np
from networks.off_policy.ddqn.dueling_dqn import DuelingDQnetwork
from networks.off_policy.replay_buffer import ReplayBuffer
from parameters import *
from copy import deepcopy
import time
import pickle
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker



class DQNAgent(object):

    def __init__(self, n_actions, env, encode, town, args):
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.action_space = np.arange(n_actions)
        self.mem_size = MEMORY_SIZE
        self.batch_size = BATCH_SIZE
        self.train_step = 0
        self.encode = encode
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE, BURN_IN)
        self.q_network = DuelingDQnetwork(n_actions, MODEL)
        self.target_network = deepcopy(self.q_network)
        self.env = env
        self.initialize()
        self.current_ep_reward = 0
        self.town = town
        self.args = args

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def save_model(self):
        self.q_network.save_checkpoint()

    def load_model(self):
        self.q_network.load_checkpoint()

    def initialize(self):
        self.cumulative_score = 0
        self.scores = []
        self.mean_scores = []
        self.training_loss = []
        self.epsilon_ev = []
        self.update_loss = []
        self.observation = self.env.reset()
        self.observation = self.encode.process(self.observation)
        self.step_count = 0


    
    def take_step(self, mode='train'):
        if mode == 'explore':
            action = np.random.choice(self.action_space)
        else:
            action = self.q_network.get_action(self.observation, self.epsilon)
        
        new_observation, reward, done, info = self.env.step(action)
        new_observation = self.encode.process(new_observation)
   
        self.current_ep_reward += reward
        # Save experience in buffer
        self.replay_buffer.save_transition(self.observation, action, reward, new_observation, done)
        self.observation = new_observation
        
        if done:
            self.observation = self.env.reset()
            self.observation = self.encode.process(self.observation)

        return done
    
    def calculate_loss(self, batch):
        states, actions, rewards, next_states, dones= [i for i in batch]

        rewards_vals = torch.FloatTensor(rewards).to(device=self.device).reshape(-1,1)
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1,1).to(
            device=self.device)
        dones_t = torch.BoolTensor(dones).to(device=self.device)
        #dones_t = torch.ByteTensor(dones).to(device=self.device)

        # Get Q-values from main network
        qvals = torch.gather(self.q_network.get_qvals(states), 1, actions_vals)

        #DQN update
        next_actions = torch.max(self.q_network.get_qvals(next_states), dim=-1)[1]
        if self.device == 'cuda':
            next_actions_vals = next_actions.reshape(-1,1).to(device=self.device)
        else:
            next_actions = torch.LongTensor(next_actions).reshape(-1,1).to(device=self.device)
            
        # Get Q-values from target network
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_vals).detach()

        qvals_next[dones_t] = 0 # 0 en estados terminales

        # Bellman equation
        expected_qvals = self.gamma * qvals_next + rewards_vals
        
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))
        
        return loss



    def update(self):
        # Set the gradients to zero
        self.q_network.optimizer.zero_grad()
        # Select a batch from buffer
        batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        # Calculate loss
        loss = self.calculate_loss(batch) 
        # Compute the gradients. 
        loss.backward()
        # Apply gradients to main network.
        self.q_network.optimizer.step()
        # Save loss
        if self.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())


    def train(self):
        start = time.time()
        tracker = EmissionsTracker()
        tracker.start()
        # Fill the buffer with random experience.
        print("Filling replay buffer...")
        while self.replay_buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')

        episode = 0

        # self.load_model()
        # with open(f"checkpoints/DDQN/{self.town}/checkpoint_ddqn.pickle", 'rb') as f:
        #     data = pickle.load(f)
        #     episode = data['epoch']
        #     self.cumulative_score = data['cumulative_score']
        #     self.epsilon = data['epsilon']
        
        training = True
        print("Training...")
        while training:
            self.current_ep_reward = 0
            print('Starting Episode: ', episode, ', Epsilon Now:  {:.3f}'.format(self.epsilon), ', ', end="")

            epdone = False
            while epdone == False:
                epdone = self.take_step(mode='train')
                # Update the main network
                if self.step_count % UPDATE_FREQ == 0:
                    self.update()

                # Sincornize main and target networks
                if self.step_count % REPLACE_NETWORK == 0:
                    self.target_network.load_state_dict(
                        self.q_network.state_dict())

                if epdone:
                    episode += 1

                    self.scores.append(self.current_ep_reward)
                    self.cumulative_score = np.mean(self.scores[-50:])
                    self.mean_scores.append(self.cumulative_score)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.epsilon_ev.append(self.epsilon)
                    
                    self.update_loss = []

                    print('Reward:  {:.2f}'.format(self.current_ep_reward), ', Average Reward:  {:.2f}'.format(self.cumulative_score))#, ', Training Loss:  {:.2f}'.format(self.update_loss))

                    if episode >= 10 and episode % 10 == 0:
                        self.save_model()                      

                        data_obj = {'cumulative_score': self.cumulative_score, 'epsilon': self.epsilon,'epoch': episode}
                        with open(f"checkpoints/DDQN/{self.town}/checkpoint_ddqn.pickle", 'wb') as handle:
                            pickle.dump(data_obj, handle)

                    if episode >= 10 and episode % 50 == 0:
                        
                        plt.close('all')
                        self.plot_rewards(self.scores, self.mean_scores)
                        self.plot_loss(self.training_loss)
                        self.plot_epsilon(self.epsilon_ev)
                    
                    if episode >= EPISODES or self.cumulative_score > 800:
                        end = time.time()
                        print("Tiempo de entrenamiento: {} minutos".format(round((end-start)/60,2)))
                        emissions: float = tracker.stop()
                        print(emissions)
                        training = False
                        
                        plt.close('all')
                        self.plot_rewards(self.scores, self.mean_scores)
                        self.plot_loss(self.training_loss)
                        self.plot_epsilon(self.epsilon_ev)
                            
                        print('\nEpisode limit reached.')
                        break

                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def test(self):
        self.load_model()
        total_reward = 0
        while True:
            action = self.q_network.get_action(self.observation, epsilon=0.0)
            new_observation, reward, done, info = self.env.step(action)
            new_observation = self.encode.process(new_observation)

            self.observation = new_observation

            total_reward += reward

            if done:
                print('Reward: {:.2f}'.format(total_reward))
                break
        

    def plot_rewards(self, tr_rewards, mean_tr_rewards):
        plt.figure(figsize=(8,4))
        plt.plot(tr_rewards, label='Rewards')
        plt.plot(mean_tr_rewards, label='Mean Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc="upper left")
        plt.savefig('plots/Rewards.png')

    def plot_loss(self, tr_loss):
        plt.figure(figsize=(8,4))
        plt.plot(tr_loss, label='Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.legend(loc="upper left")
        plt.savefig('plots/Loss.png')

    def plot_epsilon(self, eps_evolution):
        plt.figure(figsize=(8,4))
        plt.plot(eps_evolution, label='Epsilon')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.legend(loc="upper right")
        plt.savefig('plots/eps.png')


                   