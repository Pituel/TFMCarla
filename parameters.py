"""

    All the much needed hyper-parameters needed for the algorithm implementation. 

"""

MODEL_LOAD = False
SEED = 0
BATCH_SIZE = 1
IM_WIDTH = 160
IM_HEIGHT = 80
GAMMA = 0.99
MEMORY_SIZE = 10000
EPISODES = 10
BURN_IN = 1000

#VAE Bottleneck
LATENT_DIM = 95

DQN_LEARNING_RATE = 0.001
EPSILON = 1.00
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995

REPLACE_NETWORK = 5
#UPDATE_FREQ = 5
DQN_CHECKPOINT_DIR = 'preTrained_models/ddqn/town02'
MODEL = 'carla_dueling_dqn.pth'
