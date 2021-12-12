# NOT MY CODE https://github.com/taylormcnally/keras-rl2/blob/master/examples/dqn_cartpole.py | I followed their steps and I'm using this to take notes

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make('CartPole-v0')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
              target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# for initial run, comment out dqn.load_weights and use dqn.fit and dqn.save_weights to get the weights saved into your env
# and if you want to run it again with the weights you saved, comment out dqn.fit and dqn.saveweights and run dqn.load_weights

# dqn.load_weights('dqn_CartPole-v0_weights.h5f')

# dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
# dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

dqn.test(env, nb_episodes=5, visualize=True)

