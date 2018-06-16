from agent_dir.agent import Agent
from colors import *
from tqdm import *
from collections import deque

import tensorflow as tf
import numpy as np
import os
import random

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

class Agent_DQN(Agent):
  def __init__(self, env, args):
    """
    Initialize every things you need here.
    For example: building your model
    """

    super(Agent_DQN,self).__init__(env)
    self.args = args
    self.batch_size = args.batch_size
    self.lr = args.learning_rate
    self.gamma = args.gamma_reward_decay
    self.n_actions = env.action_space.n # = 4
    self.state_dim = env.observation_space.shape[0] # 84
    self.step = 0

    if args.test_dqn:
      #you can load your model here
      print('loading trained model')

    self.s = tf.placeholder(tf.float32, [None, 84, 84, 4], 
      name='s')
    self.s_ = tf.placeholder(tf.float32, [None, 84, 84, 4], 
      name='s_')
    self.y_input = tf.placeholder(tf.float32, [None]) 
    self.action_input = tf.placeholder(tf.float32, [None, self.n_actions])

    self.q_eval = self.build_net(self.s, 'eval_net')
    self.q_target = self.build_net(self.s_, 'target_net')
    self.buildOptimizer()
    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] 

    self.epsilon = self.args.epsilon_start
    self.replay_memory = deque()

    self.sess = tf.Session(config=gpu_config)

    self.summary_writer = tf.summary.FileWriter("logs/", graph=self.sess.graph)
    self.sess.run(tf.global_variables_initializer())

  def init_game_setting(self):
    """
    Testing function will call this function at the begining of new game
    Put anything you want to initialize if necessary
    """
    ##################
    # YOUR CODE HERE #
    ##################
    pass

  def build_net(self, s, var_scope):

    with tf.variable_scope(var_scope):      
      with tf.variable_scope('conv1'):
        W1 = self.init_W(shape=[8, 8, 4, 32])
        b1 = self.init_b(shape=[32])
        conv1 = self.conv2d(s, W1, strides=4)
        h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))

      # with tf.name_scope('max_pool1'):
      #   h_pool1 = self.max_pool(h_conv1)

      with tf.variable_scope('conv2'):
        W2 = self.init_W(shape=[4, 4, 32, 64])
        b2 = self.init_b(shape=[64])
        conv2 = self.conv2d(h_conv1, W2, strides=2)
        h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))

      with tf.variable_scope('conv3'):
        W3 = self.init_W(shape=[3, 3, 64, 64])
        b3 = self.init_b(shape=[64])
        conv3 = self.conv2d(h_conv2, W3, strides=1)
        h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b3))

      h_flatten = tf.reshape(h_conv3, [-1, 3136])
      
      with tf.variable_scope('fc1'):
        W_fc1 = self.init_W(shape=[3136, 512])
        b_fc1 = self.init_b(shape=[512])
        fc1 = tf.nn.bias_add(tf.matmul(h_flatten, W_fc1), b_fc1)
        h_fc1 = tf.nn.relu(fc1)

      with tf.variable_scope('fc2'):
        W_fc2 = self.init_W(shape=[512, 4])
        b_fc2 = self.init_b(shape=[4])
        fc2 = tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2, name='Q')
    
    return fc2

  def init_W(self, shape, name='weights', 
    w_initializer=tf.truncated_normal_initializer(0, 1e-1)):

    return tf.get_variable(
      name=name,
      shape=shape, 
      initializer=w_initializer)

  def init_b(self, shape, name='biases', 
    b_initializer = tf.constant_initializer(1e-1)):

    return tf.get_variable(
      name=name,
      shape=shape,
      initializer=b_initializer)

  def conv2d(self, x, kernel, strides=4):

    return tf.nn.conv2d(
      input=x, 
      filter=kernel, 
      strides=[1, strides, strides, 1], 
      padding="VALID")

  def max_pool(self, x, ksize=2, strides=2):
    return tf.nn.max_pool(x, 
      ksize=[1, ksize, ksize, 1], 
      strides=[1, strides, strides, 1], 
      padding="SAME")

  def buildOptimizer(self):
    with tf.variable_scope('loss'):
      q_actions = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)
      # self.loss = tf.reduce_mean(tf.squared_difference(self.y_input, q_actions, name='td_error'))
      self.loss = tf.reduce_mean(tf.square(self.y_input - q_actions))
      self.train_summary = tf.summary.scalar('loss', self.loss)
      tf.summary.merge_all()
    with tf.variable_scope('train'):
      self.logits = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

  def storeTransition(self, s, action, reward, s_, done):
    """
    Store transition in this step
    Input:
        s: np.array
            stack 4 last preprocessed frames, shape: (84, 84, 4)
        s_: np.array
            stack 4 last preprocessed frames, shape: (84, 84, 4)
        action: int (0, 1, 2, 3)
            the predicted action from trained model
        reward: float (0, +1, -1)
            the reward from selected action
    Return:
        None
    """
    one_hot_action = np.zeros(self.n_actions)
    one_hot_action[action] = 1
    self.replay_memory.append((s, one_hot_action, reward, s_, done))

  def learn(self):
    minibatch = random.sample(self.replay_memory, self.batch_size)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]
    done_batch = [data[4] for data in minibatch]
    # print(np.array(state_batch).shape) # (32, 84, 84, 4)

    q_eval_batch = self.sess.run(self.q_target, 
      feed_dict={self.s_: next_state_batch})

    q_eval_storage = []

    for i in range(self.batch_size):
      done = done_batch[i]
      if done:
        q_eval_storage.append(reward_batch[i])
      else:
        q = reward_batch[i] + self.gamma * np.max(q_eval_batch[i])
        q_eval_storage.append(q)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    _, summary, loss = self.sess.run([self.logits, self.train_summary, self.loss], feed_dict={
      self.s: state_batch,
      self.y_input: q_eval_storage,
      self.action_input: action_batch
      }, options=run_options)

    if self.step % self.args.update_time == 0:
      self.summary_writer.add_summary(summary, global_step=self.step)
      self.sess.run(self.replace_target_op)
      print('target_params_replaced...')

  def train(self):

    """
    Implement your training algorithm here
    """
    pbar = tqdm(range(self.args.num_episodes))
    total_eval_reward = 0
    avg_reward = 0
    for episode in pbar:
      # print('episode: ', episode)
      # "state" is also known as "observation"
      obs = self.env.reset() #(84, 84, 4)
      for s in range(self.args.max_num_steps):
        # self.env.env.render()
        action = self.make_action(obs, test=False)
        obs_, reward, done, info = self.env.step(action)
        self.storeTransition(obs, action, reward, obs_, done)
        self.step += 1
        if len(self.replay_memory) > self.args.replay_size:
          self.replay_memory.popleft()
        # once the storage stored > batch_size, start training
        if len(self.replay_memory) > self.batch_size:
          self.learn()
        
        obs = obs_
        if done:
          break

      if episode % self.args.num_eval == 0:
        total_eval_reward = 0
        for i in range(self.args.num_test_episodes):
          obs = self.env.reset()
          # self.env.env.render()
          for j in range(self.args.max_num_steps):
            action = self.make_action(obs, test=True)
            obs_, reward, done, info = self.env.step(action)
            total_eval_reward += reward
            if done:
              break
        avg_reward = total_eval_reward / float(self.args.num_test_episodes)
        print("Avg Reward(100 eps): " + "{:.4f}".format(total_eval_reward))
        if avg_reward > 40.0: # baseline
          print('baseline passed!')
          break
      pbar.set_description('Epsilon: ' + "{:.2f}".format(self.epsilon) + ", Gamma: " + "{:.2f}".format(self.gamma) + ", lr: " + "{:.6f}".format(self.lr))

    print('game over')
    # env.destroy()

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
    state = observation.reshape((1,84,84,4))
    q_value = self.sess.run(self.q_eval, feed_dict={self.s: state})[0]

    if test:
      return np.argmax(q_value)

    if random.random() <= self.epsilon and not test:
      action = random.randrange(self.n_actions)
    else:
      action = np.argmax(q_value)

    if self.epsilon > self.args.epsilon_end \
        and self.step > self.args.observe_steps:
      old_e = self.epsilon
      interval = self.args.epsilon_start - self.args.epsilon_end
      self.epsilon -= interval / float(self.args.anneal_rate)
      # print('epsilon: ', old_e, ' -> ', self.epsilon)

    return action

    # return self.env.get_random_action()


