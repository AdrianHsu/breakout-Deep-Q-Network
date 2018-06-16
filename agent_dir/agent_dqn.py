from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import os

class Agent_DQN(Agent):
  def __init__(self, env, args):
    """
    Initialize every things you need here.
    For example: building your model
    """

    super(Agent_DQN,self).__init__(env)
    self.args = args
    self.n_actions = env.action_space.n # = 4
    self.step = 0

    if args.test_dqn:
      #you can load your model here
      print('loading trained model')

    self.s = tf.placeholder(tf.float32, [None, 84, 84, 4], 
      name='s')
    self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

    self.build_net('eval_net')
    # self.build_net('target_net')

  def init_game_setting(self):
    """
    Testing function will call this function at the begining of new game
    Put anything you want to initialize if necessary
    """
    ##################
    # YOUR CODE HERE #
    ##################
    pass

  def build_net(self, var_scope):

    with tf.variable_scope(var_scope):
      cnames = [var_scope + '_params', tf.GraphKeys.GLOBAL_VARIABLES]
      
      with tf.variable_scope('conv1'):
        W1 = self.init_W(shape=[8, 8, 4, 32], 
              collections=cnames)
        b1 = self.init_b(shape=[32],
              collections=cnames)
        conv1 = self.conv2d(self.s, W1, strides=4)
        h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))

      # with tf.name_scope('max_pool1'):
      #   h_pool1 = self.max_pool(h_conv1)

      with tf.variable_scope('conv2'):
        W2 = self.init_W(shape=[4, 4, 32, 64],
              collections=cnames)
        b2 = self.init_b(shape=[64],
              collections=cnames)
        conv2 = self.conv2d(h_conv1, W2, strides=2)
        h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))

      with tf.variable_scope('conv3'):
        W3 = self.init_W(shape=[3, 3, 64, 64],
              collections=cnames)
        b3 = self.init_b(shape=[64],
              collections=cnames)
        conv3 = self.conv2d(h_conv2, W3, strides=1)
        h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b3))

      h_flatten = tf.reshape(h_conv3, [-1, 3136])
      
      with tf.variable_scope('fc1'):
        W_fc1 = self.init_W(shape=[3136, 512],
                  collections=cnames)
        b_fc1 = self.init_b(shape=[512],
                  collections=cnames)
        fc1 = tf.nn.bias_add(tf.matmul(h_flatten, W_fc1), b_fc1)
        h_fc1 = tf.nn.relu(fc1)

      with tf.variable_scope('fc2'):
        W_fc2 = self.init_W(shape=[512, self.n_actions],
                  collections=cnames)
        b_fc2 = self.init_b(shape=[self.n_actions],
                  collections=cnames)
        fc2 = tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2)

      exit(0)



  def init_W(self, shape, name='weights', collections=None,
    w_initializer=tf.truncated_normal_initializer(0., 1e-1)):
    
    return tf.get_variable(
            name=name,
            shape=shape, 
            initializer=w_initializer,
            collections=collections)

  def init_b(self, shape, name='biases', collections=None,
    b_initializer = tf.constant_initializer(1e-1)):

    return tf.get_variable(
            name=name,
            shape=shape,
            initializer=b_initializer,
            collections=collections)

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

  def storeTransition(self, obs, action, reward, obs_):
    """
    Store transition in this step
    Input:
        obs: np.array
            stack 4 last preprocessed frames, shape: (84, 84, 4)
        obs_: np.array
            stack 4 last preprocessed frames, shape: (84, 84, 4)
        action: int (0, 1, 2, 3)
            the predicted action from trained model
        reward: float (0, +1, -1)
            the reward from selected action
    Return:
        None
    """

  def learn(self):
    pass

  def train(self):
    """
    Implement your training algorithm here
    """
    for episode in range(self.args.num_episodes):

      obs = self.env.reset() #(84, 84, 4)
      print(episode)
      while True:
        # env.render() # our code doesn't support this

        action = self.make_action(obs)
        obs_, reward, done, info = self.env.step(action)
        # print(info)
        self.storeTransition(obs, action, reward, obs_)

        if (self.step > 200) and (self.step % 5 == 0):
          self.learn()
        
        obs = obs_

        if done:
          break
        
        self.step += 1

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
    ##################
    # YOUR CODE HERE #
    ##################
    return self.env.get_random_action()