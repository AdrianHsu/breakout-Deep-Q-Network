from agent_dir.agent import Agent
from colors import *
from tqdm import *
from collections import deque
import tensorflow as tf
import numpy as np
import os
import random

SEED = 11037
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.intra_op_parallelism_threads = 44 # cpu
# config.inter_op_parallelism_threads = 44 # cpu
print(config)
stages = ["[OBSERVE]", "[EXPLORE]", "[TRAIN]"]

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
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.step = 0
    self.stage = ""

    if args.test_dqn:
      #you can load your model here
      print('loading trained model')

    self.s = tf.placeholder(tf.float32, [None, 84, 84, 4], 
      name='s')
    self.s_ = tf.placeholder(tf.float32, [None, 84, 84, 4], 
      name='s_')
    self.y_input = tf.placeholder(tf.float32, [None]) 
    self.action_input = tf.placeholder(tf.float32, [None, self.n_actions])

    self.q_eval = self.build_net(self.s, 'eval_net') # online Q
    self.q_target = self.build_net(self.s_, 'target_net') # target Q
    
    self.train_summary = []
    self.buildOptimizer()
    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] 

    self.epsilon = self.args.epsilon_start
    self.replay_memory = deque()

    self.ckpts_path = self.args.save_dir + "dqn.ckpt"
    self.saver = tf.train.Saver(max_to_keep = 3)
    self.sess = tf.Session(config=config)
    
    self.summary_writer = tf.summary.FileWriter(self.args.log_dir, graph=self.sess.graph)

    self.init()

  def init(self):
    ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
    print(ckpt)
    if self.args.load_saver and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        #model_eval.saver.restore(eval_sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        self.step = self.sess.run(self.global_step)
        print('load step: ', self.step)
    else:
        print('Created new model parameters..')
        self.sess.run(tf.global_variables_initializer())

  def init_game_setting(self):
    """
    Testing function will call this function at the begining of new game
    Put anything you want to initialize if necessary
    """
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
    w_initializer=tf.truncated_normal_initializer(0, 1e-2)):

    return tf.get_variable(
      name=name,
      shape=shape, 
      initializer=w_initializer)

  def init_b(self, shape, name='biases', 
    b_initializer = tf.constant_initializer(1e-2)):

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
      # (32, 4) -> (32, 1)
      self.train_summary.append(tf.summary.scalar('avg_q', 
        tf.reduce_mean(self.q_eval)))

      self.q_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)
      self.train_summary.append(tf.summary.scalar('avg_action_q', 
        tf.reduce_mean(self.q_action)))  

      self.loss = tf.reduce_mean(tf.square(self.y_input - self.q_action))
      self.train_summary.append(tf.summary.scalar('loss', self.loss))

      self.train_summary = tf.summary.merge(self.train_summary)
    with tf.variable_scope('train'):
      self.logits = tf.train.RMSPropOptimizer(self.lr, decay=0.99, momentum=0.9, epsilon=1e-6).minimize(self.loss)

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
    if self.step == 0:
      self.stage = stages[0]
    elif self.step == self.args.observe_steps:
      self.stage = stages[1]
    elif self.step == self.args.observe_steps + self.args.explore_steps:
      self.stage = stages[2]

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

    y_batch = []

    for i in range(self.batch_size):
      done = done_batch[i]
      if done:
        y_batch.append(reward_batch[i])
      else:
        y = reward_batch[i] + self.gamma * np.max(q_eval_batch[i])
        y_batch.append(y)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    _, summary, loss = self.sess.run([self.logits, self.train_summary, self.loss], feed_dict={
      self.s: state_batch,
      self.y_input: y_batch,
      self.action_input: action_batch
      }, options=run_options)

    if self.step % self.args.summary_time == 0:
      self.summary_writer.add_summary(summary, global_step=self.step)

    if self.step % self.args.update_target == 0 and self.step > self.args.observe_steps:
      self.sess.run(self.replace_target_op)
      print(color("\n[target params replaced]", fg='white', bg='green', style='bold'))

    return loss

  def train(self):

    """
    Implement your training algorithm here
    """
    pbar = tqdm(range(self.args.num_episodes))
    current_loss = 0
    train_rewards = []
    train_episode_len = 0.0
    file_loss = open("loss.csv", "a")
    file_loss.write("episode,step,reward,loss,length\n")
    for episode in pbar:
      # print('episode: ', episode)
      # "state" is also known as "observation"
      obs = self.env.reset() #(84, 84, 4)
      self.init_game_setting()
      train_loss = 0
      
      episode_reward = 0.0
      for s in range(self.args.max_num_steps):
        # self.env.env.render()
        action = self.make_action(obs, test=False) # Performing the same action for 4 frames?
        obs_, reward, done, info = self.env.step(action)
        episode_reward += reward
        self.storeTransition(obs, action, reward, obs_, done)
        self.step = self.sess.run(self.add_global)
        
        if len(self.replay_memory) > self.args.replay_memory_size:
          self.replay_memory.popleft()
        # once the storage stored > batch_size, start training
        if len(self.replay_memory) > self.batch_size:
          if self.step % self.args.update_eval == 0:
            loss = self.learn()
            train_loss += loss

        if self.step % self.args.saver_steps == 0 and episode != 0:
          ckpt_path = self.saver.save(self.sess, self.ckpts_path, global_step = self.step)
          print(color("\nStep: " + str(self.step) + ", Saver saved: " + ckpt_path, fg='white', bg='blue', style='bold'))

        obs = obs_
        if done:
          break
      train_rewards.append(episode_reward)
      train_episode_len += s

      if episode % self.args.num_eval == 0 and episode != 0:
        current_loss = train_loss
        avg_reward_train = np.mean(train_rewards)
        train_rewards = []
        avg_episode_len_train = train_episode_len / float(self.args.num_eval)
        train_episode_len = 0.0
        
        file_loss.write(str(episode) + "," + str(self.step) + "," + "{:.2f}".format(avg_reward_train) + "," + "{:.4f}".format(current_loss) + "," + "{:.2f}".format(avg_episode_len_train) + "\n")
        file_loss.flush()
        
        print(color("\n[Train] Avg Reward: " + "{:.2f}".format(avg_reward_train) + ", Avg Episode Length: " + "{:.2f}".format(avg_episode_len_train), fg='red', bg='white'))

      pbar.set_description(self.stage + " G: " + "{:.2f}".format(self.gamma) + ', E: ' + "{:.2f}".format(self.epsilon) + ", L: " + "{:.4f}".format(current_loss) + ", D: " + str(len(self.replay_memory)) + ", S: " + str(self.step))

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
    state = observation.reshape((1, 84, 84, 4))
    q_value = self.sess.run(self.q_eval, feed_dict={self.s: state})[0]
    if random.random() <= self.epsilon:
      action = random.randrange(self.n_actions)
    else:
      action = np.argmax(q_value)

    if self.epsilon > self.args.epsilon_end \
        and self.step > self.args.observe_steps:
      old_e = self.epsilon
      interval = self.args.epsilon_start - self.args.epsilon_end
      self.epsilon -= interval / float(self.args.explore_steps)
      # print('epsilon: ', old_e, ' -> ', self.epsilon)

    return action

    # return self.env.get_random_action()

