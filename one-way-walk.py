import numpy as np
import time
import pandas as pd

np.random.seed(2)

N_STATES =   8
EPSILON  = 0.9
ALPHA    = 0.1
LAMBDA   = 0.9
ACTIONS  = ['left', 'right']

MAX_EPISODES =  20
FRESH_TIME   = 0.3

def build_q_table(n_states, actions):
    table = pd.DataFrame(
                np.zeros((n_states, len(actions))),
                columns=actions,
    )
    #print(table)
    return table



def choose_action(state, q_table):
    # state: 1, 2, 3, 4...
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or \
       (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_next = 'terminal'
            R = 1
        else:
            S_next = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_next = S
        else:
            S_next = S - 1
    return S_next, R

# environment
def update_env(S, episode, c):
    env_list = ['-']*(N_STATES - 1) + ['T']

    if S != 'terminal':
        env_list[S] = 'o'
        tmp = ''.join(env_list)
        print('\r{}'.format(tmp), end='')
        time.sleep(FRESH_TIME)
    else:
        tmp = '[FINISHED] Episode %s: total_steps = %s\n' % (episode + 1, c)
        print('\r{}'.format(tmp), end='')
        time.sleep(FRESH_TIME)


def QLearning():

    # all zeros in the beginning
    # i.e. table.all() == 0
    q_table = build_q_table(N_STATES, ACTIONS)
     
    for episode in range(MAX_EPISODES):
        c = 0
        S = 0
        end = False
        print(q_table)
        update_env(S, episode, c)
        while not end:
            A = choose_action(S, q_table)
            q_pred = q_table.ix[S, A]
            S_next, R = get_env_feedback(S, A)
            
            if S_next != 'terminal':
                q_target = R + LAMBDA * max(q_table.iloc[S_next, :])
            else:
                q_target = R
                end = True
            
            q_table.ix[S, A] = q_table.ix[S, A] + ALPHA * (q_target - q_pred)
            S = S_next
            
            update_env(S, episode, c + 1)
            c += 1
    
    return q_table

if __name__ == '__main__':
    q_table = QLearning()
    print('\r\nQ-table:\n')
    print(q_table)
