import random
import numpy as np
import sys
import matplotlib.pyplot as plt


class Sender:
    """
    A Q-learning agent that sends messages to a Receiver

    """

    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        """
        Initializes this agent with a state, set of possible actions, and a means of storing Q-values

        :param num_sym: The number of arbitrary symbols available for sending
        :type num_sym: int
        :param grid_rows: The number of rows in the grid
        :type grid_rows: int
        :param grid_cols: The number of columns in the grid
        :type grid_cols: int
        :param alpha_i: The initial learning rate
        :type alpha: float
        :param alpha_f: The final learning rate
        :type alpha: float
        :param num_ep: The total number of episodes
        :type num_ep: int
        :param epsilon: The epsilon in epsilon-greedy exploration
        :type epsilon: float
        :param discount: The discount factor
        :type discount: float
        """
        self.actions = range(num_sym)
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.q_vals = {}

        # dictionary where states is a tuple ((x, y), a), where (x, y) is the prize location and a is the action taken
        for x in range(grid_cols):
            for y in range(grid_rows):
                for a in self.actions:
                    self.q_vals[((x, y), a)] = 0.0

    def select_action(self, state):
        """
        This function is called every time the agent must act. It produces the action that the agent will take
        based on its current state

        :param state: the state the agent is acting from, in the form (x,y), which are the coordinates of the prize
        :type state: (int, int)
        :return: The symbol to be transmitted (must be an int < N)
        :rtype: int
        """

        # epsilon = make random choice (explore), 1-epsilon: go by max Q value (exploit)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # extract q values from table for said state, and return action with highest q value
            q_values = [self.q_vals[(state, a)] for a in self.actions]
            max_q = max(q_values)
            return q_values.index(max_q)

    def update_q(self, old_state, action, reward):
        """
        This function is called after an action is resolved so that the agent can update its Q-values

        :param old_state: the state the agent was in when it acted, in the form (x,y), which are the coordinates
                          of the prize
        :type old_state: (int, int)
        :param action: the action that was taken
        :type action: int
        :param reward: the reward that was received
        :type reward: float
        """
        prev_q_val = self.q_vals[(old_state, action)]
        # sender always enters a random next state, so no point in considering expected future reward 
        # and no need to consider the new state of sender
        self.q_vals[(old_state, action)] = prev_q_val + self.alpha * (reward - prev_q_val)


class Receiver:
    """
    A Q-learning agent that receives a message from a Sender and then navigates a grid

    """

    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        """
        Initializes this agent with a state, set of possible actions, and a means of storing Q-values

        :param num_sym: The number of arbitrary symbols available for sending
        :type num_sym: int
        :param grid_rows: The number of rows in the grid
        :type grid_rows: int
        :param grid_cols: The number of columns in the grid
        :type grid_cols: int
        :param alpha_i: The initial learning rate
        :type alpha: float
        :param alpha_f: The final learning rate
        :type alpha: float
        :param num_ep: The total number of episodes
        :type num_ep: int
        :param epsilon: The epsilon in epsilon-greedy exploration
        :type epsilon: float
        :param discount: The discount factor
        :type discount: float
        """
        self.actions = [0,1,2,3] # Note: these correspond to [up, down, left, right]
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.q_vals = {}
        # table in the form of a dictionary where key is a tuple of state (m, x, y) and action (0-num_sym) and 
        # value is the Q value
        for m in range(num_sym):
            for x in range(grid_cols):
                for y in range(grid_rows):
                    for a in self.actions:
                        self.q_vals[(m, x, y), a] = 0.0

    def select_action(self, state):
        """
        This function is called every time the agent must act. It produces the action that the agent will take
        based on its current state
        :param state: the state the agent is acting from, in the form (m,x,y), where m is the message received
                      and (x,y) are the board coordinates
        :type state: (int, int, int)
        :return: The direction to move, where 0 is up, 1 is down, 2 is left, and 3 is right
        :rtype: int
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [self.q_vals[(state, a)] for a in self.actions]
            max_q = max(q_values)
            return q_values.index(max_q)

    def update_q(self, old_state, new_state, action, reward):
        """
        This function is called after an action is resolved so that the agent can update its Q-values

        :param old_state: the state the agent was in when it acted in the form (m,x,y), where m is the message received
                          and (x,y) are the board coordinates
        :type old_state: (int, int, int)
        :param new_state: the state the agent entered after it acted
        :type new_state: (int, int, int)
        :param action: the action that was taken
        :type action: int
        :param reward: the reward that was received
        :type reward: float
        """
        old_q_val = self.q_vals[(old_state, action)]
        max_q_future = max([self.q_vals[(new_state, a)] for a in self.actions])
        self.q_vals[(old_state, action)] = old_q_val + self.alpha * (reward + self.discount * max_q_future - old_q_val)


def get_grid(grid_name:str):
    """
    This function produces one of the three grids defined in the assignment as a nested list

    :param grid_name: the name of the grid. Should be one of 'fourroom', 'maze', or 'empty'
    :type grid_name: str
    :return: The corresponding grid, where True indicates a wall and False a space
    :rtype: list[list[bool]]
    """
    grid = [[False for i in range(5)] for j in range(5)] # default case is 'empty'
    if grid_name == 'fourroom':
        grid[0][2] = True
        grid[2][0] = True
        grid[2][1] = True
        grid[2][3] = True
        grid[2][4] = True
        grid[4][2] = True
    elif grid_name == 'maze':
        grid[1][1] = True
        grid[1][2] = True
        grid[1][3] = True
        grid[2][3] = True
        grid[3][1] = True
        grid[4][1] = True
        grid[4][2] = True
        grid[4][3] = True
        grid[4][4] = True
    return grid


def legal_move(posn_x:int, posn_y:int, move_id:int, grid:list[list[bool]]):
    """
    Produces the new position after a move starting from (posn_x,posn_y) if it is legal on the given grid (i.e. not
    out of bounds or into a wall)

    :param posn_x: The x position (column) from which the move originates
    :type posn_x: int
    :param posn_y: The y position (row) from which the move originates
    :type posn_y: int
    :param move_id: The direction to move, where 0 is up, 1 is down, 2 is left, and 3 is right
    :type move_id: int
    :param grid: The grid on which to move, where False indicates a space and True a wall
    :type grid: list[list[bool]]
    :return: The new (x,y) position if the move was legal, or the old position if it was not
    :rtype: (int, int)
    """
    moves = [[0,-1],[0,1],[-1,0],[1,0]]
    new_x = posn_x + moves[move_id][0]
    new_y = posn_y + moves[move_id][1]
    result = (new_x,new_y)
    if new_x < 0 or new_y < 0 or new_x >= len(grid[0]) or new_y >= len(grid):
        result = (posn_x,posn_y)
    else:
        if grid[new_y][new_x]:
            result = (posn_x,posn_y)
    return result


def run_episodes(sender:Sender, receiver:Receiver, grid:list[list[bool]], num_ep:int, delta:float):
    """
    Runs the reinforcement learning scenario for the specified number of episodes

    :param sender: The Sender agent
    :type sender: Sender
    :param receiver: The Receiver agent
    :type receiver: Receiver
    :param grid: The grid on which to move, where False indicates a space and True a wall
    :type grid: list[list[bool]]
    :param num_ep: The number of episodes
    :type num_ep: int
    :param delta: The chance of termination after every step of the receiver
    :type delta: float [0,1]
    :return: A list of the reward received by each agent at the end of every episode
    :rtype: list[float]
    """
    reward_vals = []

    # Episode loop
    for ep in range(num_ep):
        # Set receiver starting position
        receiver_x = 2
        receiver_y = 2

        # Choose prize position
        prize_x = np.random.randint(len(grid[0]))
        prize_y = np.random.randint(len(grid))
        while grid[prize_y][prize_x] or (prize_x == receiver_x and prize_y == receiver_y):
            prize_x = np.random.randint(len(grid[0]))
            prize_y = np.random.randint(len(grid))

        # Initialize new episode
        # (sender acts)
        sender_state = (prize_x, prize_y)
        sender_action = sender.select_action(sender_state)
        sender_reward = 0

        # Receiver loop
        terminate = False
        while not terminate:
            # receiver state: (m, x, y)
            receiver_state = (sender_action, receiver_x, receiver_y)
            receiver_action = receiver.select_action(receiver_state)
            new_x, new_y = legal_move(receiver_x, receiver_y, receiver_action, grid)
            # update receiver's position and reward
            new_state = (sender_action, new_x, new_y)
            receiver_reward = 1.0 if (new_x, new_y) == (prize_x, prize_y) else 0.0

            if (random.random() < delta or receiver_reward > 0):
                terminate = True

            sender_reward = receiver_reward

            # performed action given state, update Q value in new state
            receiver.update_q(receiver_state, new_state, receiver_action, receiver_reward)
            receiver_x, receiver_y = new_x, new_y

        #Finish up episode
        # (update sender Q-value, update alpha values, append reward to output list)
        sender.update_q(sender_state, sender_action, sender_reward)
        alpha_diff = ((receiver.alpha_i - receiver.alpha_f) / receiver.num_ep)
        receiver.alpha = max(receiver.alpha_f, receiver.alpha - alpha_diff)
        reward_vals.append((sender_reward, receiver_reward))

    return reward_vals


def starter_code():
    # You will need to edit this section to produce the plots and other output required for hand-in

    # Define parameters here
    num_learn_episodes = 100000
    num_test_episodes = 1000
    grid_name = 'fourroom' # 'fourroom', 'maze', or 'empty'
    grid = get_grid(grid_name)
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    epsilon = 0.1
    alpha_init = 0.9
    alpha_final = 0.01

    # Initialize agents
    sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)
    receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)

    # Learn
    learn_rewards = run_episodes(sender, receiver, grid, num_learn_episodes, delta)

    # Test
    sender.epsilon = 0.0
    sender.alpha = 0.0
    sender.alpha_i = 0.0
    sender.alpha_f = 0.0
    receiver.epsilon = 0.0
    receiver.alpha = 0.0
    receiver.alpha_i = 0.0
    receiver.alpha_f = 0.0
    test_rewards = run_episodes(sender, receiver, grid, num_test_episodes, delta)

    # Print results
    print("Average reward during learning: " + str(np.average(learn_rewards)))
    print("Average reward during testing: " + str(np.average(test_rewards)))

def map_action(action):
    # Map the action to a visual arrow representation
    # Actions are [up, down, left, right]
    return {
        0: '↑',
        1: '↓',
        2: '←',
        3: '→'
    }.get(action, ' ')

def output_policy():
    epsilon = 0.1
    num_episodes = 100000
    grid_name = 'fourroom' # 'fourroom', 'maze', or 'empty'
    grid = get_grid(grid_name)
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01

    sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_episodes, epsilon, discount)
    receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_episodes, epsilon, discount)

    run_episodes(sender, receiver, grid, num_episodes, delta)

    # print 4 grids for receiver policy
    for msg in range(num_signals):
        print(f"Receiver's policy for message {msg}:")
        for y in range(5):
            for x in range(5):
                # Find the best action for this state
                state = (msg, x, y)
                max_q_action = receiver.select_action(state)
                print(map_action(max_q_action), end=' ')
            print()
        print()

    # print sender's policy
    print("Sender's policy (best message for each prize location):")
    for y in range(5):
        for x in range(5):
            # Skip if it's the starting position or a wall
            if (x, y) == (2, 2) or grid[y][x]:
                print('S' if (x, y) == (2, 2) else 'X', end=' ')
                continue
            # Find the best message for this state
            state = (x, y)
            maxq_message = sender.select_action(state)
            print(maxq_message, end=' ')
        print()
    print()

def q2b():
    epsilons = [0.01, 0.1, 0.4]
    num_episodes = [10, 100, 1000, 10000, 50000, 100000]
    num_tests = 10

    num_test_episodes = 1000
    grid_name = 'fourroom' # 'fourroom', 'maze', or 'empty'
    grid = get_grid(grid_name)
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01

    results = {}
    std_devs = {}

    for epsilon in epsilons:
        avg_rewards = []
        std_rewards = []
        episode_logs = []
        for num_ep in num_episodes:
            test_rewards = []
            sender, receiver = None, None
            for test in range(num_tests):
                print(f"Epsilon: {epsilon}, num_ep: {num_ep}, test: {test}")
                # Initialize agents
                sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_ep, epsilon, discount)
                receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_ep, epsilon, discount)
                
                # Run learning episodes
                learn_rewards = run_episodes(sender, receiver, grid, num_ep, delta)
                
                sender.epsilon = 0.0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0.0
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0
                eval_rewards = run_episodes(sender, receiver, grid, num_test_episodes, delta)
                # average reward on 1000 episode runs (test_rewards contains 10 elements)
                test_rewards.append(np.average(eval_rewards))

            mean_reward = np.mean(test_rewards)
            std_reward = np.std(test_rewards)

            # Store the results in a dict
            results[(epsilon, num_ep)] = mean_reward
            std_devs[(epsilon, num_ep)] = std_reward

            avg_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            episode_logs.append(np.log10(num_ep))

        # Plotting for the current epsilon
        plt.errorbar(episode_logs, avg_rewards, yerr=std_rewards, label=f'epsilon={epsilon}')

    # Finalize and show the plot
    plt.xlabel('Log of Number of Episodes (log(N_ep))')
    plt.ylabel('Average Test Reward')
    plt.title('Average Test Reward vs. Log of Number of Episodes')
    plt.legend()
    plt.show()

def q2cde(grid_name, num_symbol_arr):
    epsilon = 0.1
    num_eps = [10, 100, 1000, 10000, 50000, 100000]
    num_tests = 10

    num_test_episodes = 1000
    grid = get_grid(grid_name)
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01

    results = {}
    std_devs = {}

    for num_symbol in num_symbol_arr:
        avg_rewards = []
        std_rewards = []
        episode_logs = []
        for num_ep in num_eps:
            test_rewards = []
            for test in range(num_tests):
                print(f"num_symbols: {num_symbol}, num_ep: {num_ep}, test: {test}")
                # Initialize agents
                sender = Sender(num_symbol, len(grid), len(grid[0]), alpha_init, alpha_final, num_ep, epsilon, discount)
                receiver = Receiver(num_symbol, len(grid), len(grid[0]), alpha_init, alpha_final, num_ep, epsilon, discount)
                
                # Run learning episodes
                learn_rewards = run_episodes(sender, receiver, grid, num_ep, delta)
                
                sender.epsilon = 0.0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0.0
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0
                eval_rewards = run_episodes(sender, receiver, grid, num_test_episodes, delta)
                # average reward on 1000 episode runs (test_rewards contains 10 elements)
                test_rewards.append(np.average(eval_rewards))

            mean_reward = np.mean(test_rewards)
            std_reward = np.std(test_rewards)

            # Store the results in a dict
            results[(epsilon, num_ep)] = mean_reward
            std_devs[(epsilon, num_ep)] = std_reward

            avg_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            episode_logs.append(np.log10(num_ep))

        # Plotting for the current num_symbols
        plt.errorbar(episode_logs, avg_rewards, yerr=std_rewards, label=f'num_symbol={num_symbol}')

    # Finalize and show the plot
    plt.xlabel('Log of Number of Episodes (log(N_ep))')
    plt.ylabel('Average Test Reward')
    plt.title('Average Test Reward vs. Log of Number of Episodes')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # part b
    # q2b()
    # output_policy()

    # part c
    # q2cde("four room", [2, 4, 10])

    # part d
    # q2cde("maze", [2, 3, 5])

    # part e
    q2cde("empty", [1])


    


