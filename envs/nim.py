import numpy as np


class NIM(object):
    def __init__(self, rows=5):
        self.rows = rows
        self.board = [[1] * i for i in range(1, self.rows+1)]
        self.current_player = 0
        self.turns = 0
        self.action_dim = self.get_total_actions()
        self.state_dim = self.get_total_actions() - 1


    def _is_terminal(self):
        if sum([sum(r) for r in self.board]) == 0:
            return True
        else:
            return False


    def get_total_actions(self):
        return sum(list(range(1, self.rows+1, 1))) + 1


    def get_available_actions(self):
        start_ids = list(range(self.rows))
        start_ids = list(np.cumsum(start_ids))
        available_actions = []
        for i, id in enumerate(start_ids):
            for c_id, c in enumerate(self.board[i]):
                if c == 1:
                    available_actions.append(id + c_id)
        if self._is_terminal():
            available_actions = available_actions + [self.get_total_actions()-1]
        else:
            available_actions = available_actions
        return available_actions


    def get_onehot_available_actions(self):
        ohaa = [0] * self.get_total_actions()
        aa = self.get_available_actions()
        for i in range(len(ohaa)):
            if i in aa:
                ohaa[i] = 1
        return ohaa


    def get_state(self):
        return self.get_onehot_available_actions()[:-1]


    def step(self, action):
        assert action in self.get_available_actions()
        if action == self.get_total_actions() - 1:
            return self.get_state(), -1, True, None, None
        start_ids = list(range(self.rows+1))
        start_ids = list(np.cumsum(start_ids))
        for i, id in enumerate(start_ids):
            if action < id:
                break
        choose_r = i - 1
        count = action - start_ids[choose_r] + 1
        origin_counts = sum(self.board[choose_r])
        for i in range(origin_counts-count, origin_counts):
            self.board[choose_r][i] = 0

        if self._is_terminal():
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.current_player = (self.current_player + 1) % 2
        if self.current_player == 0:
            self.turns += 1

        return self.get_state(), reward, done, None, None


    def reset(self):
        self.board = [[1] * i for i in range(1, self.rows+1)]
        self.current_player = 0
        self.turns = 0
        return self.get_state(), None


    def close(self):
        pass


# game = NIM(5)
# game.step(7)
# game.step(11)
# game.step(15)
# game.step(0)
# game.step(5)
# game.step(15)