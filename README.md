# Attempt at Implementing DeepNash

This repository is a modification based on [baskuit/R-NaD: Experimentation with Regularized Nash Dynamics on a GPU accelerated game (github.com)](https://github.com/baskuit/R-NaD), especially the `vtrace.py` file.

This repository implements possible implementations of DeepNash in two scenarios: Lasertag and NIM game.

## Lasertag

Lasertag is a zero-sum game where two agents are in a grid environment. Each agent can move and shoot lasers. When a laser hits the opponent, the agent gets a reward, while the hit agent receives the same punishment.

Both agents make decisions simultaneously in this scenario. The Laser_tag gym environment in this repository is modified based on [younggyoseo/lasertag-v0: Implementation of Deepmind's LaserTag-v0 game in A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning(2017) (github.com)](https://github.com/younggyoseo/lasertag-v0). Use the following code to install it.

```shell
cd lasertag-v0
pip install -e .
```

![image](https://github.com/deligentfool/DeepNash/blob/master/resource/Lasertag.gif)

![image](https://github.com/deligentfool/DeepNash/blob/master/resource/Lasertag_result.jpg)

## NIM Game

[NIM GAME-Wiki](https://en.wikipedia.org/wiki/Nim)

The NIM game involves `n` piles of different items, with arbitrary numbers of items in each pile. Two players take turns removing any number of items from a single pile, at least one and at most all items, but they cannot take nothing or items from multiple piles. The player who removes the last item according to the rules, making the opponent unable to take any items, wins. One of the players has a winning strategy in this game.

Suppose there are `n` piles of items, with 1 item in the first pile, 2 items in the second pile, and so on, up to `n` items in the `n-th` pile. It can be observed that when `n` is odd, the second player has a winning strategy. Conversely, when `n` is even, the first player has a winning strategy.

![image](https://github.com/deligentfool/DeepNash/blob/master/resource/NIM_result.jpg)
