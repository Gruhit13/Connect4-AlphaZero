## Connect4 with AlphaZero

This project implements a game-playing agent for Connect4 using AlphaZero, a groundbreaking algorithm inspired by AlphaGo. In this `readme.md` file, we provide a simple explanation of AlphaZero and how it utilizes Monte Carlo Tree Search (MCTS) to play games.

### What is AlphaZero?

AlphaZero is a deep reinforcement learning algorithm developed by DeepMind. It combines deep neural networks and Monte Carlo Tree Search to achieve superhuman performance in various board games, including Go, chess, and shogi. AlphaZero learns the game solely by self-play, starting with random play and gradually improving through iterative training.

### Monte Carlo Tree Search (MCTS)

MCTS is a heuristic search algorithm used in decision-making processes, particularly in games with high branching factors and large state spaces. It builds a search tree by simulating random games starting from the current game state and using the results to guide the search towards promising moves.

MCTS consists of four main steps:

1. **Selection**: Starting from the root of the search tree, MCTS traverses down the tree by selecting child nodes based on some selection policy, typically UCB1 (Upper Confidence Bound 1).

2. **Expansion**: When a leaf node is reached (a node with unexplored moves), MCTS expands the tree by adding child nodes corresponding to each possible move from that state.

3. **Simulation (Rollout)**: MCTS performs a Monte Carlo simulation by randomly playing out the game from the current state until reaching a terminal state. The result of the simulation is a win, loss, or draw.

4. **Backpropagation**: The result of the simulation is backpropagated up the tree, updating the statistics (e.g., visit count, win count) of all nodes traversed during the selection.

By repeating these steps for a certain number of iterations, MCTS gradually improves its estimates of the best moves to play from a given state.

### AlphaZero and MCTS

AlphaZero combines deep neural networks with MCTS to achieve strong game-playing performance. The neural network is trained to evaluate positions and predict move probabilities based on self-play data. During the MCTS search, the neural network guides the selection of moves by providing prior probabilities and state evaluations.

AlphaZero utilizes MCTS in the following way:

1. **Policy Network**: The neural network provides prior probabilities for each move from a given game state. These probabilities guide the selection and exploration during the MCTS search, biasing it towards more promising moves.

2. **Value Network**: The neural network also evaluates the state value, estimating the expected outcome of the game from the current state. This evaluation helps in the backpropagation step, where the value is used to update the statistics of visited nodes.

By combining the exploration and exploitation capabilities of MCTS with the predictive power of neural networks, AlphaZero efficiently learns to play complex games at a high level.

### Video Demonstration

To see the Connect4 agent in action, please refer to the accompanying video demonstration provided in the video. The video showcases the gameplay and highlights the performance of the AlphaZero-based agent.

https://github.com/Gruhit13/Connect4-AlphaZero/assets/64111603/b056bc20-f152-4b9f-864b-6e93ec05acb8


For detailed instructions on how to set up and run the Connect4 project, please refer to the project documentation and code files.

**Note:** Make sure to acknowledge any relevant papers, libraries, or resources used in your project by including appropriate references in the `readme.md` file.
