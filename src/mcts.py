from model import Model
from typing import Union, Tuple
from game import Connect4
from config import Config

import torch
from torch import Tensor

import numpy as np

class Node:
    def __init__(self, state: Union[Connect4, None], model: Model, name: str):

        # Current state that the node represent
        self.state = state

        # Name of the node to trace it
        self.name = name

        # A model instance that the node will use to get value and policy
        self.model = model

        # visit count
        self.N = 0

        # Intermediate reward value
        self.W = 0

        # value of the node
        self.value = None

        # Prior policy for action from this node
        self.policy = None

        # Set the winner of the current node.
        # Node by default indicating no one has won
        self.win = None

        # Children of current node
        self.children = {}

        # valid and invalid actions that can be take from this node
        self.valid_actions = None
        self.invalid_actions = None

        # Set the valid and invalid actions
        self.set_valid_actions()

        # Initialize the branches to the childrens
        self.initialize_edges()

    # Set the valid actions that can be taken from the state that
    # the node represent
    def set_valid_actions(self) -> None:
        if self.state is not None:
            self.valid_actions = self.state.get_valid_moves()
            self.invalid_actions = ~self.valid_actions

    # initialize the edges from this node to potential childrens
    def initialize_edges(self) -> None:
        if self.state is not None:
            self.children = {}
            for act, valid_move in enumerate(self.valid_actions):
                if valid_move:
                    # set state as none for childrens as we do not have it
                    self.children[act] = Node(
                        state=None,
                        model=self.model,
                        name=self.name + '_' + str(act)
                    )

    def preprocess_state(self, x:np.ndarray) -> Tensor:
        x = torch.tensor(x, dtype=torch.float32, device=Config.device)
        x = x.unsqueeze(0)
        return x

    # define the forward pass for the current node
    def forward(self) -> None:
        with torch.no_grad():
            value, policy = self.model(self.preprocess_state(self.state.get_state()))

        value = value[0, 0]
        policy = policy[0]

        # Mask the invalid actions
        policy[self.invalid_actions] = 0.

        # Prevent from all probability from turning 0
        if policy.sum() == 0:
            policy[self.valid_actions] = 1.

        policy = policy.softmax(dim=-1)

        self.value = value.detach().cpu().numpy()
        self.policy = policy.detach().cpu().numpy()


    # Get policy for the current node
    def get_policy(self) -> np.ndarray:
        if self.policy is None:
            self.forward()

        return self.policy

    # Get the value associated with the node
    def get_value(self) -> float:
        if self.value is None:
            self.forward()

        return self.value
    
class MCTS_NN:
    def __init__(self, state:Connect4, model:Model, log=None):
        self.root = Node(state=state, model=model, name='root')

        if log is not None:
            self.log = log

    # For the simulation on the Monte-carlo tree
    def selection(self, node: Node, add_dirichlet:bool=False, iter:int=0) -> float:
        # Get the best child of the current node
        # self.log.write(f'\nSelecting Best child of {node.name}')
        best_child, best_action = self.get_best_child(node, add_dirichlet, iter)
        # self.log.write(f"Iteartion {iter} - Best Action - {best_action} - Node: {node.name}")

        # If the child is a leaf node(i.e.) either is terminal or is not expanded
        # expand that node
        if best_child.state is None:
            # self.log.write(f'\nExpanding node {best_child.name}')
            val = self.expolore_and_expand(parent=node, child=best_child, action=best_action, iter=iter)

        # If the node is already expanded than traverse that node further
        else:
            # As per paper only add dirichlet noise for root node's
            # child selection and not later on
            # self.log.write(f'\nSelecting node further on {best_child.name}')
            val = self.selection(node=best_child, add_dirichlet=False, iter=iter)

        node.N += 1
        node.W += val

        return -val

    # Expore and expand the tree
    def expolore_and_expand(self, parent: Node, child: None, action: int, iter=0) -> float:
        # self.log.write(f'\n<========== Explore or Expand Iteration {iter} ==========>')
        # Check if the current state is a terminal state
        if child.win is None:
            # It is not expanded and is not terminal
            # Perform the action for the parent state to get the next state
            next_state, win = parent.state.drop_piece(action)

            # First check if somone won in this next state
            if win is not None:
                val = -1 if win == parent.state.player_1 else 1
                child.win = win
                # self.log.write(f'\nPlayer Turn for child is {next_state.player_1} | [Winner Found]')
                # self.log.write(f'\nWinner in that state {win} - child.Value is {val}')
                # self.log.write(f'\nWinning Child in state {child.name}: state\n{next_state}\n')
                # self.log.write('='*100)
                # self.log.write('\n')

            # else check if the next state results in draw
            elif next_state.is_draw():
                # 0 value if no one has won in the state
                val = 0

                # 0 for win means no one won
                child.win = 0
                # self.log.write(f'\nPlayer Turn for child is {next_state.player_1}')
                # self.log.write(f'\nDraw Child in state {child.name}: state\n{next_state}\n')
                # self.log.write('='*100)
                # self.log.write('\n')

            # if the next_state is not winning nor it is draw
            # then expand it normally
            else:
                # If no one is winning yet then get the value for the current
                # state from the child's mode and set it
                child.state = next_state
                child.set_valid_actions()
                child.initialize_edges()
                val = child.get_value()
                # self.log.write(f'\nPlayer Turn for child is {next_state.player_1} | [No Winner]')
                # self.log.write(f'\nLeaf node expanded for "{child.name}" with val {val:.5f}\n')
                # self.log.write('='*100)
                # self.log.write('\n')

        else:
            # If the current child represent a draw state then give value 0
            if child.win == 0:
                # self.log.write(f'\nTerminal DRAW state reached for child {child.name}\n')
                # self.log.write('='*100)
                # self.log.write('\n')
                val = 0

            # If the winner in child node was the player who played a move
            # in the parent node then set -1 as value as it means that
            # the player in child node has lost
            elif child.win == parent.state.player_1:
                # self.log.write(f'\nTerminal Parent Winning state reached for child {child.name}\n')
                # self.log.write('='*100)
                # self.log.write('\n')
                val = -1

            # if the winner of child node is the same as the player of child node
            # then provide value of +1
            else:
                # self.log.write(f'\nTerminal child Winning state reached for child {child.name}\n')
                # self.log.write('='*100)
                # self.log.write('\n')
                val = 1

        # Update the visit count and intermidiate reward of child node
        child.N += 1
        child.W += val

        # Return negative of val because the player in parent node will be
        # the opposite player from the current node. Hence what is good
        # for current node's player should be bad for the parent node's player
        return -val


    # Calculate the PUCT score for a node's children
    def get_puct_score(self, parent: Node, child: Node, prior: float) -> float:
        # PUCT is the sum of q_value of current node + the U(S, a)
        q_value = 0
        if child.N == 0:
            q_value = 0
        else:
            # q_value = 1 - ((child.W/child.N) + 1)/2
            q_value = -child.W/child.N

        # C_puct represent the exploration constant
        c_puct = 1
        u_sa = c_puct * prior * (np.sqrt(parent.N))/(1+child.N)
        return q_value + u_sa

    def get_dirichlet_noise(self, node: None) -> np.ndarray:
        num_valid_action = node.valid_actions.sum()
        noise_vec = np.random.dirichlet([Config.DIRICHLET_ALPHA]*num_valid_action)
        noise_arr = np.zeros((len(node.valid_actions),), dtype=noise_vec.dtype)
        noise_arr[node.valid_actions] = noise_vec
        return noise_arr

    # Get the best child for any node
    def get_best_child(self, node: Node, add_dirichlet: bool, iter=0) -> Tuple[Node, int]:
        # the best node is simple the one with highest PUCT value
        policy = node.get_policy()

        if add_dirichlet:
            noise_arr = self.get_dirichlet_noise(node)
            policy = (1-Config.EPSILON)*policy + Config.EPSILON*noise_arr

        best_puct = float('-inf')
        best_child = None
        best_action = None
        # self.log.write(f'\n\n==================== Iteration {iter} ====================\n')
        for action, child in node.children.items():
            puct = self.get_puct_score(parent=node, child=child, prior=policy[action])
            # self.log.write(f'{action} - PUCT: {puct:.4f} | N = {child.N} | W = {child.W:.4f} | P = {policy[action]:.4f}\n')
            if puct > best_puct:
                best_puct = puct
                best_child = child
                best_action = action

        return best_child, best_action

    # return the policy pie for the root node based on the visit count
    def get_policy_pie(self, temperature:float=1):
        actions = np.zeros((len(self.root.valid_actions),))

        for action, child in self.root.children.items():
            actions[action] = (child.N)**(1/temperature)

        actions /= actions.sum()

        return actions

    # Traverse the tree by steping to one of the child node of root node
    def update_root(self, action: int) -> None:
        self.root = self.root.children[action]