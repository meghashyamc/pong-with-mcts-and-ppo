# pylint: disable-all
from __future__ import division

"""
Original MCTS implementation by Paul Sinclair
Source:  https://github.com/pbsinclair42/MCTS
Specific change made are:
 - More debug logging has been added
 - If a pre-trained PPO is provided, it is used in the expansion phase to prioritize nodes/actions to explore first
 - If a pre-trained PPO is provided, it is used to select the best child using P-UCT during the selection phase.
"""

"""
The MIT Licence

Copyright 2018 Paul Sinclair

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import time
import math
import random
import torch
import numpy as np
from src.logger.logger import logger
from src.algorithms.ppo import PPO, device


def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class treeNode:
    def __init__(self, state, parent, prior_probability=0):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
        self.P = prior_probability  # Prior probability from PPO


class mcts:
    def __init__(
        self,
        timeLimit=None,
        iterationLimit=None,
        explorationConstant=1 / math.sqrt(2),
        rolloutPolicy=randomPolicy,
        ppo_agent: PPO = None,
    ):
        self.ppo_agent = ppo_agent
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = "time"
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = "iterations"
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState):
        self.root = treeNode(initialState, None)

        if self.limitType == "time":
            counter = 0
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                counter += 1
                self.executeRound()
            logger.debug(f"iterations: {counter}")
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        action = self.getAction(self.root, bestChild)
        return action

    def executeRound(self):
        logger.debug("\n--- New Round ---")
        node = self.selectNode(self.root)
        logger.debug(f"Selected node: {node.state}")
        if node.isTerminal:
            # If terminal node, get the reward directly
            reward = node.state.getReward()
            logger.debug(f"Terminal node reward: {reward}")
        else:
            reward = self.rollout(node.state)
            logger.debug(f"Rollout reward: {reward}")
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            logger.debug(f"Current node: {node.state}")
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
                logger.debug(f"Fully expanded, selecting best child: {node.state}")
            else:
                logger.debug("Not fully expanded, expanding node")
                node_to_return = self.expand(node)
                logger.debug(f"Expanded node: {node_to_return.state}")
                return node_to_return
        logger.debug(f"  Reached terminal node: {node.state}")
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        logger.debug(f"  Expanding node. Possible actions: {actions}")
        if self.ppo_agent:
            game_state = node.state.getState()
            with torch.no_grad():
                state = torch.FloatTensor(game_state).to(device)
                action_probs = self.ppo_agent.policy_old.actor(state)
                action_probs = action_probs.cpu().numpy().flatten()
                logger.debug(f"action_probs: {action_probs}")
                logger.debug(f"action_probs shape: {action_probs.shape}")
            # Clip the probabilities to avoid extremely low values
            epsilon = 1e-6  # Small constant to prevent zero probabilities
            action_probs = np.maximum(action_probs, epsilon)

            action_prob_dict = {action: action_probs[action] for action in actions}
            # Normalize probabilities
            total_prob = sum(action_prob_dict.values())
            if total_prob > 0:
                action_prob_dict = {
                    action: prob / total_prob
                    for action, prob in action_prob_dict.items()
                }
            else:
                action_prob_dict = {action: 1.0 / len(actions) for action in actions}

        else:
            # If no PPO agent, assign equal probability to all actions
            action_prob_dict = {action: 1.0 / len(actions) for action in actions}

        for action in actions:
            if action not in node.children:
                new_state = node.state.takeAction(action)
                prior_prob = action_prob_dict.get(action, 0)
                newNode = treeNode(new_state, node, prior_probability=prior_prob)

                node.children[action] = newNode
                logger.debug(
                    f"    Created new child with action {action}: {newNode.state}"
                )
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                    logger.debug("    Node is now fully expanded")
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        logger.debug("  Backpropagating")
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            logger.debug(
                f"    Node: {node.state}, Visits: {node.numVisits}, Total Reward: {node.totalReward}"
            )
            node = node.parent

    def getBestChild(self, node, explorationValue):
        logger.debug(f"  Getting best child for node: {node.state}")
        bestValue = float("-inf")
        bestNodes = []

        for action, child in node.children.items():
            if not self.ppo_agent:
                nodeValue = (
                    child.totalReward / child.numVisits
                    + explorationValue
                    * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
                )
            else:
                Q = child.totalReward / child.numVisits if child.numVisits > 0 else 0
                U = (
                    explorationValue
                    * child.P
                    * math.sqrt(node.numVisits)
                    / (1 + child.numVisits)
                )
                nodeValue = Q + U

            logger.debug(f"    Action: {action}, Value: {nodeValue}")

            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        bestChild = random.choice(bestNodes)
        logger.debug(f"  Best child selected: {bestChild.state}")
        return bestChild

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
