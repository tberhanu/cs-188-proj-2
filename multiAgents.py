# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()

        updated = successorGameState.getScore()
        pacman_postition = newPos
        ghost_position = newGhostStates[0].getPosition()
        ghost_distance = manhattanDistance(pacman_postition, ghost_position)
        food_dist = [manhattanDistance(newPos, x) for x in newFood.asList()]
        # min_dist = float("inf")
        # min_dist = 9999

        # for width in range(len(newFood[0])):
        #     for length in range(len(newFood)):
        #         if newFood[width][length]:
        #             dist = manhattanDistance(newPos, (width, length))
        #             if dist < min_dist:
        #                 min_dist = dist
        # if min_dist == 9999:
        #     min_dis = 0

        # list = []
        # for x in range(self.width):
        #     for y in range(self.height):
        #         if self[x][y] == key: list.append((x, y))
        # return list

        if len(newScaredTimes) != 0 and newGhostStates[0].scaredTimer != 0:
            updated = updated - 20.0 / newGhostStates[0].scaredTimer
        if ghost_distance != 0:
            updated = updated - 10.0 / ghost_distance
        if len(food_dist) != 0:
            updated = updated  + 10.0 / min(food_dist)

        return updated

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    minimax_score = 0

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        result = self.maximizer(gameState, self.depth)

        return result[1]

    def maximizer(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), ""

        actions_lst = gameState.getLegalActions()
        max_score = float("-inf")

        index = 0
        for i in range(len(actions_lst)):
            score = self.minimizer(gameState.generateSuccessor(self.index, actions_lst[i]), 1, depth)
            if score > max_score:
                max_score = score
                index = i

        best_action = actions_lst[index]
        return max_score, best_action

    def minimizer(self, gameState, agent, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        actions_lst = gameState.getLegalActions(agent)
        scores = []
        if (agent == gameState.getNumAgents() - 1):
            for i in range(len(actions_lst)):
                score = self.maximizer(gameState.generateSuccessor(agent, actions_lst[i]), depth - 1)
                scores.append(score)

        else:
            for i in range(len(actions_lst)):
                score = self.minimizer(gameState.generateSuccessor(agent, actions_lst[i]), agent + 1, depth)
                scores.append(score)
        return min(scores)

        # util.raiseNotDefined()
        #########################################################################################

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-inf")
        beta = float("inf")
        result = self.maximizer(gameState, self.depth, 0, alpha, beta)
        return result[1]


    def maximizer(self, state, depth, index, alpha, beta):

        if  state.isLose() or state.isWin() or depth == 0:
            return self.evaluationFunction(state), ""
        value = float("-inf")
        action_lst = state.getLegalActions(index)
        for action in action_lst:
            successor = state.generateSuccessor(index, action)
            score = self.minimizer(successor, depth - 1, index + 1, alpha, beta)
            if score > value:
                value = score
                move = action
            if beta < value:
                return value, move
            alpha = max(alpha, value)
        return value, move

    def minimizer(self, state, depth, index, alpha, beta):
        if depth < 0 or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        value = float("inf")
        actions_lst = state.getLegalActions(index)

        for action in actions_lst:
            successor = state.generateSuccessor(index, action)
            if index >= state.getNumAgents() - 1:
                score = self.maximizer(successor, depth, 0, alpha, beta)[0]
            else:
                score = self.minimizer(successor, depth, index + 1, alpha, beta)
            if score < value:
                value = score
            if value < alpha:
                return value
            beta = min(beta, value)
        return value


        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.getActionHelper(gameState, self.depth, 0)[1]

    def getActionHelper(self, gameState, depth, agentIndex):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            eval_result = self.evaluationFunction(gameState)
            return (eval_result, '')
        else:
            if agentIndex == gameState.getNumAgents() - 1:
                depth -= 1
            if agentIndex == 0:
                maxAlpha = -99999999
            else:
                maxAlpha = 0
            maxAction = ''
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                result = self.getActionHelper(gameState.generateSuccessor(agentIndex, action), depth,
                                              nextAgentIndex)
                if agentIndex == 0:
                    if result[0] > maxAlpha:
                        maxAlpha = result[0]
                        maxAction = action
                else:
                    maxAlpha += 1.0 / len(actions) * result[0]
                    maxAction = action
            return (maxAlpha, maxAction)
            #########################################################################################

            # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    evalNum = 0
    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    minDistance = 10000
    setMinDistance = False
    for foodPosition in foodPositions:
        foodDistance = util.manhattanDistance(pacmanPosition, foodPosition)
        if foodDistance < minDistance:
            minDistance = foodDistance
            setMinDistance = True
    if setMinDistance:
        evalNum += minDistance
    evalNum += 1000 * currentGameState.getNumFood()
    evalNum += 10 * len(currentGameState.getCapsules())
    ghostPositions = currentGameState.getGhostPositions()
    for ghostPosition in ghostPositions:
        ghostDistance = util.manhattanDistance(pacmanPosition, ghostPosition)
        if ghostDistance < 2:
            evalNum = 9999999999999999
    evalNum -= 10 * currentGameState.getScore()
    # print("min distance: " + str(minDistance) + " num food: " + str(len(foodPositions)) + " eval num: " + str(evalNum*(-1)))
    return evalNum * (-1)
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

