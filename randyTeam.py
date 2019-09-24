# myTeam.py
# ---------
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


from __future__ import print_function
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np

score_change = 0
start_score = 0
start_time = time.time()
elapsed_time = 0
minimum = True
#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    #if self.index == 1:
      #print(values, file=sys.stderr)
      # print(self.getPreviousObservation(), file=sys.stderr)

    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    # if self.index == 1:
    #   print(bestActions, file=sys.stderr)
    '''
    foodLeft = len(self.getFood(gameState).asList())
    if foodLeft <= 2 or gameState.getAgentState(self.index).numCarrying > 5:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
      '''
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)

    #if self.index == 1:
      #print(str(features) + str(weights), file=sys.stderr)
      # print(gameState.getAgentState(self.index)) # Print out a text representation of the world.

    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def getScore(self, gameState):
    return gameState.data.score

  def getFoodCentroid(self, gameState):
    """ 
    This function searches the friendly team's half and calculates 
    the geometric mean of food items.
    """

    matrix = CaptureAgent.getFoodYouAreDefending(self, gameState)

    food_row = [0] * 16
    food_row = np.array(food_row)
    centroid_row = 0
    food_col = [0] * 16
    food_col = np.array(food_col)
    centroid_col = 0
    num_food = 0
    if (self.red):
      for x in range(16):
        food_col[x] = 0
        for y in range(16):
          if(matrix[x][y] == True):
            food_col[x] = food_col[x] + 1
        centroid_col += food_col[x] * x
      for y in range(16):
        food_row[y] = 0 
        for x in range(16):
          if (matrix[x][y] == True):
            num_food += 1
            food_row[y] = food_row[y] + 1
        centroid_row += food_row[y] * y
      position = int(round(centroid_col / num_food)), int(round(centroid_row / num_food))
    else: 
      for x in range(16):
        food_col[x] = 0
        for y in range(16):
          if(matrix[x+16][y] == True):
            food_col[x] = food_col[x] + 1
        centroid_col += food_col[x] * x
      for y in range(16):
        food_row[y] = 0 
        for x in range(16):
          if (matrix[x+16][y] == True):
            num_food += 1
            food_row[y] = food_row[y] + 1
        centroid_row += food_row[y] * y
      position = int(round(centroid_col / num_food)) + 16, int(round(centroid_row / num_food))
    return position
  
  def makeLegalPosition(self, gameState, x, y):
    """
    This function will adjust the position if it is a wall tile
    to an adjacent position.
    Red will try a location on the right first to be closer to 
    the border and blue will look for locations on the left.
    """
    if (self.red):
      if (gameState.data.layout.isWall((x,y))):
        if (not gameState.data.layout.isWall((x+1, y))):
          return (x+1, y)
        elif (not gameState.data.layout.isWall((x+1, y-1))):
          return (x+1, y-1)
        elif (not gameState.data.layout.isWall((x+1, y+1))):
          return (x+1, y+1)
        elif (not gameState.data.layout.isWall((x, y-1))):
          return (x, y-1)
        elif (not gameState.data.layout.isWall((x, y+1))):
          return (x, y+1)
        elif (not gameState.data.layout.isWall((x-1, y-1))):
          return (x-1, y-1)
        elif (not gameState.data.layout.isWall((x-1, y))):
          return (x-1, y)
        else:
          return (x-1, y-1)
      else:
        return (x,y)
    else:
      if (gameState.data.layout.isWall((x,y))):
        if (not gameState.data.layout.isWall((x-1, y))):
          return (x-1, y)
        elif (not gameState.data.layout.isWall((x-1, y+1))):
          return (x-1, y+1)
        elif (not gameState.data.layout.isWall((x-1, y-1))):
          return (x-1, y-1)
        elif (not gameState.data.layout.isWall((x, y+1))):
          return (x, y+1)
        elif (not gameState.data.layout.isWall((x, y-1))):
          return (x, y-1)
        elif (not gameState.data.layout.isWall((x+1, y+1))):
          return (x+1, y+1)
        elif (not gameState.data.layout.isWall((x+1, y))):
          return (x+1, y)
        else:
          return (x+1, y-1)
      else:
        return (x,y)
  


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    global start_score
    global score_change
    global minimum
    global start_time
    global elapsed_time
    retreat = True
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Determine if the enemy is closer to you than they were last time
    # and you are in their territory.
    # Note: This behavior isn't perfect, and can force Pacman to cower 
    # in a corner.  I leave it up to you to improve this behavior.
    startDistance = self.getMazeDistance(myPos, self.start)
    close_dist = 9999.0
    if gameState.getScore() != 0:
      retreat = False
      opp_fut_state = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      chasers = [p for p in opp_fut_state if p.getPosition() != None and not p.isPacman]
      if len(chasers) > 0:
        close_dist = min([float(self.getMazeDistance(myPos, c.getPosition())) for c in chasers])
      
      # View the action and close distance information for each 
      # possible move choice.
    #print("Action: "+str(action))
    #print("\t\t"+str(close_dist), sys.stderr)
      
    foodLeft = len(self.getFood(gameState).asList())
    food_carried = gameState.getAgentState(self.index).numCarrying
    
    x, y = myPos
    x = int(x)
    y = int(y)
    
    count = 0
    if gameState.hasWall(x-1, y):
        count = count + 1
    if gameState.hasWall(x+1, y):
        count = count + 1
    if gameState.hasWall(x, y-1):
        count = count + 1
    if gameState.hasWall(x, y+1):
        count = count + 1
    

    if close_dist > 9:
        features['fleeEnemy'] = 1.0/close_dist
    elif close_dist > 5:
        features['fleeEnemy'] = 100.0
        features['distanceToFood'] = startDistance
    elif close_dist > 2:
        features['fleeEnemy'] = 1000.0
        features['distanceToFood'] = startDistance
    elif close_dist > 0:
        features['fleeEnemy'] = 10000.0
        features['distanceToFood'] = startDistance
        
        
    if foodLeft <= 2 or food_carried > 5:
        features['distanceToFood'] = startDistance
    
    if count > 2 and close_dist <= 7:
        features["distanceToFood"] = startDistance
    
    if retreat and food_carried > 0:
        features["distanceToFood"] = startDistance
        
    elapsed_time = time.time() - start_time
    score_change = gameState.getScore() - start_score
    
    if elapsed_time > 6 and score_change == 0 and not retreat:
        if minimum:
            minimum = False
        else:
            minimum = True
        start_time = time.time()
        start_score = gameState.getScore()
    '''    
    if not minimum:
        features['fleeEnemy'] = 0
        features['distanceToFood'] = 0
        features['successorScore'] = 0
    ''' 
    
    if(self.red and ReflexCaptureAgent.getScore(self, gameState) > 0 or
       not self.red and ReflexCaptureAgent.getScore(self, gameState) < 0):

       # Computes distance to invaders we can see
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      defenders = [b for b in enemies if not b.isPacman and b.getPosition() != None]
      features['numInvaders'] = len(invaders)

      if len(invaders) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)
    
      foodCentroid = ReflexCaptureAgent.getFoodCentroid(self, gameState)
      foodCentroid = ReflexCaptureAgent.makeLegalPosition(self, gameState, foodCentroid[0], foodCentroid[1])
      # features['foodCentroid'] = self.getMazeDistance(myPos, foodCentroid)

      capsule = ReflexCaptureAgent.getCapsulesYouAreDefending(self,gameState)
      if (capsule):
        # features['capsule'] = self.getMazeDistance(myPos, capsule[0])
        meanCapsFood = ((capsule[0][0] + foodCentroid[0])/2, (capsule[0][1] + foodCentroid[1])/2)
        meanCapsFood = ReflexCaptureAgent.makeLegalPosition(self, gameState, meanCapsFood[0], meanCapsFood[1])
        features['mean'] = self.getMazeDistance(myPos, meanCapsFood)      
      else:
        meanCapsFood = foodCentroid
        features['mean'] = self.getMazeDistance(myPos, meanCapsFood)
      features['onDefense'] = 1 
      meanCapsFood = ((meanCapsFood[0] + 16)/2, (meanCapsFood[1])/2)
      meanCapsFood = ReflexCaptureAgent.makeLegalPosition(self, gameState, meanCapsFood[0], meanCapsFood[1])
      # features['foodCentroid'] = 0
      # features['capsule'] = 0
      print(meanCapsFood)
      features['mean'] = self.getMazeDistance(myPos, meanCapsFood)  

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -15, 
            'mean': -10, 'successorScore': 100, 'distanceToFood': -1, 'fleeEnemy': -100.0}

  
class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    defenders = [b for b in enemies if not b.isPacman and b.getPosition() != None]
    features['numInvaders'] = len(invaders)

    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
    
    foodCentroid = ReflexCaptureAgent.getFoodCentroid(self, gameState)
    # features['foodCentroid'] = self.getMazeDistance(myPos, foodCentroid)

    capsule = ReflexCaptureAgent.getCapsulesYouAreDefending(self,gameState)
    if (capsule):
      # features['capsule'] = self.getMazeDistance(myPos, capsule[0])
      meanCapsFood = ((capsule[0][0] + foodCentroid[0])/2, (capsule[0][1] + foodCentroid[1])/2)
      meanCapsFood = ((meanCapsFood[0]+16)/2, (meanCapsFood[1]+8)/2)
      meanCapsFood = ReflexCaptureAgent.makeLegalPosition(self, gameState, meanCapsFood[0], meanCapsFood[1])
      features['mean'] = self.getMazeDistance(myPos, meanCapsFood)      
    else:
      meanCapsFood = foodCentroid
      meanCapsFood = ((meanCapsFood[0]+16)/2, (meanCapsFood[1]+8)/2)
      meanCapsFood = ReflexCaptureAgent.makeLegalPosition(self, gameState, meanCapsFood[0], meanCapsFood[1])
      features['mean'] = self.getMazeDistance(myPos, meanCapsFood)
    
    if(self.red and ReflexCaptureAgent.getScore(self, gameState) > 0 or
       not self.red and ReflexCaptureAgent.getScore(self, gameState) < 0):
      print(meanCapsFood)
      meanCapsFood = ((meanCapsFood[0] + 16)/2, (meanCapsFood[1] + 16)/2)
      meanCapsFood = ReflexCaptureAgent.makeLegalPosition(self, gameState, meanCapsFood[0], meanCapsFood[1])
      print(meanCapsFood)
      # features['foodCentroid'] = 0
      # features['capsule'] = 0
      print()
      features['mean'] = self.getMazeDistance(myPos, meanCapsFood)  
    
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -15, 'mean': -10}