import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import metrics
from joblib import dump, load

#Function to check whether the input player has won
def check_winner(board,mark):
    return((board[0]==mark and board[1]== mark and board[2]==mark )or #Row 1 

            (board[3]==mark and board[4]==mark and board[5]==mark )or #Row 2

            (board[6]==mark and board[7]==mark and board[8]==mark )or #Row 3

            (board[0]==mark and board[3]==mark and board[6]== mark )or#Column 1 

            (board[1]==mark and board[4]==mark and board[7]==mark )or #Column 2

            (board[2]==mark and board[5]==mark and board[8]==mark )or #Column 3

            (board[0]==mark and board[4]==mark and board[8]==mark )or #Diagonal 1

            (board[2]==mark and board[4]==mark and board[6]==mark )) #Diagonal 2


network = load('chosenNetwork.joblib') #load the saved network

#Construct starting game array
gameArray = [np.zeros(9)]
gameOver = False
totalMoves = 0 #counter to end game at 9 moves
userInput = None
while not gameOver:
  print(gameArray[0][0:3])
  print(gameArray[0][3:6])
  print(gameArray[0][6:9])
  while True:
    userInput = input("Please enter an integer from 0-8 for position.")
    try:
      userInt = int(userInput) #convert to int, if fails due to ValueError, tell user not an int
      if userInt >= 0 and userInt <= 8: #check if int is valid for tic tac toe
        if gameArray[0][userInt] == 0: #check if spot is taken
          break #move on if all checks are passed
        else:
          print("This spot is taken.")
      else:
        print("This is not from 0-8")
    except ValueError:
      print("Not an integer!")
      
  gameArray[0][int(userInput)] = 1 #set spot to player
  totalMoves+= 1
 
  #check if player won or if it is a tie
  if check_winner(gameArray[0], 1):
    print("Player win")
    gameOver = True
    break
  if totalMoves == 9:
    gameOver = True
    print("Tie!\n")
    break
  
  #get network prediction and convert it to int
  networkInput = network.predict(gameArray)
  networkInt = int(networkInput)
  print('Network move: ' + str(networkInt))

  #Check if the AI chose a valid spot. If not, give it the next open spot.
  if gameArray[0][networkInt] == 0:
    gameArray[0][networkInt] = -1
  else:
    
    for i in range(9):
      if gameArray[0][i] == 0:
        print('Network attempted invalid move. Filling next open spot: ' + str(i))
        gameArray[0][i] = -1
        break;
  totalMoves+= 1
  
  #check if AI won or game is tie
  if check_winner(gameArray[0], -1):
    print("Network Win!")
    gameOver = True
    break
  if totalMoves == 9:
    gameOver = True
    print("Tie!\n")
print(gameArray[0][0:3])
print(gameArray[0][3:6])
print(gameArray[0][6:9])
