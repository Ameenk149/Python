from enum import Enum
from typing import Optional, Callable, Tuple
import numpy as np

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played

class SavedState:
    pass

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros((6,7))

def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    raw_board = ""

    print("-----BOARD-----")

    for row in board:
        if(raw_board!=""):
            raw_board += ","
        raw_row = "|"
        for col in row:
            raw_board += str(int(col))
            if col == PLAYER1:
                raw_row+="X"+"|"
            elif col == PLAYER2:
                raw_row+="O"+"|"
            else:
                raw_row+=" "+"|"
        
        print(raw_row)
    print("---------------")
    print("|0|1|2|3|4|5|6|")
    return raw_board

def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last 
    board state as a string.
    """
    raise NotImplementedError()

def apply_player_action(
    board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    work_col = action
    work_row = -1
    for i, row in enumerate(board):
        if row[work_col]>0:
            work_row = i-1
            break

    if 0 not in board.T[work_col]:
        return board

    board[work_row, work_col] = player
    return board

def connected_four(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    player_board = (board==player)+0
    signature = np.array([1,1,1,1])
    N = 4
    
    horizontal_board = player_board
    vertical_board = player_board.T
    for row in horizontal_board:
        max_count = 0
        for col in row:
            if col>0:
                max_count+=1
            else:
                max_count=0
            
            if max_count>=4:
                return True

    for row in vertical_board:
        max_count = 0
        for col in row:
            if col>0:
                max_count+=1
            else:
                max_count=0
            
            if max_count>=4:
                return True

    possible_diagonals = [player_board[::-1,:].diagonal(i) for i in range(-player_board.shape[0]+1,player_board.shape[1])]
    possible_diagonals.extend(player_board.diagonal(i) for i in range(player_board.shape[1]-1,-player_board.shape[0],-1))

    for diagonal in possible_diagonals:
        if len(diagonal)<4:
            continue
        max_count = 0
        for col in diagonal:
            if col>0:
                max_count+=1
            else:
                max_count=0

            if max_count>=4:
                return True

    return False

def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game, 
    or is play still on-going (GameState.STILL_PLAYING)?
    """

    if np.all(board>0):
        return GameState.IS_DRAW

    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    else:
        return GameState.STILL_PLAYING