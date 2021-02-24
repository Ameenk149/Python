import numpy as np
from agents.common import PLAYER1, PLAYER2, GameState
from agents.common import apply_player_action, check_end_state, connected_four
from agents.common import BoardPiece

def get_valid_moves(board: np.ndarray):
    valid_moves = []
    for i, col in enumerate(board.T):
        if col[0]==0:
            valid_moves.append(i)
    return valid_moves

def get_opponent(player: BoardPiece):
    if player==PLAYER1:
        return PLAYER2
    else:
        return PLAYER1

def check_winner(board: np.ndarray):
    winner = None
    if connected_four(board, PLAYER1):
        winner=10

    if connected_four(board, PLAYER2):
        winner=-10

    moves_left = len(get_valid_moves(board))
    if winner == None and moves_left == 0:
        return 0
    elif winner==None and moves_left >0:
        return None
    else:
        return winner

def minimax(board: np.ndarray, depth: int, isMaximizing: bool, player: BoardPiece):
    result = check_winner(board)

    if result!=None:
        return result

    if isMaximizing:
        bestScore = float("-inf")
        validMoves = get_valid_moves(board)
        for move in validMoves:
            tmp_moved = apply_player_action(board, move, player)
            score = minimax(tmp_moved, depth + 1, False, player)
            bestScore = max([score, bestScore])
        return bestScore
    else:
        bestScore = float("inf")
        validMoves = get_valid_moves(board)
        for move in validMoves:
            tmp_moved = apply_player_action(board, move, get_opponent(player))
            score = minimax(tmp_moved, depth + 1, True, get_opponent(player))
            bestScore = min([score, bestScore])
        return bestScore