import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove
from agents.common import PLAYER1, PLAYER2, GameState
from agents.common import apply_player_action, check_end_state, connected_four
from typing import Optional, Callable, Tuple
from random import shuffle
from agents.agent_minimax import *
from agents.agent_learner import *

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    validMoves = get_valid_moves(board)
    shuffle(validMoves)
    bestMove = validMoves[0]
    bestScore = float("-inf")

    use_ml = True

    for move in validMoves:
        tmp_mov = apply_player_action(board, move, player)
        if not use_ml:
            score = minimax(tmp_mov, 0, False, player)
        else:
            score = max(probability(tmp_mov)[0][1:2])
        if score > bestScore:
            bestScore = score
            move = move
        
        print(bestScore)
    return bestMove, saved_state