# MIT 6.034 Games Lab
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            return True
    for i in range(7):
        if not board.is_column_full(i):
            return False 
    return True 

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    if is_game_over_connectfour(board):
        return []

    moves = []

    for i in range(7):
        if board.is_column_full(i):
            continue
        moves.append(board.add_piece(i))
    return moves

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            if is_current_player_maximizer:
                return -1000
            else:
                return 1000
    return 0

def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    modifier = board.count_pieces() * 4
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            if is_current_player_maximizer:
                return -2000 + modifier
            else:
                return 2000 - modifier
    return 0
def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    if is_current_player_maximizer:
        chains_cur = board.get_all_chains(True)
        chains_oth = board.get_all_chains(False)

        cur_score, oth_score = len(chains_cur), len(chains_oth)
        for cur_chain in chains_cur:
            if len(cur_chain) >= 3:
                cur_score += 100
            elif len(cur_chain) == 2:
                cur_score += 50
            elif len(cur_chain) == 1:
                cur_score += 10

        for cur_chain in chains_oth:
            if len(cur_chain) >= 3:
                oth_score += 100
            elif len(cur_chain) == 2:
                oth_score += 50
            elif len(cur_chain) == 1:
                oth_score += 10

        modifier = cur_score - oth_score 

        return modifier
    else:
        chains_cur = board.get_all_chains(False)
        chains_oth = board.get_all_chains(True)

        cur_score, oth_score = len(chains_cur), len(chains_oth)
        for cur_chain in chains_cur:
            if len(cur_chain) >= 3:
                cur_score += 100
            elif len(cur_chain) == 2:
                cur_score += 50
            elif len(cur_chain) == 1:
                cur_score += 10

        for cur_chain in chains_oth:
            if len(cur_chain) >= 3:
                oth_score += 100
            elif len(cur_chain) == 2:
                oth_score += 50
            elif len(cur_chain) == 1:
                oth_score += 10

        modifier = cur_score - oth_score 

        return  modifier 


# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    
    q = [[state]]
    best_path,best_score = [],0
    evals = 0
    while q:
        currPath = q.pop()
        currState = currPath[-1]
        if currState.is_game_over():
            evals += 1
            if currState.get_endgame_score() > best_score:
                best_path = currPath
                best_score = currState.get_endgame_score()

        for choice in currState.generate_next_states():
            q.append(currPath+[choice])

    return [best_path,best_score,evals]


# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    best_path,best_score = [], float('-inf') if maximize else float('inf')

    evals = 0

    currPath = [state]
    currState = currPath[-1]

    if currState.is_game_over():
        evals += 1
        best_score = currState.get_endgame_score(maximize)
        best_path = currPath
    for choice in currState.generate_next_states():
        (nx_path,nx_score,nx_evals) = minimax_endgame_search(choice,not maximize)
        evals += nx_evals
        if (maximize and nx_score > best_score) or (not maximize and nx_score < best_score):
            best_score = nx_score
            best_path = currPath + nx_path

    return (best_path,best_score,evals)

# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    best_path,best_score = [], float('-inf') if maximize else float('inf')

    evals = 0

    currPath = [state]
    currState = currPath[-1]

    if currState.is_game_over():
        evals += 1
        best_score = currState.get_endgame_score(maximize)
        best_path = currPath
    elif depth_limit == 0:
        evals += 1
        best_score = heuristic_fn(state.get_snapshot(),maximize)
        best_path = currPath
    else:
        for choice in currState.generate_next_states():
            (nx_path,nx_score,nx_evals) = minimax_search(choice,maximize = not maximize, depth_limit = depth_limit - 1,heuristic_fn=heuristic_fn)
            evals += nx_evals
            if (maximize and nx_score > best_score) or (not maximize and nx_score < best_score):
                best_score = nx_score
                best_path = currPath + nx_path

    return (best_path,best_score,evals)



# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    best_path,best_score = [], float('-inf') if maximize else float('inf')

    evals = 0

    currPath = [state]
    currState = currPath[-1]

    if currState.is_game_over():
        evals += 1
        best_score = currState.get_endgame_score(maximize)
        best_path = currPath
    elif depth_limit == 0:
        evals += 1
        best_score = heuristic_fn(state.get_snapshot(),maximize)
        best_path = currPath
    else:
        for choice in currState.generate_next_states():
            (nx_path,nx_score,nx_evals) = minimax_search_alphabeta(choice,alpha=alpha,beta=beta,maximize = not maximize, depth_limit = depth_limit - 1,heuristic_fn=heuristic_fn)
            evals += nx_evals
            if (maximize and nx_score > best_score) or (not maximize and nx_score < best_score):
                best_score = nx_score
                best_path = currPath + nx_path
            if maximize:
                if best_score > alpha:
                    alpha = best_score
            else:
                if best_score < beta:
                    beta = best_score
            if alpha >= beta:
                return (best_path,best_score,evals)

    return (best_path,best_score,evals)



# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    raise NotImplementedError


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = ''

ANSWER_2 = ''

ANSWER_3 = ''

ANSWER_4 = ''


#### SURVEY ###################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
