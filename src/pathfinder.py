'''
CMSI 2130 - Homework 1
Author: Colin Bajo-Smith

Modify only this file as part of your submission, as it will contain all of the logic
necessary for implementing the A* pathfinder that solves the target practice problem.
'''
from queue import PriorityQueue
from maze_problem import MazeProblem
from dataclasses import *
from typing import *

@dataclass
class SearchTreeNode:
    """
    SearchTreeNodes contain the following attributes to be used in generation of
    the Search tree:

    Attributes:
        player_loc (tuple[int, int]):
            The player's location within the node.
        action (str):
            The action taken to either move in a direction or shoot.
        parent (Optional[SearchTreeNode]):
            The parent node that the node was generated from (None if the root).
        cost (int):
            The initial cost of the action when taken.
        targets_hit (set[tuple[int, int]]):
            The set of targets hit by the player (inherits targets hit from parent).
        remaining_targets (set[tuple[int, int]]):
            The set of remaining targets to hit.
    """
    player_loc: tuple[(int, int)] 
    action: str
    parent: Optional["SearchTreeNode"]
    cost: int
    targets_hit: set[tuple[(int, int)]]
    remaining_targets: set[tuple[(int, int)]]
    
    def __lt__(self, other: "SearchTreeNode") -> bool:
        return self.cost < other.cost
    
    def __hash__(self) -> int:
        return hash((self.player_loc, self.action, *set(self.remaining_targets)))
    
    def __eq__(self, other) -> bool:
        return self.player_loc == other.player_loc and self.action == other.action and self.remaining_targets == other.remaining_targets
        
    
def pathfind(problem: "MazeProblem") -> Optional[list[str]]:
    """
    The main workhorse method of the package that performs A* graph search to find the optimal
    sequence of actions that takes the agent from its initial state and shoots all targets in
    the given MazeProblem's maze, or determines that the problem is unsolvable.

    Parameters:
        problem (MazeProblem):
            The MazeProblem object constructed on the maze that is to be solved or determined
            unsolvable by this method.

    Returns:
        Optional[list[str]]:
            A solution to the problem: a sequence of actions leading from the 
            initial state to the goal (a maze with all targets destroyed). If no such solution is
            possible, returns None.
    """
    # Create frontier and graveyard
    frontier: PriorityQueue[Tuple[int, "SearchTreeNode"]] = PriorityQueue() 
    graveyard: set["SearchTreeNode"] = set()

    # Create initial values for the root parent node, then create root
    initial_state_loc = problem.get_initial_loc()
    targets_left = problem.get_initial_targets()
    initial_cost = 0
    initial_targets_hit: set[tuple[int, int]] = set()
    
    initial_state_node = (0, SearchTreeNode(initial_state_loc, "", None, initial_cost, initial_targets_hit, targets_left)) 
    
    # Add the initial parent to the frontier
    frontier.put(initial_state_node)
    
    # If no targets in existence, end game
    if len(targets_left) == 0:
        return None
    else:
        # Process nodes in the frontier until it's empty or solution is found
        while not frontier.empty(): 
            # Retrieve the node with the lowest cost
            _, parent_node = frontier.get()

            if parent_node in graveyard:
                continue
            
            # Check if all targets are hit
            num_targets_left = len(parent_node.remaining_targets)
            if num_targets_left <= 0:
                return return_solution(parent_node)
            
            # Generate children for the current node
            node_dictionary = problem.get_transitions(parent_node.player_loc, parent_node.remaining_targets)
            create_children(parent_node, node_dictionary, problem, frontier)
            
            # Add the current node to the graveyard
            graveyard.add(parent_node)
                
        # If no solution is found, return None
        return None   

def create_children(parent_node: "SearchTreeNode", 
                 node_dictionary: dict, 
                 problem: Any, frontier: PriorityQueue) -> Any:
    """
    Generates child nodes from the parent node based on available transitions and adds them
    to the frontier.

    Args:
        parent_node (SearchTreeNode): The current node being expanded.
        node_dictionary (dict): Dictionary of children
        problem (MazeProblem): The maze problem instance used for target visibility and transitions.
        frontier (PriorityQueue): The frontier priority queue used to store nodes for expansion.
    """
    for action, information in node_dictionary.items():
        # Import the targets hit from the parent node and update cost and location
        targets_hit: set[tuple[int, int]] = parent_node.targets_hit
        new_loc = information["next_loc"]
        new_cost = parent_node.cost + information["cost"]
        
        # Get visible targets from current location and, if action is shoot, remove 
        # all visible targets and add to targets hit
        visible_targets = problem.get_visible_targets_from_loc(new_loc, parent_node.remaining_targets)
        remaining_targets_copy = parent_node.remaining_targets.copy()
        if action == 'S':
            for target in visible_targets:
                targets_hit.add(target)
            for target in targets_hit:
                if target in remaining_targets_copy:
                    remaining_targets_copy.remove(target)
        
        # Update targets hit and remaining for child node creation
        new_targets_hit = targets_hit.copy()
        new_targets_left = remaining_targets_copy
        
        # Create the child node and add it to the frontier
        child_node = SearchTreeNode(new_loc, action, parent_node, new_cost, new_targets_hit, new_targets_left)
        
        # Adds the node to the frontier with initial cost combined with heuristic cost and giving priority to more targets hit
        frontier.put((child_node.cost + (len(child_node.remaining_targets) + heuristic(child_node)), child_node))


def heuristic(node: "SearchTreeNode") -> int:
    """
    A heuristic function for A* search that estimates the cost of reaching
    remaining targets from the current node.

    Args:
        node (SearchTreeNode): The current search tree node.

    Returns:
        float: The estimated heuristic cost.
    """
    if len(node.targets_hit) <= 0:
        return 0

    # Get the current player's location
    player_loc = node.player_loc
    min_cost = float('inf')

    # Find the minimum Manhattan distance from the player to any target hit, taking the closest x or y cost
    for target in node.targets_hit:
        cost = min(abs(target[0] - player_loc[0]), abs(target[1] - player_loc[1]))
        if cost < min_cost:
            min_cost = cost

    return int(min_cost)


def return_solution(node: "SearchTreeNode") -> list[str]:
    """
    Reconstructs the solution path from the goal node back to the initial node.

    Args:
        node (SearchTreeNode): The node used to form the solution.

    Returns:
        list[str]: The list of actions detailing the most efficient and optimal way to shoot all targets.
    """
    solution_path: list[str] = []
    current_node = node

    # Backtrack from the goal node to the root node, collecting actions
    while current_node.parent is not None:
        solution_path.append(current_node.action)
        current_node = current_node.parent
    
    return solution_path[::-1]

# ===================================================
# >>> [MT] Summary
# A great submission that shows strong command of
# programming fundamentals, generally good style,
# and a good grasp on the problem and supporting
# theory of A*. Indeed, there is definitely
# a lot to like in what you have above, but
# I think you could have tested it a little more just
# to round out the several edge cases that evaded your
# detection. Give yourself more time to test + debug
# future submissions and you'll be golden!
# ---------------------------------------------------
# >>> [MT] Style Checklist
# [X] = Good, [~] = Mixed bag, [ ] = Needs improvement
#
# [X] Variables and helper methods named and used well
# [X] Proper and consistent indentation and spacing
# [X] Proper docstrings provided for ALL methods
# [X] Logic is adequately simplified
# [X] Code repetition is kept to a minimum
# ---------------------------------------------------
# Correctness:          92 / 100 (-2 / missed unit test)
# Mypy Penalty:        -2 (-2 if mypy wasn't clean)
# Style Penalty:       -0
# Total:                90 / 100
# ===================================================

