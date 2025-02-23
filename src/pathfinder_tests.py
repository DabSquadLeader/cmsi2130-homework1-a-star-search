from pathfinder import *
import unittest

class PathfinderTests(unittest.TestCase):
    """
    Unit tests for validating the pathfinder's efficacy. Notes:
    - If this is the set of tests provided in the solution skeleton, it represents an
      incomplete set that you are expected to add to to adequately test your submission!
    - Your correctness score on the assignment will be assessed by a more complete,
      grading set of unit tests.
    - A portion of your style grade will also come from proper type hints; remember to
      validate your submission using `mypy .` and ensure that no issues are found.
    """
    
    def run_maze(self, maze: list[str], solution_expected: bool, optimal_cost: int = 0) -> None:
        """
        For a given maze (a list of strings denoting the maze contents), runs your pathfinder algorithm
        and determines whether or not it returns the correct and optimal solution, if one exists.
        
        Attributes:
            maze (list[str]):
                The 2D list of strings denoting the maze layout, including locations of the player's
                initial state, walls, mud tiles, and targets to hit.
            solution_expected (bool):
                Whether or not the maze can be solved. If True, it is expected that your pathfinder will
                yield the correct series of steps needed to shoot all targets in the lowest cost possible.
                If False, it is expected that your pathfinder will return None.
            optimal_cost (int):
                If there is indeed a solution expected, the optimal_cost will indicate the lowest cost
                possible in solving the maze. It is possible to have a solution that may solve mazes but
                not in the optimal way, which will not receive credit.
        """
        problem = MazeProblem(maze)
        solution = pathfind(problem)
        error_suffix = "Test Failure: " + self._testMethodName + "\nGiven Solution: " + str(solution) + "\nMaze:\n" + "\n".join(maze)
        
        if not solution_expected: 
            if not solution is None:
                self.fail("[X] You returned a solution where none was possible on this maze:\n" + error_suffix)
            else: 
                return
        elif solution is None:
            self.fail("[X] You returned an answer of no solution (None) where one was expected on maze:\n" + error_suffix)
        
        result = problem.test_solution(solution)
        
        self.assertTrue(result["is_solution"], "[X] You returned a solution that was incorrect on this maze:\n" + error_suffix)
        self.assertEqual(result["cost"], optimal_cost, "[X] You returned a suboptimal solution on this maze:\n" + error_suffix)
        
    # Tests with solutions
    # ---------------------------------------------------------------------------
    def test_pathfinder_t0(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "XT...X", # 1
            "X....X", # 2
            "X@...X", # 3
            "XXXXXX", # 4
        ]
        
        self.run_maze(maze, True, 2)
        
    def test_pathfinder_t1(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "X.T..X", # 1
            "X....X", # 2
            "X@...X", # 3
            "XXXXXX", # 4
        ]
        
        self.run_maze(maze, True, 3)
        
    def test_pathfinder_t2(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "XT...X", # 1
            "X....X", # 2
            "X@..TX", # 3
            "XXXXXX", # 4
        ]
        
        self.run_maze(maze, True, 2)
    
    def test_pathfinder_t3(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "XT...X", # 1
            "X.XT.X", # 2
            "X@..TX", # 3
            "XXXXXX", # 4
        ]
        
        self.run_maze(maze, True, 6)
        
    def test_pathfinder_t4(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "XT..XX", # 1
            "XX@X.X", # 2
            "XX.X.X", # 3
            "X...TX", # 4
            "XXXXXX", # 5
        ]
        
        self.run_maze(maze, True, 8)
        
    def test_pathfinder_t5(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "XTM.XX", # 1
            "XXMX.X", # 2
            "XX@X.X", # 3
            "X.M.TX", # 4
            "XXXXXX", # 5
        ]
        
        self.run_maze(maze, True, 14)
        
    def test_pathfinder_t6(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "XT..TX", # 1
            "X..T.X", # 2
            "X....X", # 3
            "X@X..X", # 4
            "XXXXXX", # 5
        ]
        
        self.run_maze(maze, True, 7)
        
    def test_pathfinder_t7(self) -> None:
        maze = [
        #    0123456
            "XXXXXXX", # 0
            "XT..T.X", # 1
            "X..X..X", # 2
            "X...X.X", # 3
            "X@X..TX", # 4
            "XXXXXXX", # 5
        ]
        
        self.run_maze(maze, True, 11)
        
    def test_pathfinder_t8(self) -> None:
        maze = [
        #    012345
            "XXXXXX", # 0
            "XT..TX", # 1
            "X..TTX", # 2
            "X...TX", # 3
            "X@X.TX", # 4
            "XXXXXX", # 5
        ]
        
        self.run_maze(maze, True, 9) 
    
    def test_pathfinder_t9(self) -> None:
        maze = [
        #    0123456
            "XXXXXXX", # 0
            "XTTTTTX", # 1
            "XTT@TTX", # 2
            "XTTTTTX", # 3
            "XTTTTTX", # 4
            "XXXXXXX", # 5
        ]
        
        self.run_maze(maze, True, 12) 
        
    def test_pathfinder_t10(self) -> None:
        maze = [
        #    01234567890123456789012345678
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX", # 0
            "X.......X...............X...X", # 1
            "X....T..X...........X.X...T.X", # 2
            "XXMXXXXXX........XXXXXXXXXXXX", # 3
            "X....XXXXXXXX..MMMMMMMM.....X", # 4
            "X...........................X", # 5
            "X.....T.....XXXXXXXXX.......X", # 6
            "X.MM........................X", # 7
            "X.....XXXXXXX.......T.......X", # 8
            "X...........................X", # 9
            "X..MMM.....XXXXXX......XXX..X", # 10
            "X......X...........M.....X..X", # 11
            "X......X.................X..X", # 12
            "X.XXXXXX.....XXXXXXXXX..XXX.X", # 13
            "X...T..X................X...X", # 14
            "X......X......@.........XT..X", # 15
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX", # 16

        ]

        self.run_maze(maze, True, 115)
        
        
    # Tests with NO solutions
    # ---------------------------------------------------------------------------
    def test_pathfinder_nosoln_t0(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "XTX..X", # 1
            "XX...X", # 2
            "X@...X", # 3
            "XXXXXX", # 4
        ]
        
        self.run_maze(maze, False)
        
    def test_pathfinder_nosoln_t1(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "XTX.TX", # 1
            "XX...X", # 2
            "X@...X", # 3
            "XXXXXX", # 4
        ]
        
    def test_pathfinder_nosoln_t2(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "XTX.TX", # 1
            "XX...X", # 2
            "X@X..X", # 3
            "XXXXXX", # 4
        ]
        
        self.run_maze(maze, False)
        
    def test_pathfinder_nosoln_t3(self) -> None:
        maze = [
           # 012345
            "XXXXXX", # 0
            "X.X..X", # 1
            "X....X", # 2
            "X@...X", # 3
            "XXXXXX", # 4
        ]
        
        self.run_maze(maze, False)
        
if __name__ == '__main__':
    unittest.main()