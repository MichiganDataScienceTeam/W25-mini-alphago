import json
from random import randint
from itertools import combinations

from typing import Tuple

from tree_node import TreeNode
from game_node import GameNode
from bot import Bot, RandomBot, SupervisedLearningBot, MonteCarloBot
from network import load_model


class Elo_calculator:
    """
    Manages games between players and tracks elo ratings

    Args:
        min_elo: global minimum elo for players
        max_gain: aka the "K-factor" for elo calculations
        default_elo: the initial elo
        prev_elo_data: path to elo json (overwrites other params)
    """


    def __init__(self, min_elo: int = 50, max_gain: int = 250, default_elo: int = 1000, prev_elo_data: str = None):
        self.game = TreeNode(GameNode(9))
        
        # Tracker player info and elo: Dict[name: str, List[elo: float, strategy: Bot]]
        self.players = {}
        self.min_elo = min_elo
        self.max_gain = max_gain
        self.default_elo = default_elo

        # Load the data saved from the previous elo counter
        if prev_elo_data is not None:
            self.load(prev_elo_data)


    def register_bot(self, name: str, bot: Bot):
        """
        This function will take in some strategy and name for that strategy, and create a 'player' 
        entry for it so we can track and store its elo

        Args:
            name: the player's name
            bot: the Bot object that plays
        """

        if name in self.players.keys():
            raise ValueError(f"Name {name} not found")
        
        self.players[name] = [self.default_elo, bot]


    def save(self, file_path: str):
        """
        Save the current instance to a json file

        Args:
            file_path: path to json file to save to
        """

        elo_data = {}
        for key, value in self.players.items():
            elo_data[key] = value[0]

        with open(file_path, 'w') as file:
            json.dump(elo_data, file)
        
        print(f"Data saved to {file_path}")


    def load(self, file_path):
        """
        Load data from a json file
        """
        
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        if isinstance(data, dict):
            self.__dict__.update(data)
            print(f"Data loaded from {file_path}")
        else:
            raise ValueError("The json file does not contain a valid dictionary.")


    def play_game(self, name1: str, name2: str) -> str:
        """
        Plays a game between 2 players, updates their elos,
        and returns the winner's name

        The selection of white/black is determined randomly

        Args:
            name1: the first player's name
            name2: the second player's name
        """

        # Check players are valid
        if name1 not in self.players.keys():
            raise ValueError(f"Invalid player name {name1}")
        elif name2 not in self.players.keys():
            raise ValueError(f"Invalid player name {name2}")
        
        # Initialize main GameNode
        game = GameNode(9)

        # Randomly select which player goes first
        if randint(1,2) % 2 == 1:
            name1, name2 = name2, name1

        player1_elo = self.players[name1][0]
        player2_elo = self.players[name2][0]

        bot1 = self.players[name1][1]
        bot2 = self.players[name2][1]

        bot1.reset()
        bot2.reset()

        # Run a game until end
        while not game.is_terminal():
            move = None

            if game.move % 2 == 0:
                move = bot1.choose_move()
            else:
                move = bot2.choose_move()

            bot1.register_move(move)
            bot2.register_move(move)

            game = game.create_child(move)

        # Calculate ratings
        result = game.compute_winner()

        p1_new_elo, p2_new_elo = self.calc_elo(
            result = result,
            p1_elo = player1_elo,
            p2_elo = player2_elo
        )

        # Enforce min elo
        self.players[name1][0] = max(self.min_elo, p1_new_elo)
        self.players[name2][0] = max(self.min_elo, p2_new_elo)
        
        return name1 if result == 1 else name2


    def calc_elo(self, result: int, p1_elo: float, p2_elo: float) -> Tuple[float, float]:
        """
        Calculates the new elos of a game between 2 players
        and returns a tuple of the new ratings

        Args:
            result: 1 if player 1 won, -1 if player 2 won
            p1_elo: Black's elo rating
            p2_elo: White's elo rating
        """

        p1_expected = 1/(1 + 10**((p2_elo - p1_elo)/400.0))
        p2_expected = 1/(1 + 10**((p1_elo - p2_elo)/400.0))

        result = (result + 1)/2 # Map {-1, 1} -> {0, 1}

        p1_elo += self.max_gain * (result - p1_expected)
        p2_elo += self.max_gain * (1 - result - p2_expected)

        return p1_elo, p2_elo


if __name__ == "__main__":
    # Init calculator
    elo = Elo_calculator(100, 100)

    # Init string
    print("="*36)
    print(" "*11 + "Elo Calculator" + " "*11)
    print("-"*36)
    print(f"""min_elo: {elo.min_elo}
max_gain: {elo.max_gain}
default_elo: {elo.default_elo}""")
    print("="*36)

    # Load models
    sl_model = load_model("SL_weights.pt")
    rl_model = load_model("RL_weights.pt")
    
    # Init bots
    random_player = RandomBot()

    sl_notree = SupervisedLearningBot(model = sl_model)
    sl_tree = MonteCarloBot(model = sl_model, device = "cpu", always_allow_pass = True)

    rl_notree = SupervisedLearningBot(model = rl_model)
    rl_tree = MonteCarloBot(model = rl_model, device = "cpu", always_allow_pass = True)

    # Register bots
    elo.register_bot(name = "Random", bot = random_player)
    elo.register_bot(name = "RL_No_Tree", bot = rl_notree)
    elo.register_bot(name = "RL_Tree", bot = rl_tree)
    elo.register_bot(name = "SL_No_Tree", bot = sl_notree)
    elo.register_bot(name = "SL_Tree", bot = sl_tree)

    # Play tournament
    players = elo.players.keys()

    for _ in range(5):
        for player1, player2 in combinations(players, 2):
            print(f"Playing: {player1} vs {player2}")
            print(f"  Winner: {elo.play_game(player1, player2)}")

        print("-"*14 + " saving " + "-"*14)
        elo.save("elo_data_log.json")
        print("-"*32)
    
    elo.save("elo_data.json")

