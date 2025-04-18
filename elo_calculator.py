import json
from typing import Any
from board import Board
from tree_node import TreeNode
from game_node import GameNode
from bot import Bot, RandomBot, SupervisedLearningBot, MonteCarloBot
from network import NeuralNet, load_model
from random import randint
import random



class Elo_calculator:
    """
    The Elo caluclator class will be a class that manages the playing
    of agents against each other and the counting of their elo
    """

    def __init__(self, game: Board, max_elo = 200, prev_elo_data =  None):
        self.game = game
        #Dict storing key information needed to track and count a player's elo (elo, strategy), with some arb string 'name' as the key
        self.players = {}
        self.max_gain = max_elo

        #Load the data saved from the previous elo counter
        if(prev_elo_data):
            self.load(prev_elo_data)
    
    def register_bot(self, name: str, bot: Bot):
        """
        This function will take in some strategy and name for that strategy, and create a 'player' 
        entry for it so we can track and store its elo
        """

        if name in self.players.keys():
            print(f"Error. Name {name} is registered under this elo tracker")
            return 
        
        player = [100, bot]
        self.players[name] = player

    def save(self, file_path):
        """
        Save the current instance to a pickle file.
        """

        elo_data = {}
        for key, value in self.players.items():
            elo_data[key] = value[0]

        with open(file_path, 'w') as file:
            json.dump(elo_data, file)
        print(f"Data saved to {file_path}")

    #TODO: Write with json instead
    def load(self, file_path):
        """
        Load data from a pickle file and set it as attributes.
        """
        
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            if isinstance(data, dict):
                self.__dict__.update(data)
                print(f"Data loaded from {file_path}")
            else:
                raise ValueError("The pickle file does not contain a valid dictionary.")
        except Exception as e:
            print(f"Failed to load data: {e}")

    def play_match(self, name1: str, name2: str) -> str:
        """
        This function will take in the names of each player, and run a match between them, 
        returning the name of the winner, and updating each player's elo score
        """

        if name1 not in self.players.keys() or name2 not in self.players.keys():
            print("Error, not a registered player name")
            return ''
        
        #randomly select which player goes first
        if randint(1,2) % 2 == 1:
            name1, name2 = name2, name1 

        player1_elo = self.players[name1][0]
        player2_elo = self.players[name2][0]

        player1 = self.players[name1][1]
        player2 = self.players[name2][1]

        #Use the game node class to simulate the running of these 2 agents till the end of the game
        while(not self.game.is_terminal()):
            move = (0,0)
            if self.game.move == 1:
                move = player1.choose_move(game = self.game)
            else:
                move = player2.choose_move(game = self.game)

            self.game = self.game.create_child(move)

        result = self.game.compute_winner()

        p1_new_elo, p2_new_elo = self._calc_elo(result=result, p1_elo=player1_elo, p2_elo=player2_elo)


        #Elo isn't defined under 0, so we do this
        if p1_new_elo > 0:
            self.players[name1][0] = p1_new_elo
        else:
            self.players[name1][0] = 50


        if p2_new_elo > 0:
            self.players[name2][0] = p2_new_elo
        else:
            self.players[name2][0] = 50


        #reset the game to its starting state,
        #the idea behind this is that the other 2 bots hold a reference to the self.game instance managed by the elo calculator
        self.game = TreeNode(GameNode(9))
        # player1.reset_tree(self.game)
        # player2.reset_tree(self.game)

        return name1 if result == 1 else name2
            
    def _calc_elo(self, result: int, p1_elo: float, p2_elo: float) -> tuple[float, float]:
        """
        This function will take in the result and elo of two players and calculate their updated scores
        (Assuming that result = 1 means p1 won and result = -1 means p2 won)
        """

        p1_expected = 1/(1 + 10**((p2_elo - p1_elo)/400.0))
        p2_expected = 1/(1 + 10**((p1_elo - p2_elo)/400.0))

        print(f"p1 expected {p1_expected}")
        print(f"p2 expected {p2_expected}")

        if(result == 1):
            p1_elo = p1_elo + self.max_gain*(result - p1_expected)
            p2_elo = p1_elo + self.max_gain*(-1*result - p2_expected)
        else:
            p1_elo = p1_elo + self.max_gain*(-1*result - p1_expected)
            p2_elo = p1_elo + self.max_gain*(result - p2_expected)

        print(p1_elo, p2_elo)

        return p1_elo, p2_elo
    


if __name__ == "__main__":
    
    # go = Board(9)
    go = TreeNode(GameNode(9))

    elo = Elo_calculator(game=go, prev_elo_data= "elo_data.json")
    print(elo.__dict__)

    random_player = RandomBot()

    sl_model = load_model("./model_weights/SL_weights.pt")
    sl_notree = SupervisedLearningBot(model = sl_model)
    sl_tree = MonteCarloBot(model = sl_model)

    rl_model = load_model("./model_weights/Great_Lakes_Weights.pt")
    rl_notree = SupervisedLearningBot(model=rl_model)
    rl_tree = MonteCarloBot(model = rl_model)

    elo.register_bot(name = "Random_Player", bot= random_player)
    elo.register_bot(name = "Supervised_Learning_No_Tree", bot= sl_notree)
    elo.register_bot(name = "Supervised_Learning_Tree", bot= sl_tree)
    elo.register_bot(name = "Reinforcement_Learning_No_Tree", bot= rl_notree)
    elo.register_bot(name = "Reinforcement_Learning_Tree", bot= rl_tree)

    players = ["Random_Player", "Supervised_Learning_No_Tree", "Supervised_Learning_Tree", "Reinforcement_Learning_No_Tree", "Reinforcement_Learning_Tree"]

    for i in range(1_000):
        if i % 50 == 0:
            elo.save("elo_data.json")
        player_list = random.choices(players, k = 2)
        print(elo.play_match(player_list[0],player_list[1]))

   
    elo.save("elo_data.json")
