"""
Usage:
1. Verify flask is installed (run "pip install -r requirements.txt" if not)
2. Run "flask --app web_vis run" in a terminal
"""

# Imports
from flask import Flask, request, jsonify
import torch

from game_node import GameNode
from network import NeuralNet, load_model

from data_preprocess import node_to_tensor

from tree_node import TreeNode
from monte_carlo import MonteCarlo

# Model setup
MODEL_PATH = "RL_weights.pt" # Update this as needed
BOT_SEARCHES = 256 # Visits per position before the bot replies

model = NeuralNet()

try:
    model = load_model(MODEL_PATH)
except:
    print(f"Failed to load model at {MODEL_PATH}")

    res = ""

    while res not in list("yn"):
        res = input("Load random model (y/n)? ")
        res = res.lower()
    
    if res == "n":
        print("Program exited early: cannot run without model")
        exit(1)

# GameNode setup
SIZE = 9
temp_node = GameNode(SIZE)

# MCTS setup
tree = MonteCarlo(
    model,
    TreeNode(temp_node),
    device = "cpu"
)

def search():
    global tree

    # Reuse the existing subtree and search until the position has BOT_SEARCHES visits
    while tree.curr.num_visits < BOT_SEARCHES:
        tree.search()

search()

# Game node utils
invert = lambda s: s.replace("○", "B").replace("●", "W").replace("W", "○").replace("B", "●")

def small_string(node: GameNode):
    global SIZE
    return "\n".join([invert(s.replace(" ", "")[-SIZE:]) for s in str(node).split("\n")[3:]])

# Flask things (assumes model behaves well)
app = Flask(__name__, static_folder="web_vis")

@app.route("/")
def main():
    return app.send_static_file("index.html")

@app.route("/play_move", methods=["POST"])
def play_move():
    global tree, SIZE

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    if "row" not in data or "col" not in data:
        return jsonify({"error": "JSON data missing row and/or col fields"}), 400
    
    if (data["row"], data["col"]) != (-1, -1) and (not (0 <= data["row"] < SIZE) or not (0 <= data["col"] < SIZE)):
        return jsonify({"error": f"Specified location {data['row'], data['col']} is out of bounds"}), 400

    if not tree.curr.is_valid_move(data["row"], data["col"]):
        return jsonify({"error": f"Specified location {data['row'], data['col']} is an invalid move"}), 400

    # New version should do a linear search to find the correct child
    for child in tree.curr.nexts:
        if child.prev_move == (data["row"], data["col"]):
            tree.curr = child
            break
    else:
        # Pass isn't in the tree before NUM_MOVES_ALLOW_PASS, but the player can still pass
        child = tree.curr.create_child((data["row"], data["col"]))
        tree.curr.nexts.append(child)
        tree.curr = child

    # The bot searches in /bot_move so the player's move shows right away
    return "Good", 200

@app.route("/bot_move", methods=["POST"])
def bot_move():
    global tree

    if tree.curr.is_terminal():
        return "Game over", 200

    search()

    # Play the most visited move (no randomness against a human)
    tree.curr = max(tree.curr.nexts, key=lambda child: child.num_visits)

    # Search the player's position too so the policy and value show the search result
    search()

    return "Good", 200

@app.route("/get_board", methods=["POST"])
def get_board():
    global tree

    return small_string(tree.curr.gamenode_str()), 200

@app.route("/reset", methods=["POST"])
def reset():
    global tree, SIZE

    tree.curr = TreeNode(GameNode(SIZE))

    search()

    return "Good", 200

@app.route("/undo", methods=["POST"])
def undo():
    global tree
    
    if tree.curr.prev is None:
        return jsonify({"error": "No move to undo"}), 400

    # Undo the bot's reply and the player's move so it's the player's turn again
    tree.curr = tree.curr.prev

    if tree.curr.prev is not None and tree.curr.move % 2 == 1:
        tree.curr = tree.curr.prev

    return "Good", 200

@app.route("/network", methods=["POST"])
def network():
    global tree

    policy, val = model(node_to_tensor(tree.curr).unsqueeze(0))

    policy = policy.flatten().detach() # Already softmaxed by the PolicyHead
    val = val.detach().item()

    # Show the search result instead of the network once the position has been searched
    if sum(child.num_visits for child in tree.curr.nexts) > 0:
        policy = torch.zeros_like(policy)

        for child in tree.curr.nexts:
            row, col = child.prev_move
            policy[-1 if (row, col) == (-1, -1) else row * SIZE + col] = child.num_visits

        val = tree.curr.Q_value()

    policy /= policy.max()
    policy = policy / 5

    policy *= torch.tensor(tree.curr.available_moves_mask())

    return jsonify({
        "policy": policy.tolist(),
        "value": val
    }), 200

@app.route("/get_tree", methods=["POST"])
def get_tree():
    global tree

    stringify = lambda node: invert("\n".join(node.gamenode_str().split("\n")[2:]))

    q = [(-1, tree.curr)]
    out = []

    while len(q) != 0:
        nq = []
        for prev_i, node in [(p, r) for p, r in q if r.num_visits > 0]:
            out.append({
                "prev": prev_i,
                "val": stringify(node),
                "tooltip": f"""Visits: {node.num_visits}
Q: {node.Q_value():.3f}
U: {node.u_value():.3f}"""
            })
            nq += [(len(out) - 1, s) for s in node.nexts]
        
        q = nq

    return jsonify(out), 200
