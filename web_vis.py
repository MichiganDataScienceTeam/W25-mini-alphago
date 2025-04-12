"""
Usage:
1. Verify flask is installed (run "pip install -r requirements.txt" if not)
2. Run "flask --app web_vis run" in a terminal
"""

# Imports
from flask import Flask, request, jsonify
import torch

from game_node import GameNode
from network import NeuralNet

from data_preprocess import node_to_tensor

from tree_node import TreeNode
from monte_carlo import MonteCarlo

# Model setup
MODEL_STATE_DICT_PATH = "model.pt" # Update this as needed

model = NeuralNet()

try:
    model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, weights_only=True))
except:
    print(f"Failed to load model at {MODEL_STATE_DICT_PATH}")

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
    TreeNode(temp_node)
)

def search():
    global tree

    for _ in range(10):
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

    tree.move_curr((data["row"], data["col"]))

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

    tree.curr = tree.curr.prev

    return "Good", 200

@app.route("/network", methods=["POST"])
def network():
    global tree

    policy, val = model(node_to_tensor(tree.curr).unsqueeze(0))

    policy = policy.softmax(1).flatten().detach()

    policy /= policy.max()
    policy = policy / 5

    policy *= torch.tensor(tree.curr.available_moves_mask())

    return jsonify({
        "policy": policy.tolist(),
        "value": val.detach().item()
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
                "val": stringify(node)
            })
            nq += [(len(out) - 1, s) for s in node.nexts]
        
        q = nq

    return jsonify(out), 200
