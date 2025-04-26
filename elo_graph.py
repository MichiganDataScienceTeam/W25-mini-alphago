import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json

def create_graph(data_path = "./elo_data.json", filename_path = "Elo_data.png"):
    # with open(data_path, "r") as file:
    #     data = json.load(file)
    # Provided data

    data = {
        "Random_Player": 50,
        "Supervised_Learning_No_Tree": 360.97881517090735,
        "Supervised_Learning_Tree": 50,
        "Reinforcement_Learning_No_Tree": 245.25506595336094,
        "Reinforcement_Learning_Tree": 467.75690596722126
    }

    colors = {
        "Random_Player": "cornflowerblue",
        "Supervised_Learning_No_Tree": "royalblue",
        "Supervised_Learning_Tree": "slateblue",
        "Reinforcement_Learning_No_Tree": "mediumorchid",
        "Reinforcement_Learning_Tree": "darkorchid"
    }

    # Extract labels and values
    labels = list(data.keys())
    values = list(data.values())
    bar_colors = [colors[label] for label in labels]

    label_title = ["Random Player", "SL No Tree", "SL Tree", "RL No Tree", "RL Tree"]
    

    plt.figure(figsize=(10, 6))
    plt.axhline(y=50, color='Black', linestyle='dotted', linewidth=2, label='Baseline (50)')

    plt.text(
    x=-1.225,        # x-position just to the left of the first bar
    y=51,          # a bit above the line for clarity
    s="Floor (50)", 
    color='Black',
    fontsize=10,
    va='bottom'
    )

    plt.bar(label_title, values, color=bar_colors)

    plt.xticks(rotation=30, ha='right', fontsize = 14)


    # Custom legend
    # legend_elements = [Patch(facecolor=colors[label], label=label) for label in labels]
    # plt.legend(handles=legend_elements, title="Model Type")

    plt.ylabel("Elo Rating", fontsize = 16)

    # plt.title("Elo Score of different models")

    plt.tight_layout()
    plt.savefig(filename_path)


if __name__ == "__main__":
    create_graph()