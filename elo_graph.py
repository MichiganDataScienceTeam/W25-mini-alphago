import matplotlib.pyplot as plt
import json

def create_graph(data_path = "./elo_data.json", filename_path = "Elo_data.png"):
    with open(data_path, "r") as file:
        data = json.load(file)
    
    fig, ax = plt.subplots()
    
    colors = {
        "Random": "cornflowerblue",
        "SL_No_Tree": "royalblue",
        "SL_Tree": "slateblue",
        "RL_No_Tree": "mediumorchid",
        "RL_Tree": "darkorchid"
    }

    # Extract labels and values
    labels = list(data.keys())
    values = list(data.values())
    bar_colors = [colors[label] for label in labels]

    label_title = ["Random", "SL", "SL Tree", "RL", "RL Tree"]

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.bar(label_title, values, color=bar_colors, zorder=2)

    plt.xticks(rotation=0, fontsize = 14)
    plt.yticks(fontsize = 14)

    plt.ylim(0, 1400)

    plt.ylabel("Elo Rating", fontsize = 14)

    ax.grid(axis="y", zorder=1)

    fig.tight_layout()
    plt.figure(figsize=(10, 6))

    fig.savefig(filename_path)


if __name__ == "__main__":
    create_graph()