from config import *
from reinforcement_learning import create_dataset, update_dataset, train_one_epoch
from bot import MonteCarloBot
from network import NeuralNet, save_model, load_model


def main():
    print_init()

    print("Loading model ... ", end="")
    model = NeuralNet()
    print("Done")

    print("Loading bot ... ", end="")
    bot = MonteCarloBot(model=model)
    print("Done", end="\n\n")
    
    ds = create_dataset(bot, tqdm_desc="Building initial dataset")
    print("")

    for i in range(EPOCHS):
        print(f"Epoch {i+1}")
        print(f"  Training Loss: {train_one_epoch(ds, bot)}")

        print("  ", end="")
        save_model(model, "./Great_Lakes_Weights.pt")

        try:
            update_dataset(ds, bot, tqdm_desc="  Rebuilding dataset: ")
        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except:
            print("  Failed")

        print("  ", end="")
        ds.save("./Great_Lakes_DS.pt")


def print_init():
    print("="*36)
    print(" "*8 + "Mini-AlphaGo Trainer" + " "*8)
    print("-"*36)
    print(f"""DEVICE: {DEVICE}
INPUT_CHANNELS: {INPUT_CHANNELS}
OUTPUT_CHANNELS: {OUTPUT_CHANNELS}
KERNEL: {KERNEL}
NUM_RESIDUALS: {NUM_RESIDUALS}
C_PUCT: {C_PUCT}
SEARCHES_PER_MOVE: {SEARCHES_PER_MOVE}
LEARNING_RATE: {LEARNING_RATE}
EPOCHS: {EPOCHS}
BATCH_SIZE: {BATCH_SIZE}
SELF_PLAY_KEEP_PROB: {SELF_PLAY_KEEP_PROB}
NEW_GAMES_PER_DS: {NEW_GAMES_PER_DS}
RL_DS_SIZE: {RL_DS_SIZE}
NUM_MOVES_ALLOW_PASS: {NUM_MOVES_ALLOW_PASS}
NUM_MOVES_MAX_TEMPERATURE: {NUM_MOVES_MAX_TEMPERATURE}
MAX_MOVES: {MAX_MOVES}""")
    print("="*36, end="\n\n")


if __name__ == "__main__":
    main()

