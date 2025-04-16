from config import *
from reinforcement_learning import train_one_epoch
from mp_self_play import create_dataset, update_dataset
from bot import MonteCarloBot
from network import NeuralNet, save_model, load_model


def main():
    print_init()

    MODEL_PATH = "./Great_Lakes_Weights.pt"
    DS_PATH = "./Great_Lakes_DS.pt"

    print("Loading model ... ", end="")
    model = NeuralNet()
    print("Done")

    save_model(model, MODEL_PATH)

    print("Loading bot ... ", end="")
    bot = MonteCarloBot(model=model)
    print("Done", end="\n\n")
    
    print("Building initial dataset...")
    ds = create_dataset(MODEL_PATH, PROCESSES, RL_DS_SIZE//PROCESSES, True)

    for i in range(EPOCHS):
        print(f"Epoch {i+1}")
        print(f"  Training Loss: {train_one_epoch(ds, bot)}")

        print("  ", end="")
        save_model(model, MODEL_PATH)

        try:
            print("Rebuilding dataset:")
            update_dataset(ds, MODEL_PATH, PROCESSES, NEW_GAMES_PER_DS//PROCESSES, True)
        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except:
            print("Failed")

        print("  ", end="")
        ds.save(DS_PATH)


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

