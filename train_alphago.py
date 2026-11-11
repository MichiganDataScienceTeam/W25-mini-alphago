from config import *
from reinforcement_learning import create_dataset, update_dataset, train_one_epoch, make_optimizer
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

    optimizer = make_optimizer(model)

    for i in range(EPOCHS):
        print(f"Epoch {i+1}")
        loss, value_loss, policy_loss = train_one_epoch(ds, bot, optimizer)
        print(f"  Training Loss: {loss} (value {value_loss}, policy {policy_loss})")

        print("  ", end="")
        save_model(model, "./RL_weights.pt")

        update_dataset(ds, bot, tqdm_desc="  Rebuilding dataset: ")

        print("  ", end="")
        ds.save("./RL_ds.pt")


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
RL_LEARNING_RATE: {RL_LEARNING_RATE}
EPOCHS: {EPOCHS}
BATCH_SIZE: {BATCH_SIZE}
SELF_PLAY_KEEP_PROB: {SELF_PLAY_KEEP_PROB}
NEW_GAMES_PER_DS: {NEW_GAMES_PER_DS}
RL_DS_START_SIZE: {RL_DS_START_SIZE}
RL_DS_SIZE: {RL_DS_SIZE}
RL_AUGMENT: {RL_AUGMENT}
NUM_MOVES_ALLOW_PASS: {NUM_MOVES_ALLOW_PASS}
NUM_MOVES_MAX_TEMPERATURE: {NUM_MOVES_MAX_TEMPERATURE}
MAX_MOVES: {MAX_MOVES}""")
    print("="*36, end="\n\n")


if __name__ == "__main__":
    main()

