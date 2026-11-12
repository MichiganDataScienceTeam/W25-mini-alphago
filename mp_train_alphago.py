import os

from config import *
from reinforcement_learning import train_one_epoch, make_optimizer, evaluate_loss
from mp_self_play import create_dataset, update_dataset
from bot import MonteCarloBot
from network import NeuralNet, save_model, load_model


def main():
    print_init()

    MODEL_PATH = "./RL_weights.pt"
    DS_PATH = "./RL_ds.pt"
    # Separate folder per job so resuming doesn't overwrite old checkpoints
    CHECKPOINT_PATH = os.path.join("./checkpoints", os.environ.get("SLURM_JOB_ID", "local"))

    if RL_INIT_WEIGHTS is None:
        print("Loading model ... ", end="")
        model = NeuralNet()
        print("Done")
    else:
        model = load_model(RL_INIT_WEIGHTS)

    save_model(model, MODEL_PATH)

    print("Loading bot ... ", end="")
    bot = MonteCarloBot(model=model)
    print("Done", end="\n\n")
    
    print("Building initial dataset...")
    ds = create_dataset(MODEL_PATH, PROCESSES, RL_DS_START_SIZE//PROCESSES, True)

    optimizer = make_optimizer(model)

    for i in range(EPOCHS):
        print(f"Epoch {i+1}")

        # Newest games haven't been trained on yet
        new_positions = ds.positional_data[ds.start_indices[-NEW_GAMES_PER_DS]:]
        loss, value_loss, policy_loss = evaluate_loss(new_positions, model)
        print(f"  New Games Loss: {loss} (value {value_loss}, policy {policy_loss})")

        loss, value_loss, policy_loss = train_one_epoch(ds, bot, optimizer)
        print(f"  Training Loss: {loss} (value {value_loss}, policy {policy_loss})")

        print("  ", end="")
        save_model(model, MODEL_PATH)

        # Keep old weights to roll back to if training breaks
        if (i + 1) % RL_CHECKPOINT_EVERY == 0:
            os.makedirs(CHECKPOINT_PATH, exist_ok=True)
            print("  ", end="")
            save_model(model, os.path.join(CHECKPOINT_PATH, f"RL_weights_{i+1}.pt"))

        print("Rebuilding dataset:")
        update_dataset(ds, MODEL_PATH, PROCESSES, NEW_GAMES_PER_DS//PROCESSES, True)

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
FULL_SEARCHES: {FULL_SEARCHES}
FAST_SEARCHES: {FAST_SEARCHES}
FULL_SEARCH_PROB: {FULL_SEARCH_PROB}
RL_LEARNING_RATE: {RL_LEARNING_RATE}
EPOCHS: {EPOCHS}
BATCH_SIZE: {BATCH_SIZE}
NEW_GAMES_PER_DS: {NEW_GAMES_PER_DS}
RL_DS_START_SIZE: {RL_DS_START_SIZE}
RL_DS_SIZE: {RL_DS_SIZE}
RL_AUGMENT: {RL_AUGMENT}
NUM_MOVES_ALLOW_PASS: {NUM_MOVES_ALLOW_PASS}
NUM_MOVES_MAX_TEMPERATURE: {NUM_MOVES_MAX_TEMPERATURE}
MAX_MOVES: {MAX_MOVES}
DIRICHLET_ALPHA: {DIRICHLET_ALPHA}
DIRICHLET_FRACTION: {DIRICHLET_FRACTION}
RL_INIT_WEIGHTS: {RL_INIT_WEIGHTS}
RL_CHECKPOINT_EVERY: {RL_CHECKPOINT_EVERY}
PROCESSES: {PROCESSES}
INFERENCE_SERVER: {INFERENCE_SERVER}
WORKERS_PER_PROCESS: {WORKERS_PER_PROCESS}""")
    print("="*36, end="\n\n")


if __name__ == "__main__":
    main()

