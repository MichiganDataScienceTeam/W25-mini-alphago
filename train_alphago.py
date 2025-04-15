from config import *
from reinforcement_learning import self_play, create_dataset, update_dataset, train_one_epoch
from bot import MonteCarloBot
from network import NeuralNet, save_model, load_model

def main():
    # model = load_model("./Great_Lakes_Weights.pt")
    model = NeuralNet()
    bot = MonteCarloBot(model=model)
    print("Building initial dataset... ", end="")
    ds = create_dataset(bot)
    print("Done")


    for i in range(EPOCHS):
        print(f"Epoch {i+1} - Training Loss: ", end="")
        print(train_one_epoch(ds, bot))
        save_model(model, "./Great_Lakes_Weights.pt")

        print("Rebuilding dataset... ", end="")

        try:
            update_dataset(ds)
            print("Done")
        except:
            print("Failed")

        ds.save("./Great_Lakes_DS.pt")

        
   


if __name__ == "__main__":
    main()