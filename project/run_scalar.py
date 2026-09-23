import argparse
import json
import random
from pathlib import Path
import minitorch


class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.in_size, self.out_size = in_size, out_size
        self.weights = [[self.add_parameter(f"weight_{i}_{j}",
                         minitorch.Scalar(2 * (random.random() - 0.5)))
                         for j in range(out_size)] for i in range(in_size)]
        self.bias = [self.add_parameter(f"bias_{j}",
                     minitorch.Scalar(2 * (random.random() - 0.5)))
                     for j in range(out_size)]

    def forward(self, inputs):
        return [sum((inputs[i] * self.weights[i][j].value
                     for i in range(self.in_size)), self.bias[j].value)
                for j in range(self.out_size)]


class Network(minitorch.Module):
    def __init__(self, hidden_layers: int) -> None:
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        x = [v.relu() for v in self.layer1(x)]
        x = [v.relu() for v in self.layer2(x)]
        return self.layer3(x)[0].sigmoid()


def default_log(epoch, total_loss, correct, losses):
    print(f"Epoch {epoch:04d} loss={total_loss:.6f} correct={correct}", flush=True)


class ScalarTrain:
    def __init__(self, hidden_layers: int) -> None:
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(tuple(minitorch.Scalar(v, back=None) for v in x))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log):
        self.model = Network(self.hidden_layers)
        optimizer = minitorch.SGD(self.model.parameters(), learning_rate)
        self.losses = []
        self.history = []
        for epoch in range(1, max_epochs + 1):
            optimizer.zero_grad()
            total_loss, correct = 0.0, 0
            for x, y in zip(data.X, data.y):
                out = self.run_one(x)
                probability = out if y == 1 else 1.0 - out
                loss = -probability.log()
                total_loss += loss.data
                correct += int((out.data > 0.5) == bool(y))
                (loss / data.N).backward()
            optimizer.step()
            self.losses.append(total_loss)
            self.history.append({"epoch": epoch, "loss": total_loss, "correct": correct})
            if epoch % 10 == 0 or epoch == 1 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, self.losses)
        return self.history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(minitorch.datasets), default="Simple")
    parser.add_argument("--points", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    random.seed(args.seed)
    data = minitorch.datasets[args.dataset](args.points)
    trainer = ScalarTrain(args.hidden)
    history = trainer.train(data, args.lr, args.epochs)
    predictions = [trainer.run_one(x).data for x in data.X]
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": args.dataset, "seed": args.seed, "points": data.N,
               "hidden": args.hidden, "learning_rate": args.lr, "epochs": args.epochs,
               "history": history, "X": data.X, "y": data.y, "predictions": predictions,
               "accuracy": sum((p > 0.5) == bool(y) for p, y in zip(predictions, data.y)) / data.N,
               "parameters": {k: p.value.data for k, p in trainer.model.named_parameters()}}
    (args.output / f"{args.dataset}.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
