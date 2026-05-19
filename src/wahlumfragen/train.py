from wahlumfragen.model import Model
from wahlumfragen.data import load_poll_csv


def train():
    polls = load_poll_csv("data/sample.csv")
    model = Model()
    print(f"Loaded {len(polls)} polls for {model.__class__.__name__}.")


if __name__ == "__main__":
    train()
