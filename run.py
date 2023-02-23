from enum import Enum
import typer

app = typer.Typer()


class Dataset(str, Enum):
    cifar10 = "cifar10"
    mnist = "mnist"


@app.command(
    "visualize",
    help="Visualize dataset of your choice with the options that you specify, like count, transformations etc.",
    short_help="Visualize dataset",
)
def visualize_dataset(
    dataset: Dataset = typer.Option(
        default=Dataset.cifar10,
        help="Dataset name",
    ),
    count: int = typer.Option(
        help="Number of images to visualize",
        default=10,
    ),
):
    print(f"Visualizing {count} images of {dataset} dataset")


@app.command()
def delete():
    print("Deleting user: Hiro Hamada")


if __name__ == "__main__":
    app()
