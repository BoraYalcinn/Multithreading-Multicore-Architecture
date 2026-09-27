# PC40 Lab session 2, A26, J.Gaber, gaber@utbm.fr
#!/usr/bin/env python3
"""
PC40 - MNIST Dataset Viewer

Displays one example of each MNIST digit (0..9).

From the Lab package root:
    python3 data/view_mnist.py

If the script and MNIST files are in the current directory:
    python3 view_mnist.py --data .

To save the visualization:
    python3 data/view_mnist.py --save mnist_examples.png

or, for a local test:
    python3 view_mnist.py --data . --save mnist_examples.png

Requires matplotlib:
    python3 -m pip install matplotlib
"""

import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Read MNIST IDX image file
# ---------------------------------------------------------------------------
def read_images(path):
    """
    Read an MNIST IDX image file.

    Returns:
        n     : number of images
        rows  : number of rows per image
        cols  : number of columns per image
        raw   : raw pixel bytes
    """
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(
                f"{path}: invalid MNIST image IDX file "
                f"(magic number = {magic})"
            )

        raw = f.read(n * rows * cols)

    expected = n * rows * cols

    if len(raw) != expected:
        raise ValueError(
            f"{path}: truncated image file "
            f"({len(raw)} bytes instead of {expected})"
        )

    return n, rows, cols, raw


# ---------------------------------------------------------------------------
# Read MNIST IDX label file
# ---------------------------------------------------------------------------
def read_labels(path):
    """
    Read an MNIST IDX label file.

    Returns:
        labels : bytes object containing labels 0..9
    """
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(
                f"{path}: invalid MNIST label IDX file "
                f"(magic number = {magic})"
            )

        labels = f.read(n)

    if len(labels) != n:
        raise ValueError(
            f"{path}: truncated label file "
            f"({len(labels)} labels instead of {n})"
        )

    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Display one MNIST training example for each digit 0..9."
    )

    parser.add_argument(
        "--data",
        default="data",
        help="directory containing the MNIST IDX files "
             "(default: data)"
    )

    parser.add_argument(
        "--save",
        metavar="FILE",
        help="save the figure, for example: "
             "--save mnist_examples.png"
    )

    args = parser.parse_args()

    data_dir = Path(args.data)

    image_file = data_dir / "train-images-idx3-ubyte"
    label_file = data_dir / "train-labels-idx1-ubyte"

    # -----------------------------------------------------------------------
    # Check that MNIST exists
    # -----------------------------------------------------------------------
    if not image_file.exists() or not label_file.exists():

        raise SystemExit(
            "MNIST files not found.\n\n"
            "From the Lab package root, first run:\n"
            "    python3 data/download_mnist.py\n\n"
            "If the MNIST files are in the current directory, use:\n"
            "    python3 view_mnist.py --data ."
        )

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    n, rows, cols, raw = read_images(image_file)
    labels = read_labels(label_file)

    if len(labels) != n:
        raise SystemExit(
            "The numbers of MNIST images and labels do not match."
        )

    # -----------------------------------------------------------------------
    # Find the first example of each digit 0..9
    # -----------------------------------------------------------------------
    chosen = {}

    for i, label in enumerate(labels):

        if label not in chosen:
            chosen[label] = i

        if len(chosen) == 10:
            break

    missing = [digit for digit in range(10) if digit not in chosen]

    if missing:
        raise SystemExit(
            f"Could not find examples for digits: {missing}"
        )

    # -----------------------------------------------------------------------
    # Display the 10 examples
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(10, 4.8)
    )

    fig.suptitle(
        "MNIST examples - each 28x28 image becomes 784 NN inputs",
        fontsize=13
    )

    image_size = rows * cols

    for digit, ax in zip(range(10), axes.flat):

        index = chosen[digit]

        start = index * image_size

        # IMPORTANT:
        # raw[...] returns a bytes object.
        # Matplotlib needs numerical values, not byte strings.
        # Convert each row explicitly to a list of integers (0..255).
        image = [
            list(
                raw[
                    start + r * cols:
                    start + (r + 1) * cols
                ]
            )
            for r in range(rows)
        ]

        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax.set_title(
            f"Label: {digit}\nSample #{index}"
        )

        ax.axis("off")

    plt.tight_layout(
        rect=(0, 0, 1, 0.92)
    )

    # -----------------------------------------------------------------------
    # Optional save
    # -----------------------------------------------------------------------
    if args.save:

        plt.savefig(
            args.save,
            dpi=160,
            bbox_inches="tight"
        )

        print(
            f"Figure saved to: {args.save}"
        )

    # -----------------------------------------------------------------------
    # Information for the students
    # -----------------------------------------------------------------------
    print(
        f"Loaded {n} MNIST training images "
        f"({rows}x{cols} pixels)."
    )

    print(
        f"Each image provides {rows * cols} "
        f"input values to the neural network."
    )

    print(
        "Displayed one example of each digit from 0 to 9."
    )

    plt.show()


if __name__ == "__main__":
    main()