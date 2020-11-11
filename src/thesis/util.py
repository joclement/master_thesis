import matplotlib.pyplot as plt


def finish_plot(name: str, output_folder, show: bool):
    if output_folder:
        plt.savefig(f"{output_folder}/{name}.png")
    if show:
        plt.show()
