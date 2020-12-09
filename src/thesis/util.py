import matplotlib.pyplot as plt


def finish_plot(name: str, output_folder, show: bool = False):
    if output_folder:
        plt.savefig(f"{output_folder}/{name}.svg")
    if show:
        plt.show()
    plt.close()
