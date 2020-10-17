from sklearn.metrics import confusion_matrix


# Original taken from here: https://gist.github.com/zachguo/10296432
def print_confusion_matrix(true, pred):
    labels = set(true + pred)
    cm = confusion_matrix(true, pred)
    # find which fixed column width will be used for the matrix
    columnwidth = max([len(str(x)) for x in labels] + [5])

    # top-left cell of the table that indicates that top headers are predicted classes,
    # left headers are true classes
    padding_fst_cell = (columnwidth - 3) // 2  # double-slash is int division
    fst_empty_cell = (
        padding_fst_cell * " " + "t/p" + " " * (columnwidth - padding_fst_cell - 3)
    )

    # Print header
    print("    " + fst_empty_cell, end=" ")
    for label in labels:
        print(
            f"{label:{columnwidth}}", end=" "
        )  # right-aligned label padded with spaces to columnwidth

    print()  # newline
    # Print rows
    for i, label in enumerate(labels):
        print(
            f"    {label:{columnwidth}}", end=" "
        )  # right-aligned label padded with spaces to columnwidth
        for j in range(len(labels)):
            # cell value padded to columnwidth with spaces and displayed with 1 decimal
            cell = f"{cm[i, j]:{columnwidth}.1f}"
            print(cell, end=" ")
        print()
