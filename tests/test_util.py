from thesis import util


def test_util_print_confusion_matrix():
    true = list(range(23))
    predicted = true
    util.print_confusion_matrix(true, predicted)
