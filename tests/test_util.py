from sklearn.metrics import confusion_matrix

from thesis import util


def test_util_print_confusion_matrix():
    cm = confusion_matrix(range(23), range(23))
    util.print_confusion_matrix(cm, range(23))
