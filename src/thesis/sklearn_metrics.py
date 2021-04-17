# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from sklearn.utils._encode import _encode, _unique

from . import constants


# Adapted from:
# https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/_ranking.py#L1572
def top_k_accuracy_score(y_true, predictions, labels, k=constants.K):
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_true = column_or_1d(y_true)
    y_score = check_array(predictions, ensure_2d=False)
    check_consistent_length(y_true, y_score)

    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2

    labels = column_or_1d(labels)
    classes = _unique(labels)
    n_labels = len(labels)
    n_classes = len(classes)

    if n_classes != n_labels:
        raise ValueError("Parameter 'labels' must be unique.")

    if not np.array_equal(classes, labels):
        raise ValueError("Parameter 'labels' must be ordered.")

    if n_classes != y_score_n_classes:
        raise ValueError(
            f"Number of given labels ({n_classes}) not equal to the "
            f"number of classes in 'y_score' ({y_score_n_classes})."
        )

    if len(np.setdiff1d(y_true, classes)):
        raise ValueError("'y_true' contains labels not in parameter 'labels'.")

    y_true_encoded = _encode(y_true, uniques=classes)
    sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[:, ::-1]
    hits = (y_true_encoded == sorted_pred[:, :k].T).any(axis=0)

    return np.average(hits)
