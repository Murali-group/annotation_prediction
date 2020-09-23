

def calc_benjamini_hochberg_corrected_value(p_value, index,
        total_num_tests):
    """
    Perform the k-calculation for Benjamini-Hochberg correction.

    See
    http://en.wikipedia.org/wiki/False_discovery_rate#Independent_tests
    for more detail.

    :Parameters:
    - `p_value`: the uncorrected p-value of a test
    - `index`: where in the total list of test values this value is
      [NOTE: this should be one-index based, not zero-index (e.g.,
      the first element is index `1`)]
    - `total_num_tests`: the total number of tests done

    """
    if index > total_num_tests:
        raise ValueError("index is greater than the total number of "
                "tests")

    bh_corrected_value = p_value * (total_num_tests / float(index))
    if bh_corrected_value > 1:
        bh_corrected_value = 1.0
    return bh_corrected_value


def calc_benjamini_hochberg_corrections(p_values, num_total_tests=None):
    """
    Calculates the Benjamini-Hochberg correction for multiple hypothesis
    testing from a list of p-values *sorted in ascending order*.

    See
    http://en.wikipedia.org/wiki/False_discovery_rate#Independent_tests
    for more detail on the theory behind the correction.

    **NOTE:** This is a generator, not a function. It will yield values
    until all calculations have completed.

    :Parameters:
    - `p_values`: a list or iterable of p-values sorted in ascending
      order
    - `num_total_tests`: the total number of tests (p-values) for 
      which to correct. Default: len(p_values)

    """
    if num_total_tests is None:
        num_total_tests = len(p_values) 
    prev_bh_value = 0
    for i, p_value in enumerate(p_values):
        bh_value = calc_benjamini_hochberg_corrected_value(
                p_value, i + 1, num_total_tests)
        # One peculiarity of this correction is that even though our
        # uncorrected p-values are in monotonically increasing order,
        # the corrected p-values may actually not wind up being
        # monotonically increasing. That is to say, the corrected
        # p-value at i may be less than the corrected p-value at i-1. In
        # cases like these, the appropriate thing to do is use the
        # larger, previous corrected value.
        if bh_value < prev_bh_value:
            bh_value = prev_bh_value
        prev_bh_value = bh_value
        yield bh_value
