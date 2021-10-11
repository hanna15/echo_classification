# TODO: See if will make this into a class, or else re-name it as label_utils

# Mapping the original four labels (including in-between labels) to 3 classes.
# Keys represent original labels, values new labels.
# .5 keys represent labels in-between two original labels.
label_map_3class = {
    0: 0,
    0.5: 0,
    1: 1,
    1.5: 1,
    2: 2,
    2.5: 2,
    3: 2
}

# Mapping the original four labels (including in-between labels) to 2 classes.
# Keys represent original labels, values new labels.
# .5 keys represent labels in-between two original labels.
label_map_2class = {  # 0 and 0-1 is 'norma', rest (1, 1-2, 2, 2-3, 3) is 'abnormal)
    0: 0,
    0.5: 0,
    1: 1,
    1.5: 1,
    2: 1,
    2.5: 1,
    3: 1
}


def get_legal_float_labels(raw_ph_label):
    """
    Given a raw label, return empty string if not legal label (e.g. nan, 'undecided', or wrong range).
    In case of a legal label, return the floating point equivalent of the label (0, 0.5, 1, 1.5, 2, 2.5, 3).
    (The .5 comes from labels 'between' two categories.)
    :param raw_ph_label:
    :return: Floating point mapping of the label, or 0.0 if non-legal.
    """
    if isinstance(raw_ph_label, int) and 0 <= raw_ph_label <= 3:  # legal
        return float(raw_ph_label)
    if isinstance(raw_ph_label, str):  # string is legal if contains 'bis'
        if 'bis' in raw_ph_label:
            return (int(raw_ph_label[-1]) + int(raw_ph_label[0]))/2.0
        else:  # Not a legal label (e.g. 'nichts bestimmt', etc.)
            return -1
    return -1   # if the label is not int, nor string, e.g. a nan - then not legal
