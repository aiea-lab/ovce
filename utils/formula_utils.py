from src import formula as F

def remove_not_from_formula(formula):
    """
    Recursively removes NOT operations from a formula.
    Args:
        formula: A compositional formula that may contain NOT operations.
    Returns:
        A new formula with NOT operations removed.
    Raises:
        ValueError: If the input formula is a NOT operation.
    """
    if isinstance(formula, F.Not):
        raise ValueError("Cannot remove NOT from a formula that is a NOT operation.")
    elif isinstance(formula, F.And) and isinstance(formula.right, F.Not):
        return formula.left
    elif isinstance(formula, F.And):
        return F.And(remove_not_from_formula(formula.left), remove_not_from_formula(formula.right))
    elif isinstance(formula, F.Or):
        return F.Or(remove_not_from_formula(formula.left), remove_not_from_formula(formula.right))
    else:
        return formula

def get_concepts_from_formula(formula):
    """
    Recursively extracts concepts from a formula.
    Args:
        formula: A compositional formula that may contain AND, OR, and NOT operations.
    Returns:
        A set of concepts extracted from the formula.
    """
    if isinstance(formula, F.And) or isinstance(formula, F.Or):
        return get_concepts_from_formula(formula.left).union(get_concepts_from_formula(formula.right))
    elif isinstance(formula, F.Not):
        return get_concepts_from_formula(formula.val)
    else:
        return {formula}


def extract_base_atoms(positive_label):
    """
    Extracts the base atoms from a positive (no NOT operators) label formula.
    Args:
        positive_label: A positive compositional formula.
    Returns:
        A list of base atoms.
    """
    label_atoms = positive_label.get_atoms()
    base_atoms = []
    for atom in label_atoms:
        if isinstance(atom, F.And):
            base_atoms.extend(extract_base_atoms(atom.left))
            base_atoms.extend(extract_base_atoms(atom.right))
        elif isinstance(atom, F.Leaf):
            base_atoms.append(atom)
        else:
            raise ValueError(f"Unexpected atom type: {type(atom)}. This fuction doesn't work for OR or NOT formulas")
    return base_atoms

def get_positive_concepts(formula):
    """Get the positive concepts from a formula by removing NOT operations and extracting base atoms.
    Args:
        formula: A compositional formula that may contain NOT operations.
    Returns:
        A list of base atoms representing the positive concepts in the formula.
    """
    positive_formula = remove_not_from_formula(formula)
    atoms = positive_formula.get_atoms()
    positive_concepts = []
    for atom in atoms:
        positive_concepts.extend(extract_base_atoms(atom))
    return positive_concepts

def convert_formula(formula, mapping):
    """Convert a formula using the provided mapping.
    Args:
        formula: A compositional formula to convert.
        mapping: A dictionary mapping from the indices of the original concept set to the indices of the target concept set.
    Returns:
        The converted formula.
    """
    if isinstance(formula, F.Leaf):
        converted_leaf = F.Leaf(mapping.get(formula.val, None))
        return converted_leaf
    elif isinstance(formula, F.Or):
        left_leaf = convert_formula(formula.left, mapping)
        right_leaf = convert_formula(formula.right, mapping)
        return F.Or(left_leaf, right_leaf)
    elif isinstance(formula, F.And):
        left_leaf = convert_formula(formula.left, mapping)
        right_leaf = convert_formula(formula.right, mapping)
        return F.And(left_leaf, right_leaf)
    elif isinstance(formula, F.Not):
        return F.Not(convert_formula(formula.val, mapping))
    elif isinstance(formula, int):
        return mapping.get(formula, None)
    else:
        raise ValueError(f"Unknown formula type {type(formula)}")