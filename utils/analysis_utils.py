import torch

from utils import formula_utils, concept_mask_utils
from src import formula as F

def get_atom_undetected_concepts(atom_to_check, other_atoms):
    """Get the concepts in atom_to_check that are not detected in other_atoms.
    Args:
        atom_to_check: A compositional formula representing the atom to check.
        other_atoms: A list of compositional formulas representing the other atoms.
    Returns:
        A list of compositional formulas representing the undetected concepts in atom_to_check.
    """
    for atom in other_atoms:
        if isinstance(atom_to_check, F.Leaf):
            # If the atom in the reference explanation is a concept in OR (leaf) then the other explanation can detect it
            # either by having the same concept in OR (leaf) or by having it in AND (left or right). In the second case, 
            # it is a case of specialization
            if isinstance(atom, F.Leaf) and atom_to_check.val == atom.val:
                return []
            if isinstance(atom, F.And) and (atom_to_check.val == atom.left.val or atom_to_check.val == atom.right.val):
                return []
        elif isinstance(atom_to_check, F.And):
            # If the atom in the reference explanation is a specialization (AND) then the other explanation can detect it
            # either by having the same specialization (AND) or by having one of its concepts in OR (leaf). In the second case, 
            # it is a case of generalization
            if isinstance(atom, F.And) and atom_to_check == atom:
                return []
            elif isinstance(atom, F.And) and (atom_to_check.left.val == atom.left.val or atom_to_check.right.val == atom.right.val or
                                              atom_to_check.left.val == atom.right.val or atom_to_check.right.val == atom.left.val):
                # In this case, the other explanation has detected only one of the concepts in the specialization
                return get_atom_undetected_concepts(atom_to_check, atom.left) + get_atom_undetected_concepts(atom_to_check, atom.right)
            elif isinstance(atom, F.Leaf) and (atom_to_check.left.val == atom.val or atom_to_check.right.val == atom.val):
                    # In this case, the other explanation has detected only one of the concepts in the specialization
                    # Note that if the 
                return get_atom_undetected_concepts(atom_to_check, other_atoms.remove(atom))
        else:
            raise ValueError(f"Unsupported atom type: {type(atom_to_check)}")    
    return formula_utils.extract_base_atoms(atom_to_check)


def get_undetected_concepts(*, ref_explanation, other_explanation):
    """Get the concepts in ref_explanation that are not detected in other_explanation.
    Args:
        ref_explanation: A compositional formula representing the reference explanation.
        other_explanation: A compositional formula representing the other explanation.
    Returns:
        A list of concepts representing the undetected concepts in ref_explanation.
    """
    ref_explanation_nonot = formula_utils.remove_not_from_formula(ref_explanation)
    other_explanation_nonot = formula_utils.remove_not_from_formula(other_explanation)

    # Get componenets of the explanations
    ref_atoms = ref_explanation_nonot.get_atoms()
    other_atoms = other_explanation_nonot.get_atoms()

    ref_undetected_atoms = []
    for atom in ref_atoms:
        undetected_atoms = get_atom_undetected_concepts(atom, other_atoms)
        if undetected_atoms:
            ref_undetected_atoms.extend(undetected_atoms)

    return ref_undetected_atoms



def get_atom_conditionally_activation(atom, formula, masks, bitmaps,  device):
    """Get the samples where the atom is active, the formula is active, and the bitmaps are active.
    Args:
        atom: A compositional formula representing the atom to check.
        formula: A compositional formula representing the formula to check.
        masks: A tensor of shape (num_samples, num_concepts, height, width) representing the concept masks.
        bitmaps: A tensor of shape (num_samples, height, width) representing the bitmaps.
        device: The device to use for computation.
    Returns:
        A tensor of shape (num_samples,) representing the samples where the atom is active and the formula is active.
    """

    # Atom Active In
    mask_atom = concept_mask_utils.get_formula_mask(atom, masks).to(
        device)
    samples_atom = torch.any(mask_atom, dim=1)

    # Formula Active In
    mask_formula = concept_mask_utils.get_formula_mask(formula, masks).to(
        device)
    samples_formula = torch.any(mask_formula, dim=1)

    # Both Active In
    active_samples = samples_atom & samples_formula & bitmaps
    return active_samples



def map_explanation_to_wordnet(explanation, labels, mapping):
    """Map the labels in an explanation to Wordnet synsets using a provided mapping.
    Args:
        explanation: A compositional formula representing the explanation.
        labels: A list of labels corresponding to the concepts in the explanation.
        mapping: A dictionary mapping labels to Wordnet synsets.    
Returns:
        A new compositional formula with labels replaced by their corresponding Wordnet synsets.
    """
    if isinstance(explanation, F.And):
        return F.And(
            map_explanation_to_wordnet(explanation.left, labels, mapping),
            map_explanation_to_wordnet(explanation.right, labels, mapping)
        )
    elif isinstance(explanation, F.Or):
        return F.Or(
            map_explanation_to_wordnet(explanation.left, labels, mapping),
            map_explanation_to_wordnet(explanation.right, labels, mapping)
        )
    elif isinstance(explanation, F.Not):
        return F.Not(
            map_explanation_to_wordnet(explanation.val, labels, mapping)
        )
    elif isinstance(explanation, F.Leaf):
        label = labels[explanation.val]
        if label in mapping:
            # given oewn-xxxxx-n extract xxxx and cast as int
            return F.Leaf(int(mapping[label].split('-')[1]))
        else:
            return F.Leaf(-explanation.val)  # Return a negative value to indicate that the label was not found in the mapping
    else: 
        raise ValueError(f"Unsupported formula type: {type(explanation)}")    
