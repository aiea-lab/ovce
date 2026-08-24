import os
import json
import math

# Noisy synset we manually fix
MANUAL_MAPPING = {'water': 'oewn-07951744-n', 'cushion':'oewn-03156166-n',
                      'van': 'oewn-04527465-n', 'plate': 'oewn-03965779-n',
                      'radiator': 'oewn-04047545-n'}

# Taxonomy ignore nodes
ABSTRACT_NODES = ['attribute', 'form, shape', 'part, portion, component, constituent, component part', 'relation', 'artefact, artifact', 'physical entity', 'unit, whole', 'language unit, linguistic unit', 'being, organism', 'abstraction, abstract entity', 'measure, amount, quantity', 'cause, causal agency, causal agent', 'animate thing, living thing', 'creation',  'entity', 'grouping, group', 'matter', 'instrumentality, instrumentation, means',  'substance']
GENERAL_NODES = ['natural object', 'object, physical object', 'covering', 'surface',  'good, trade good, commodity', 'durable goods, durables, consumer durables', 'structure, construction', 'consumer goods', 'impediment, impedimenta, obstructer, obstruction, obstructor','tracheophyte, vascular plant', 'device']
TASK_NODES =  ['equipment', 'container', 'piece of furniture, article of furniture, furniture','furnishing','barrier','art, fine art','transport, conveyance', 'vessel', 'craft', 'way', 'path']
IGNORE_NODES = ABSTRACT_NODES + GENERAL_NODES + TASK_NODES


def get_words_best_synset(wordnet, words):
    """Get the best synset for a set of words based on the percentage of matches with the synset's lemmas.
    Args:
        wordnet: Wordnet object
        words: List of words to find the best synset for
    Returns:
        The best synset found, or None if no synset was found
    """
    best_synset = (None, 0, 0) # This ensure that the default synset is the first one (most common)    
    for word in words:
        # Get candidate synsets for the word
        candidate_synsets = wordnet.synsets(word, pos='n')        
        for synset in candidate_synsets:
            # Get the lemmas of the synset
            lemmas_synset = synset.lemmas()
            lemmas_synset = [lemma.lower() for lemma in lemmas_synset]

            # Get the intersection of the lemmas and the full set of words and compute the percentage of matches
            common = list(set(lemmas_synset).intersection(words))
            percentage_words = len(common)/len(words)
            percentage_lemmas = len(common)/len(lemmas_synset)

            # Update the best synset if the current one has a higher percentage of matches
            if percentage_words > best_synset[1]:
                best_synset = (synset.id, percentage_words, percentage_lemmas)
            elif percentage_words == best_synset[1] and percentage_lemmas > best_synset[2]:
                best_synset = (synset.id, percentage_words, percentage_lemmas)
    # Return the best synset found, or None if no synset was found
    return best_synset[0]

def build_synset_mapping_best_synset(*, wordnet, labels):
    """Build a mapping from a list of labels to their best synset in Wordnet based on the percentage of matches with the synset's lemmas.
    Args:
        wordnet: Wordnet object
        labels: List of labels to map to synsets
    Returns:
        A dictionary mapping each label to its best synset found, or None if no synset was found
    """
    mapping = {}
    for concept in labels:
        if ',' in concept:
            # Case of concept expressed by multiple synonyms, e.g., "car, automobile"
            words = concept.split(',')
            words = [word.strip() for word in words] # Necessary to fix some typos
        else:
            if concept in MANUAL_MAPPING:
                mapping[concept] = MANUAL_MAPPING[concept]
                continue
            # Single word concept, e.g., "car"
            words = [concept]

        # Get the best synset for the set of words     
        best_synset = get_words_best_synset(wordnet, words)
        if best_synset is not None:
            mapping[concept] = best_synset
    return mapping



def save_synset_mapping(mapping, mapping_file):
    """Save a mapping to a file.
    Args:
        mapping: Dictionary mapping each label to its best synset found
        mapping_file: Path to the mapping file
    """
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f)

def get_synset_from_number(wordnet, number):
    """Get a synset from its number in Wordnet.
    Args:
        wordnet: Wordnet object
        number: Number of the synset to get
    Returns:
        The synset corresponding to the number, or None if no synset was found
    """
    id = 'oewn-' + str(number).zfill(8) + '-n'
    synset = wordnet.synset(id)
    return synset

def get_ignore_synsets(wordnet):
    """Get the synsets corresponding to a list of ignore nodes in Wordnet.
    Args:
        wordnet: Wordnet object
        ignore_nodes: List of ignore nodes to get the synsets for
    Returns:
        A list of synsets corresponding to the ignore nodes
    """

    # We consider all the the possible synsets for the ignore nodes
    mapping_unmeaningfull_concepts = {}
    for concept in IGNORE_NODES:
        if ',' in concept:
            words = concept.split(',')
            words = [word.strip() for word in words] # Necessary to fix some typos
            candidate_synsets = []
            for word in words:
                candidate_synsets += wordnet.synsets(word, pos='n')
        else:
            candidate_synsets = wordnet.synsets(concept, pos='n')

        if len(candidate_synsets) == 0:
            continue
        else:
            mapping_unmeaningfull_concepts[concept] = candidate_synsets

    unmeaningfull_synsets = []
    for concept in mapping_unmeaningfull_concepts.keys():
        unmeaningfull_synsets.extend(mapping_unmeaningfull_concepts[concept])
    unmeaningfull_synsets = list(set(unmeaningfull_synsets))
    return unmeaningfull_synsets

def get_common_ancestor(wordnet, id1, id2):
    """Unify two synsets in Wordnet by finding their lowest common hypernyms that are not unmeaningful.
    Args:
        wordnet: Wordnet object
        id1: Number of the first synset to unify
        id2: Number of the second synset to unify
    Returns:
        The lowest common hypernym of the two synsets that is not unmeaningful
    """
    # Get synset from their numbers
    synset1 = get_synset_from_number(wordnet, id1)
    synset2 = get_synset_from_number(wordnet, id2)

    UNMEANGINGFUL_SYNSETS = get_ignore_synsets(wordnet)

    # Compute the lowest common hypernyms of the two synsets that are not unmeaningful
    common_hypernyms = [h for h in synset1.lowest_common_hypernyms(synset2) if h not in UNMEANGINGFUL_SYNSETS]

    # Extract the lowest common ancestors in case there are multiple
    min_distance = math.inf
    lowest_common_ancestor = None
    for hypernym in common_hypernyms:
        # Sum of distances
        distance = len(hypernym.shortest_path(synset1)) + len(hypernym.shortest_path(synset2))
        if distance < min_distance:
            min_distance = distance
            lowest_common_ancestor = hypernym
        elif distance == min_distance:
            # TIe break based on the distance from the ref synset (Synset 1)
            distance_from_ref = len(hypernym.shortest_path(synset1))
            distance_from_ref_lowest = len(lowest_common_ancestor.shortest_path(synset1))
            if distance_from_ref < distance_from_ref_lowest:
                lowest_common_ancestor = hypernym                
    return lowest_common_ancestor, min_distance

def search_common_ancestors(*, wordnet, concept_to_unify, candidate_concepts):
    """
    Find the best common ancestors in Wordnet between an undetected concept and the concepts in an explanation.
    Args:
        wordnet: Wordnet object
        concept_to_unify: A  concept to unify.
        candidate_concepts: A list of concepts to compare.
    Returns:
        A list of tuples containing the best common ancestor synsets, the corresponding concepts from the explanation, and the distances to the ancestors.
    """
    if concept_to_unify.val < 0:
        # We assume that a val < 0 means that the concept is not mapped to Wordnet, so we cannot find a common ancestor
        return ([], [])
    best_common_ancestor = []
    best_matched_concept = []
    for concept in candidate_concepts:
        if concept.val < 0:
            # No mapping in wordnet
            continue
        ancestor_synset, distance = get_common_ancestor(wordnet, concept_to_unify.val, concept.val)
        if ancestor_synset is not None:
            best_common_ancestor.append(ancestor_synset)
            best_matched_concept.append(concept)
    return best_common_ancestor, best_matched_concept

def get_number_from_synset(synset):
    """Get the number of a synset in Wordnet.
    Args:
        synset: Synset to get the number for
    Returns:
        The number of the synset
    """
    id = synset.id
    number = int(id.split('-')[1])
    return number

def generalize_labels_with_ancestors(*, synsets, ancestors):
    """
    Generalize a list of synsets to their ancestors in Wordnet.
    Args:
        synsets: List of synsets to generalize
        ancestors: List of ancestor synsets to generalize to
    Returns:
        A list of tuples containing the generalized synsets, the corresponding original synsets, and the ancestor synsets.
    """
    unification = []
    for synset_label in synsets:
        number_label = get_number_from_synset(synset_label)
        for path in synset_label.hypernym_paths(synset_label):
            for hyp in path:
                if hyp in ancestors and hyp != synset_label:
                    # We generalize the label to the ancestor
                    unification.append((number_label, number_label, hyp))
    return unification