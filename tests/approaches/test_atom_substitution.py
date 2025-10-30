"""Tests atom substitution."""

from relational_structs import Object, Predicate, Type

from tamp_improv.approaches.improvisational.policies.multi_rl import (
    find_atom_substitution,
)


def test_find_atom_substitution():
    """Test for our find_atom_substitution function."""
    block_type = Type("block")
    surface_type = Type("surface")

    block1 = Object("block1", block_type)
    block2 = Object("block2", block_type)
    block3 = Object("block3", block_type)
    table = Object("table", surface_type)
    target_area = Object("target_area", surface_type)

    on_pred = Predicate("On", [block_type, surface_type])
    clear_pred = Predicate("Clear", [surface_type])

    on_block1_table = on_pred([block1, table])
    on_block2_table = on_pred([block2, table])
    on_block3_table = on_pred([block3, table])
    on_block1_target = on_pred([block1, target_area])
    on_block2_target = on_pred([block2, target_area])
    clear_table = clear_pred([table])
    clear_target = clear_pred([target_area])

    # Test 1: Exact match
    train_atoms = {on_block1_table, clear_table}
    test_atoms = {on_block1_table, clear_table}

    found, assignment = find_atom_substitution(train_atoms, test_atoms)
    assert found
    assert assignment == {block1: block1, table: table}

    # Test 2: test_atoms has more atoms than needed
    train_atoms = {on_block1_table}
    test_atoms = {on_block1_table, on_block2_table, clear_table}

    found, assignment = find_atom_substitution(train_atoms, test_atoms)
    assert found
    assert assignment == {block1: block1, table: table}

    # Test 3: Structure matches with different objects
    train_atoms = {on_block1_table, clear_target}
    test_atoms = {on_block2_table, clear_target}

    found, assignment = find_atom_substitution(train_atoms, test_atoms)
    assert found
    assert assignment == {block1: block2, table: table, target_area: target_area}

    # Test 4: Multiple valid substitutions - should find one
    train_atoms = {on_block1_table}
    test_atoms = {on_block1_table, on_block2_table, on_block3_table}

    found, assignment = find_atom_substitution(train_atoms, test_atoms)
    assert found
    assert assignment[block1] in [block1, block2, block3]
    assert assignment[table] == table

    # Test 5: No match - missing predicate
    train_atoms = {on_block1_table, clear_target}
    test_atoms = {on_block1_table, on_block2_table}

    found, assignment = find_atom_substitution(train_atoms, test_atoms)
    assert not found
    assert assignment == {}

    # Test 6: No match - not enough objects
    train_atoms = {on_block1_table, on_block2_target}
    test_atoms = {on_block1_table}

    found, assignment = find_atom_substitution(train_atoms, test_atoms)
    assert not found
    assert assignment == {}

    # Test 7: Structure matches with swapped objects/surfaces
    train_atoms = {on_block1_table, on_block2_target}
    test_atoms = {on_block1_target, on_block2_table}

    found, assignment = find_atom_substitution(train_atoms, test_atoms)
    assert found
    assert assignment in [
        {block1: block2, block2: block1, table: table, target_area: target_area},
        {block1: block1, block2: block2, table: target_area, target_area: table},
    ]
