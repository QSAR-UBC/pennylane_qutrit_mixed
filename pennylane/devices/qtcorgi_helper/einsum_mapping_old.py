import time
import jax
import jax.numpy as jnp
from jax.lax import scan

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
import pennylane as qml
from string import ascii_letters as alphabet

import numpy as np
from functools import partial, reduce

alphabet_array = np.array(list(alphabet))


def get_einsum_mapping(wires, state):
    r"""Finds the indices for einsum to apply kraus operators to a mixed state

    Args:
        wires
        state (array[complex]): Input quantum state

    Returns:
        str: Indices mapping that defines the einsum
    """
    num_ch_wires = len(wires)
    num_wires = int(len(qml.math.shape(state)) / 2)
    rho_dim = 2 * num_wires

    # Tensor indices of the state. For each qutrit, need an index for rows *and* columns
    state_indices = alphabet[:rho_dim]

    # row indices of the quantum state affected by this operation
    row_wires_list = wires
    row_indices = "".join(alphabet_array[row_wires_list].tolist())

    # column indices are shifted by the number of wires
    col_wires_list = [w + num_wires for w in row_wires_list]
    col_indices = "".join(alphabet_array[col_wires_list].tolist())

    # indices in einsum must be replaced with new ones
    new_row_indices = alphabet[rho_dim : rho_dim + num_ch_wires]
    new_col_indices = alphabet[rho_dim + num_ch_wires : rho_dim + 2 * num_ch_wires]

    # index for summation over Kraus operators
    kraus_index = alphabet[rho_dim + 2 * num_ch_wires : rho_dim + 2 * num_ch_wires + 1]
    print(kraus_index)

    # apply mapping function
    op_1_indices = f"{kraus_index}{new_row_indices}{row_indices}"
    op_2_indices = f"{kraus_index}{col_indices}{new_col_indices}"

    new_state_indices = get_new_state_einsum_indices(
        old_indices=col_indices + row_indices,
        new_indices=new_col_indices + new_row_indices,
        state_indices=state_indices,
    )
    # index mapping for einsum, e.g., '...iga,...abcdef,...idh->...gbchef'
    return f"...{op_1_indices},...{state_indices},...{op_2_indices}->...{new_state_indices}"


def get_new_state_einsum_indices(old_indices, new_indices, state_indices):
    """Retrieves the einsum indices string for the new state

    Args:
        old_indices (str): indices that are summed
        new_indices (str): indices that must be replaced with sums
        state_indices (str): indices of the original state

    Returns:
        str: The einsum indices of the new state
    """
    return reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(old_indices, new_indices),
        state_indices,
    )


QUDIT_DIM = 3

print(get_einsum_mapping([0, 1], np.zeros((3, 3, 3, 3, 3, 3, 3, 3))))
