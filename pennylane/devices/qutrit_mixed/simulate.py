# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulate a quantum script for a qutrit mixed state device."""
# pylint: disable=protected-access
from numpy.random import default_rng

import pennylane as qml
from pennylane.typing import Result

from .initialize_state import create_initial_state
from .measure import measure
from .sampling import measure_with_samples
from ..qtcorgi_helper.apply_operations import qutrit_branches
import jax
import jax.numpy as jnp
import numpy as np

INTERFACE_TO_LIKE = {
    # map interfaces known by autoray to themselves
    None: None,
    "numpy": "numpy",
    "autograd": "autograd",
    "jax": "jax",
    "torch": "torch",
    "tensorflow": "tensorflow",
    # map non-standard interfaces to those known by autoray
    "auto": None,
    "scipy": "numpy",
    "jax-jit": "jax",
    "jax-python": "jax",
    "JAX": "jax",
    "pytorch": "torch",
    "tf": "tensorflow",
    "tensorflow-autograph": "tensorflow",
    "tf-autograph": "tensorflow",
}


def get_qutrit_final_state_from_initial(operations, initial_state):
    """
    TODO

    Args:
        operations ():TODO
        initial_state ():TODO

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    ops_type_indices, ops_subspace, ops_wires, ops_params = [[], []], [], [[], []], [[], [], []]
    two_qutrit_ops = False
    for op in operations:
        wires = op.wires

        if isinstance(op, qml.operation.Channel):
            ops_type_indices[0].append(1)
            ops_type_indices[1].append(
                [qml.QutritDepolarizingChannel, qml.QutritAmplitudeDamping, qml.TritFlip].index(
                    type(op)
                )
            )
            params = op.parameters + ([0] * (3 - op.num_params))
            wires = [wires[0], -1]
        elif len(wires) == 1:
            ops_type_indices[0].append(0)
            ops_type_indices[1].append([qml.THadamard, qml.TRX, qml.TRY, qml.TRZ].index(type(op)))
            subspace_index = [None, (0, 1), (0, 2), (1, 2)].index(op.subspace)
            if ops_type_indices[1][-1] == 0:
                params = [0.0, 0.0, 0.0]
            else:
                params = list(op.parameters) + [0.0, 0.0]
            wires = [wires[0], subspace_index]
        elif len(wires) == 2:
            ops_type_indices[0].append(2)
            ops_type_indices[1].append(0 if isinstance(op, qml.TAdd) else 1)
            params = [0.0, 0.0, 0.0]
            two_qutrit_ops = True
        else:
            raise ValueError("TODO")
        ops_params[0].append(params[0])
        ops_params[1].append(params[1])
        ops_params[2].append(params[2])

        if len(wires) == 1:
            wires = [wires[0], -1]
        ops_wires[0].append(wires[0])
        ops_wires[1].append(wires[1])

    ops_info = {
        "type_indices": np.array(ops_type_indices).T,
        "wires": [np.array(ops_wires[0]), np.array(ops_wires[1])],
        "params": [jnp.array(ops_params[0]), np.array(ops_params[1]), np.array(ops_params[2])],
    }
    branches = qutrit_branches[: 2 + two_qutrit_ops]

    @jax.jit
    def switch_function(state, op_info):
        return jax.lax.switch(op_info["type_indices"][0], branches, state, op_info), None

    return jax.lax.scan(switch_function, initial_state, ops_info)[0]


def measure_final_state(  # pylint: disable=too-many-arguments
    circuit, state, is_state_batched, rng=None, prng_key=None, readout_errors=None
) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by ``default.qutrit.mixed``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        state (TensorLike): The state to perform measurement on
        is_state_batched (bool): Whether the state has a batch dimension or not.
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, the default ``sample_state`` function and a ``numpy.random.default_rng``
            will be for sampling.
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        Tuple[TensorLike]: The measurement results
    """

    circuit = circuit.map_to_standard_wires()

    if not circuit.shots:
        # analytic case
        if len(circuit.measurements) == 1:
            return measure(circuit.measurements[0], state, is_state_batched, readout_errors)

        return tuple(
            measure(mp, state, is_state_batched, readout_errors) for mp in circuit.measurements
        )

    # finite-shot case
    rng = default_rng(rng)
    results = tuple(
        measure_with_samples(
            mp,
            state,
            shots=circuit.shots,
            is_state_batched=is_state_batched,
            rng=rng,
            prng_key=prng_key,
            readout_errors=readout_errors,
        )
        for mp in circuit.measurements
    )

    if len(circuit.measurements) == 1:
        return results[0]
    if circuit.shots.has_partitioned_shots:
        return tuple(zip(*results))
    return results


def get_final_state_qutrit(circuit, **kwargs):
    """
    TODO

    Args:
        circuit (.QuantumScript): The single circuit to simulate

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """

    circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = jnp.complex128(create_initial_state(sorted(circuit.op_wires), prep, like="jax"))
    return get_qutrit_final_state_from_initial(circuit.operations[bool(prep) :], state)


def simulate(  # pylint: disable=too-many-arguments
    circuit: qml.tape.QuantumScript,
    rng=None,
    prng_key=None,
    debugger=None,
    interface=None,
    readout_errors=None,
) -> Result:
    """Simulate a single quantum script.

    This is an internal function that will be called by ``default.qutrit.mixed``.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with
        readout_errors (List[Callable]): List of channels to apply to each wire being measured
        to simulate readout errors.

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.TRX(1.2, wires=0)], [qml.expval(qml.GellMann(0, 3)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.        , 0.31882112, 0.        , 0.        ], requires_grad=True))

    """
    state = get_final_state_qutrit(
        circuit, debugger=debugger, interface=interface, rng=rng, prng_key=prng_key
    )
    return measure_final_state(
        circuit, state, False, rng=rng, prng_key=prng_key, readout_errors=readout_errors
    )
