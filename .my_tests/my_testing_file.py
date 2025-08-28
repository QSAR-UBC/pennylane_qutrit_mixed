import pennylane as qml
@qml.qnode(qml.device('default.mixed', wires=range(4)))
def circuit(x):
    qml.RX(x, 1)
    qml.CNOT((0,1))
    return qml.density_matrix(1)

print(circuit(0.1))

