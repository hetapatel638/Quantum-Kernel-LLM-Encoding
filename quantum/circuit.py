import pennylane as qml
import numpy as np
from typing import Callable, List


class QuantumCircuitBuilder:
    """Build shallow quantum circuits with angle encoding"""
    
    def __init__(self, n_qubits: int = 10, max_depth: int = 12):
        self.n_qubits = n_qubits
        self.max_depth = max_depth
        self.dev = qml.device('default.qubit', wires=n_qubits)
    
    def build_circuit(self, angle_functions: List[Callable], entanglement: str = "linear") -> Callable:
        """
        Build quantum feature map circuit with multiple rotation layers
        
        Better architecture uses:
        - Multiple rotation axes (RX, RY, RZ) 
        - Entanglement between rotation layers
        - Data re-uploading for expressivity
        
        Args:
            angle_functions: List of functions that compute rotation angles
            entanglement: Type of entanglement ("linear", "full", or None)
            
        Returns:
            PennyLane QNode
        """
        n_layers = len(angle_functions)
        
        if n_layers > self.max_depth // 3:
            raise ValueError(f"Too many layers: {n_layers} exceeds max_depth/3")
        
        @qml.qnode(self.dev)
        def circuit(x):
            """Multi-layer quantum feature map with data re-uploading"""
            
            # Get angles from encoding function
            if len(angle_functions) > 0:
                theta_result = angle_functions[0](x)
                if isinstance(theta_result, (list, np.ndarray)):
                    angles = np.array(theta_result)
                else:
                    # Fallback to simple encoding
                    angles = np.array([np.pi * x[i % len(x)] for i in range(self.n_qubits)])
            else:
                angles = np.array([np.pi * x[i % len(x)] for i in range(self.n_qubits)])
            
            # Layer 1: Initial RX rotations with primary angles
            for i in range(self.n_qubits):
                qml.RX(angles[i], wires=i)
            
            # Layer 2: RY rotations with shifted data (data re-uploading)
            for i in range(self.n_qubits):
                # Use different linear combination of features for RY
                shift_idx = (i + 3) % len(x)
                ry_angle = np.pi * 0.5 * (x[i % len(x)] + x[shift_idx])
                qml.RY(ry_angle, wires=i)
            
            # First entanglement layer
            if entanglement == "linear":
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            elif entanglement == "full":
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CNOT(wires=[i, j])
            
            # Layer 3: RZ rotations with different feature combination
            for i in range(self.n_qubits):
                shift_idx = (i + 7) % len(x)
                rz_angle = np.pi * 0.3 * (x[i % len(x)] + x[shift_idx])
                qml.RZ(rz_angle, wires=i)
            
            # Layer 4: Second RX layer with modified angles for expressivity
            for i in range(self.n_qubits):
                rx2_angle = angles[i] * 0.5
                qml.RX(rx2_angle, wires=i)
            
            # Final entanglement layer
            if entanglement == "linear":
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.state()
        
        return circuit
    
    def get_circuit_depth(self, angle_functions: List[Callable], entanglement: str = "linear") -> int:
        """Calculate circuit depth"""
        depth = len(angle_functions)  # Rotation layers
        
        if entanglement == "linear":
            depth += len(angle_functions) - 1  # CNOT layers
        
        return depth
    
    def get_gate_count(self, angle_functions: List[Callable], entanglement: str = "linear") -> int:
        """Calculate total gate count"""
        # Rotation gates
        gates = len(angle_functions) * self.n_qubits
        
        # CNOT gates
        if entanglement == "linear":
            gates += (self.n_qubits - 1) * (len(angle_functions) - 1)
        
        return gates


# Test
if __name__ == "__main__":
    # Test circuit building
    builder = QuantumCircuitBuilder(n_qubits=4)
    
    # Define simple angle functions
    def angle_func_1(x):
        return np.pi * x[0]
    
    def angle_func_2(x):
        return np.pi * (x[0] + x[1]) / 2
    
    circuit = builder.build_circuit([angle_func_1, angle_func_2], entanglement="linear")
    
    # Test on dummy input
    x_test = np.random.rand(10)
    state = circuit(x_test)
    
    print(f"State shape: {state.shape}")
    print(f"Circuit depth: {builder.get_circuit_depth([angle_func_1, angle_func_2])}")
    print(f"Gate count: {builder.get_gate_count([angle_func_1, angle_func_2])}")
    
    # Draw circuit
    print("\nCircuit diagram:")
    print(qml.draw(circuit)(x_test))