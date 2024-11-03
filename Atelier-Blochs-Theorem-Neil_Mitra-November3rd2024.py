import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def create_hamiltonian(N, V0, a):
    """
    Create the Hamiltonian matrix for a periodic potential.
    
    Args:
        N (int): Number of basis states
        V0 (float): Potential strength
        a (float): Lattice constant
    
    Returns:
        numpy.ndarray: Hamiltonian matrix
    """
    # Create kinetic energy matrix
    K = np.zeros((N, N))
    for i in range(N):
        k = 2 * np.pi * (i - N//2) / (N * a)
        K[i, i] = k**2 / 2

    # Create potential energy matrix
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if abs(i-j) == 1:
                V[i, j] = V0/4
            elif i == j:
                V[i, j] = V0/2

    return K + V

def calculate_band_structure(k_points, N, V0, a):
    """
    Calculate the band structure for given k-points.
    
    Args:
        k_points (numpy.ndarray): Array of k-points
        N (int): Number of basis states
        V0 (float): Potential strength
        a (float): Lattice constant
    
    Returns:
        numpy.ndarray: Eigenvalues for each k-point
    """
    energies = []
    
    for k in k_points:
        # Create Hamiltonian with Bloch phase factors
        H = create_hamiltonian(N, V0, a)
        for i in range(N):
            for j in range(N):
                H[i, j] *= np.exp(1j * k * a * (i-j))
        
        # Calculate eigenvalues
        eigenvals = eigh(H, eigvals_only=True)
        energies.append(eigenvals)
    
    return np.array(energies)

def plot_results(k_points, energies, V0, a, num_bands=4):
    """
    Plot the band structure and periodic potential.
    
    Args:
        k_points (numpy.ndarray): Array of k-points
        energies (numpy.ndarray): Array of energies
        V0 (float): Potential strength
        a (float): Lattice constant
        num_bands (int): Number of bands to plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot band structure
    for band in range(num_bands):
        ax1.plot(k_points, energies[:, band], 'b-')
    
    ax1.set_xlabel('Wave vector k')
    ax1.set_ylabel('Energy E')
    ax1.set_title('Band Structure')
    ax1.grid(True)
    
    # Plot periodic potential
    x = np.linspace(-2*a, 2*a, 1000)
    V = V0 * np.cos(2*np.pi*x/a)**2
    
    ax2.plot(x, V, 'r-')
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Potential V(x)')
    ax2.set_title('Periodic Potential')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Simulation parameters
N = 50  # Number of basis states
V0 = 1.0  # Potential strength
a = 1.0  # Lattice constant
k_points = np.linspace(-np.pi/a, np.pi/a, 100)  # k-points in first Brillouin zone

# Run simulation
energies = calculate_band_structure(k_points, N, V0, a)
plot_results(k_points, energies, V0, a)

# Example usage:
if __name__ == "__main__":
    # Calculate and plot band structure
    energies = calculate_band_structure(k_points, N, V0, a)
    plot_results(k_points, energies, V0, a)
    
    # Optional: Print band gaps
    for i in range(min(3, N-1)):
        band_gap = np.min(energies[:, i+1]) - np.max(energies[:, i])
        print(f"Band gap {i+1}: {band_gap:.4f}")