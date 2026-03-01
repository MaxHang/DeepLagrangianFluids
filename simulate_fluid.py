import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    # ==========================================
    # Simulation Parameters
    # ==========================================
    Nx = 400    # Grid size X
    Ny = 100    # Grid size Y
    tau = 0.53  # Relaxation time
    Nt = 3000   # Time steps
    plot_every = 50 # Plot every N steps

    # LBM D2Q9 Parameters
    NL = 9
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # ==========================================
    # Initialization
    # ==========================================
    np.random.seed(42)
    F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
    F[:,:,:] = weights[np.newaxis, np.newaxis, :] * (1 + 0.01 * np.random.randn(Ny, Nx, NL))

    # Cylinder Obstacle
    cylinder_cx, cylinder_cy = Nx // 4, Ny // 2
    cylinder_r = Ny // 9
    Y, X = np.ogrid[:Ny, :Nx]
    cylinder_mask = (X - cylinder_cx)**2 + (Y - cylinder_cy)**2 < cylinder_r**2

    print(f"Start Simulation: Grid {Nx}x{Ny}, Steps {Nt}")

    fig = plt.figure(figsize=(10, 4))
    frames = []

    for it in range(Nt):
        # 1. Macroscopic
        rho = np.sum(F, axis=2)
        ux = np.sum(F * cxs, axis=2) / rho
        uy = np.sum(F * cys, axis=2) / rho

        # 2. Collision (BGK)
        F_eq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            eu = 3 * (cx * ux + cy * uy)
            u2 = 1.5 * (ux**2 + uy**2)
            eu2 = 4.5 * (cx * ux + cy * uy)**2
            F_eq[:,:,i] = rho * w * (1 + eu + eu2 - u2)
        
        F += -(1.0 / tau) * (F - F_eq)

        # 3. Boundary (Bounce-back)
        bndryF = F[cylinder_mask, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        F[cylinder_mask, :] = bndryF

        # 4. Streaming
        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
        
        # 5. Inlet/Outlet
        F[:, -1, :] = F[:, -2, :] # Outlet
        
        # Inlet (Left)
        u_inlet = 0.1
        rho_inlet = 1.0
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            eu = 3 * (cx * u_inlet + cy * 0)
            u2 = 1.5 * (u_inlet**2)
            eu2 = 4.5 * (cx * u_inlet)**2
            F[:, 0, i] = rho_inlet * w * (1 + eu + eu2 - u2)

        # Collect Data
        if it % plot_every == 0:
            # Calculate Curl
            dy_ux = np.gradient(ux, axis=0)
            dx_uy = np.gradient(uy, axis=1)
            curl = dx_uy - dy_ux
            curl[cylinder_mask] = np.nan
            
            img = plt.imshow(curl, cmap='RdBu_r', animated=True, vmin=-0.05, vmax=0.05)
            frames.append([img])
            
            if it % 100 == 0:
                print(f"Step {it}/{Nt}")

    print("Generating Animation...")
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
    ani.save('fluid_simulation.gif', writer='pillow', fps=20)
    print("Done: fluid_simulation.gif")

if __name__ == "__main__":
    main()
