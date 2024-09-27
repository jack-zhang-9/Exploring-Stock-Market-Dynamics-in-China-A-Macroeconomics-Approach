from Model_Economy import InvSubsEcon
from Bisection import bisection
from Shooting_Algorithm import shooting
import matplotlib.pyplot as plt

def plot_paths(econ_list, c_0, k0_list, T, save=False):

    # Create a figure with 6 subplots (3 rows, 2 columns)
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    titles = ['Investment', 'Investment Ratio', 'Dividend', 'Dividend Ratio', 'Valuation', 'Return']

    line_styles = ['-', '--', '-.', ':']  # Different line styles for distinguishing between models

    for i, (econ, k_0) in enumerate(zip(econ_list, k0_list)):

        c_path, k_path = bisection(econ, c_0, k_0, T, k_ter=econ.k_ss, verbose=True)

        # Valuation
        v_path = (1 - econ.τ) * k_path[1:T+2]
        r_path = (econ.f_prime(k_path) / (1 - econ.τ) + 1 - econ.δ)
        pb_path = v_path / k_path[1:]

        # Investment
        x_path = k_path[1:T + 2] - (1 - econ.δ) * k_path[:T+1]
        x_ratio = x_path / k_path[:T+1]

        # Dividend (assumed here based on profit, adapt as necessary)
        p_path = econ.α * econ.f(k_path[:T+1]) - (1-econ.τ) * x_path
        p_ratio = p_path / k_path[:T+1]

        # Define all the paths to be plotted
        inv_sum = [x_path, x_ratio]
        prof_sum = [p_path, p_ratio]
        value_sum = [v_path, r_path]

        # Plot Investment and Investment Ratio (first row)
        for j, path in enumerate(inv_sum):
            axs[0, j].plot(path, linestyle=line_styles[i], label=f"β = {econ.β}")
            axs[0, j].set(xlabel='t', ylabel=titles[j], title=titles[j])
            axs[0, j].legend()

        # Plot Dividend and Dividend Ratio (second row)
        for j, path in enumerate(prof_sum):
            axs[1, j].plot(path, linestyle=line_styles[i], label=f"β = {econ.β}")
            axs[1, j].set(xlabel='t', ylabel=titles[j + 2], title=titles[j + 2])
            axs[1, j].legend()

        # Plot Valuation and Return (third row)
        for j, path in enumerate(value_sum):
            axs[2, j].plot(path, linestyle=line_styles[i], label=f"β = {econ.β}")
            axs[2, j].set(xlabel='t', ylabel=titles[j + 4], title=titles[j + 4])
            axs[2, j].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure if requested
    if save:
        fig.savefig('Figure/Economic_Paths.jpg')

    plt.show()


if __name__ == "__main__":
    econ1 = InvSubsEcon()
    econ2 = InvSubsEcon(τ=0.2)
    # print(econ1.k_ss)
    # print(econ2.k_ss)
    # plot_paths([econ1, econ2], 1, [0.5 * econ1.k_ss, 0.5 * econ1.k_ss], 300, save=True)

    econ3 = InvSubsEcon(β = 0.99)
    plot_paths([econ1, econ3], 1, [0.5 * econ1.k_ss, 0.5 * econ1.k_ss], 300)
