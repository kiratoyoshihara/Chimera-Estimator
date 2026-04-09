import torch
import numpy as np
import matplotlib.pyplot as plt

def run_experiment(dims=[1, 10, 50, 100, 500, 1000, 2000, 5000], offset=0.05, batch_size=32, num_trials=300):
    var_k1_list = []
    var_k3_list = []
    var_chimera_list = []
    
    # Target distribution Q is a standard normal distribution N(0, I)
    dist_q_base = torch.distributions.Normal(0.0, 1.0)
    
    for D in dims:
        grads_k1 = []
        grads_k3 = []
        grads_chimera = []
        
        for _ in range(num_trials):
            mu_k1 = torch.nn.Parameter(torch.full((D,), offset))
            mu_k3 = torch.nn.Parameter(torch.full((D,), offset))
            mu_chimera = torch.nn.Parameter(torch.full((D,), offset))

            # Reparameterization Trick 
            eps = torch.randn(batch_size, D)
            x_k1 = mu_k1 + eps
            x_k3 = mu_k3 + eps
            x_chimera = mu_chimera + eps
            
            # get log_p(x) and log_q(x) for each estimator's current parameters
            dist_p_k1 = torch.distributions.Normal(mu_k1, 1.0)
            log_p_k1 = dist_p_k1.log_prob(x_k1).sum(dim=-1)
            log_q_k1 = dist_q_base.log_prob(x_k1).sum(dim=-1)
            
            dist_p_k3 = torch.distributions.Normal(mu_k3, 1.0)
            log_p_k3 = dist_p_k3.log_prob(x_k3).sum(dim=-1)
            log_q_k3 = dist_q_base.log_prob(x_k3).sum(dim=-1)

            dist_p_chimera = torch.distributions.Normal(mu_chimera, 1.0)
            log_p_chimera = dist_p_chimera.log_prob(x_chimera).sum(dim=-1)
            log_q_chimera = dist_q_base.log_prob(x_chimera).sum(dim=-1)

            log_r_chimera = log_p_chimera - log_q_chimera
            r_chimera = torch.exp(log_r_chimera)
            
            # empirical KL estimates for each estimator's current parameters
            log_r_k1 = log_p_k1 - log_q_k1
            loss_k1 = log_r_k1.mean()  # Naive Estimator
            
            log_r_k3 = log_p_k3 - log_q_k3
            r_k3 = torch.exp(log_r_k3)
            loss_k3 = (r_k3 - 1.0 - log_r_k3).mean()  # Schulman's k3 Estimator

            loss_k1_part = log_r_chimera.mean()
            loss_k3_part = (r_chimera - 1.0 - log_r_chimera).mean()
            loss_chimera = loss_k3_part.detach() + loss_k1_part - loss_k1_part.detach()

            loss_k1.backward()
            loss_k3.backward()
            loss_chimera.backward()
            
            grads_k1.append(mu_k1.grad.detach())
            grads_k3.append(mu_k3.grad.detach())
            grads_chimera.append(mu_chimera.grad.detach())
            
        # [num_trials, D]
        grads_k1 = torch.stack(grads_k1)
        grads_k3 = torch.stack(grads_k3)
        grads_chimera = torch.stack(grads_chimera)
        
        var_k1 = grads_k1.var(dim=0).mean().item()
        var_k3 = grads_k3.var(dim=0).mean().item()
        var_chimera = grads_chimera.var(dim=0).mean().item()
        
        var_k1_list.append(var_k1)
        var_k3_list.append(var_k3)
        var_chimera_list.append(var_chimera)

        print(f"Dim: {D:4d} | Var(k1): {var_k1:.6f} | Var(k3): {var_k3:.6f} | Var(Chimera): {var_chimera:.6f}")
        
    return dims, var_k1_list, var_k3_list, var_chimera_list

def plot_results(dims, var_k1, var_k3, var_chimera):
    bg_color = '#faf9f6'
    text_color = '#1a1a1a'
    k1_color = '#6495ED' 
    k3_color = '#d63031' 
    chimera_color = '#2ecc71' 
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    ax.plot(dims, var_k1, marker='o', color=k1_color, linewidth=5, alpha=0.4, label='Naive (k1)')
    ax.plot(dims, var_k3, marker='s', color=k3_color, linewidth=2, label="Schulman's (k3)")
    ax.plot(dims, var_chimera, marker='^', color=chimera_color, linewidth=2, linestyle='--', label='Chimera Estimator')

    ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.set_title('Gradient Variance in Empirical KL Estimation', fontsize=14, color=text_color, pad=15)
    ax.set_xlabel('Dimensionality (D) - Log Scale', fontsize=12, color=text_color)
    ax.set_ylabel('Variance of Gradients - Log Scale', fontsize=12, color=text_color)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#d4d0c8')
    ax.spines['left'].set_color('#d4d0c8')
    ax.tick_params(colors=text_color)
    
    ax.legend(frameon=False, fontsize=11, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5, color='#d4d0c8')
    plt.tight_layout()
    
    plt.savefig('kl_variance_plot.png', dpi=300, facecolor=bg_color)
    plt.show()

if __name__ == "__main__":
    print("Running experiment...")
    dims, var_k1, var_k3, var_chimera = run_experiment()
    print("Plotting results...")
    plot_results(dims, var_k1, var_k3, var_chimera)
