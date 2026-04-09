import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def train_toy_model(estimator_type, D=5000, offset=0.05, batch_size=32, steps=2000, lr=0.02):
    dist_q_base = torch.distributions.Normal(0.0, 1.0)
    
    mu = torch.nn.Parameter(torch.full((D,), offset))
    optimizer = optim.Adam([mu], lr=lr)
    
    loss_history = []
    
    for step in range(steps):
        optimizer.zero_grad()
        
        eps = torch.randn(batch_size, D)
        x = mu + eps
        
        dist_p = torch.distributions.Normal(mu, 1.0)
        log_p = dist_p.log_prob(x).sum(dim=-1)
        log_q = dist_q_base.log_prob(x).sum(dim=-1)
        
        log_r = log_p - log_q
        r = torch.exp(log_r)
        
        if estimator_type == 'k3':
            loss = (r - 1.0 - log_r).mean()
        elif estimator_type == 'chimera':
            loss_k1_part = log_r.mean()
            loss_k3_part = (r - 1.0 - log_r).mean()
            loss = loss_k3_part.detach() + loss_k1_part - loss_k1_part.detach()
        elif estimator_type == 'k1':
            loss = log_r.mean()
            
        loss.backward()
        
        if torch.isnan(mu.grad).any() or torch.isinf(mu.grad).any():
            print(f"[{estimator_type}] Gradient exploded at step {step}!")
            loss_history.extend([np.nan] * (steps - step))
            break
            
        optimizer.step()
        
        track_loss = log_r.mean().item()
        loss_history.append(track_loss)
        
        if torch.isnan(mu).any():
            print(f"[{estimator_type}] Parameter became NaN at step {step}!")
            loss_history.extend([np.nan] * (steps - step - 1))
            break
            
    return loss_history

def plot_training_curves():
    dims_to_test = [10, 100, 500, 1000, 2000, 5000] 
    
    bg_color = '#faf9f6'
    text_color = '#1a1a1a'
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor=bg_color)
    
    axes_flat = axes.flatten()
    
    for i, D in enumerate(dims_to_test):
        print(f"\n--- Running Training for D={D} ---")
        loss_k1 = train_toy_model('k1', D=D)
        loss_k3 = train_toy_model('k3', D=D)
        loss_chimera = train_toy_model('chimera', D=D)
        
        ax = axes_flat[i]
        ax.set_facecolor(bg_color)
        steps = np.arange(len(loss_k1))
        
        ax.plot(steps, loss_k1, color='#6495ED', linewidth=1, alpha=0.4, label='Naive (k1)')
        ax.plot(steps, loss_chimera, color='#2ecc71', linewidth=1, linestyle='--', label='Chimera Estimator')
        ax.plot(steps, loss_k3, color='#d63031', linewidth=1, label="Schulman's (k3)")
        
        ax.set_title(f'Dimension $D={D}$', fontsize=14, color=text_color, pad=10)
        ax.set_xlabel('Training Steps', fontsize=12, color=text_color)
        
        if i % 3 == 0:
            ax.set_ylabel('Empirical KL Loss', fontsize=12, color=text_color)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#d4d0c8')
        ax.spines['left'].set_color('#d4d0c8')
        ax.tick_params(colors=text_color)
        ax.grid(True, linestyle='--', alpha=0.5, color='#d4d0c8')

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3) 
    
    plt.savefig('kl_phase_transition.png', dpi=300, facecolor=bg_color, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_training_curves()
