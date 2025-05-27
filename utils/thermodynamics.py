import torch
import torch.nn.functional as F

def compute_log10P(x1, x2, gamma1, gamma2, log10P1sat, log10P2sat):
    """
    Compute predicted log10P using modified Raoult’s Law:
    log10P = log10(x1*gamma1*P1sat + x2*gamma2*P2sat)
    """
    P1 = 10 ** log10P1sat
    P2 = 10 ** log10P2sat
    total_P = x1 * gamma1 * P1 + x2 * gamma2 * P2
    log10P_pred = torch.log10(total_P)
    return log10P_pred

def thermodynamic_loss_log10P(log10P_pred, log10P_exp):
    """
    MSE loss between predicted and experimental log10P
    """
    return F.mse_loss(log10P_pred, log10P_exp)
