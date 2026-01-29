import torch
from models.gp_mech_stgnn import GPMechSTGNN

torch.manual_seed(0)

B, L, N, F = 2, 20, 30, 8
H = 5

x_seq = torch.randn(B, L, N, F)

# A_mech 形状是 (N,N)
A_mech = torch.eye(N)  # 最简单先用单位阵测试

# --- prior-only (实验一要的) ---
m_prior = GPMechSTGNN(num_nodes=N, in_dim=F, horizon=H, mode="mech")
with torch.no_grad():
    y, A_learn, gamma = m_prior(x_seq, A_mech)

print("prior-only y:", y.shape)          # (B,N,H)
print("prior-only A_learn:", A_learn.shape)
print("prior-only gamma:", float(gamma)) # 1.0

# --- learn-only ---
m_learn = GPMechSTGNN(num_nodes=N, in_dim=F, horizon=H, mode="learn")
with torch.no_grad():
    y2, A_learn2, gamma2 = m_learn(x_seq, A_mech)

print("learn-only gamma:", float(gamma2)) # 0.0

# --- hybrid ---
m_hyb = GPMechSTGNN(num_nodes=N, in_dim=F, horizon=H, mode="prior_residual")
with torch.no_grad():
    y3, A_learn3, gamma3 = m_hyb(x_seq, A_mech)

print("hybrid gamma:", float(gamma3))    # sigmoid(0.5) ~ 0.62
