import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 1. シンプルなニューラルネットワーク（速度場 v_theta を近似）
class VectorFieldNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # 入力は (x, y) 座標と時刻 t の計3次元
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(), # Flow Matchingは滑らかな関数が良いのでTanhやGELUが好まれる
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) # 出力は速度ベクトル (vx, vy)
        )

    def forward(self, x, t):
        # tの形状を (Batch, 1) にして x と結合
        t_embed = t.view(-1, 1).expand(x.shape[0], 1)
        x_in = torch.cat([x, t_embed], dim=1)
        return self.net(x_in)

# 2. Optimal Transport Flow Matching の学習ロジック
def train_flow_matching(model, data, epochs=1000, batch_size=128, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = torch.tensor(data, dtype=torch.float32)
    
    loss_history = []
    
    print("Training Start...")
    for epoch in range(epochs):
        # バッチ取得
        idx = torch.randint(0, len(dataset), (batch_size,))
        x_1 = dataset[idx] # データ分布（ターゲット）
        x_0 = torch.randn_like(x_1) # ガウスノイズ（ソース）
        
        # 時刻 t を [0, 1] で一様サンプリング
        t = torch.rand(batch_size, 1)
        
        # 線形補間による経路 (Optimal Transport Path)
        # psi_t(x) = (1 - t) * x_0 + t * x_1
        x_t = (1 - t) * x_0 + t * x_1
        
        # 目標となる速度場（Target Vector Field）
        # u_t(x | x_1) = x_1 - x_0
        u_t = x_1 - x_0
        
        # モデル予測
        v_pred = model(x_t, t)
        
        # Loss: 予測した速度場と，目標速度場の二乗誤差
        loss = torch.mean((v_pred - u_t) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
    return loss_history

# 3. 推論（サンプリング）：ODEを解く
# ここでは単純なオイラー法を使用
@torch.no_grad()
def sample_flow_matching(model, n_samples=1000, steps=100):
    # ノイズからスタート
    x_t = torch.randn(n_samples, 2)
    dt = 1.0 / steps
    
    traj = [x_t.cpu().numpy()] # 軌跡保存用
    
    for i in range(steps):
        t = torch.ones(n_samples, 1) * (i / steps)
        v_pred = model(x_t, t)
        
        # x_{t+1} = x_t + v(x_t, t) * dt
        x_t = x_t + v_pred * dt
        traj.append(x_t.cpu().numpy())
        
    return x_t.cpu().numpy(), np.array(traj)

# --- 実行部分 ---
if __name__ == "__main__":
    # Toyデータセット作成 (Moons)
    data, _ = make_moons(n_samples=2000, noise=0.05)
    data = (data - data.mean(axis=0)) / data.std(axis=0) # 正規化
    
    # モデル定義
    model = VectorFieldNet()
    
    # 学習
    losses = train_flow_matching(model, data, epochs=2000)
    
    # サンプリング
    generated_data, trajectory = sample_flow_matching(model)
    
    # --- 可視化 ---
    plt.figure(figsize=(12, 4))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    
    # 生成結果
    plt.subplot(1, 3, 2)
    plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.5, label="Real Data")
    plt.scatter(generated_data[:, 0], generated_data[:, 1], s=5, alpha=0.5, label="Generated")
    plt.legend()
    plt.title("Result")
    
    # 軌跡（Vector Fieldの流れ）
    plt.subplot(1, 3, 3)
    # 一部のサンプルの軌跡を描画
    for i in range(50): 
        plt.plot(trajectory[:, i, 0], trajectory[:, i, 1], alpha=0.3, color="black")
    plt.title("Flow Trajectories")
    
    plt.tight_layout()
    plt.show()
    print("Done! Check the plot.")