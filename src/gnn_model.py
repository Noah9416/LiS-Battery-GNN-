import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

class GNNModel(nn.Module):
    """GNN模型定义（基于PyTorch Geometric）"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv3(x, edge_index))
        x = self.fc(x)
        return x

def train_model(train_loader, val_loader, epochs=500, lr=1e-4):
    """模型训练与验证"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNModel(input_dim=128, hidden_dim=256, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        
        # 验证集评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                val_loss += criterion(out, data.y).item()
        val_loss /= len(val_loader)
        
        # 早停法（patience=20）
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
        
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    # 示例：加载图结构数据（需根据实际数据格式调整）
    # 假设每个材料的图数据存储在列表中
    dataset = [Data(x=node_features, edge_index=edge_index, y=target) for ...]
    train_loader = DataLoader(dataset[:70], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[70:90], batch_size=32)
    test_loader = DataLoader(dataset[90:], batch_size=32)
    
    train_model(train_loader, val_loader)
