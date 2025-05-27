import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Load the dataset
df = pd.read_csv('fashion_retail_sales.csv')

# 2. Drop rows with missing target (Purchase Amount)
df = df.dropna(subset=['Purchase Amount (USD)'])

# 3. Fill other missing values with simple strategies
df['Review Rating'] = df['Review Rating'].fillna(df['Review Rating'].mean())

# 4. Encode categorical data
le_item = LabelEncoder()
df['Item Purchased'] = le_item.fit_transform(df['Item Purchased'])

le_payment = LabelEncoder()
df['Payment Method'] = le_payment.fit_transform(df['Payment Method'])

# Optional: Convert date to useful features
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'], dayfirst=True)
df['Purchase Month'] = df['Date Purchase'].dt.month
df['Purchase Year'] = df['Date Purchase'].dt.year
df = df.drop(columns=['Date Purchase'])

# 5. Features & labels
X = df.drop(columns=['Purchase Amount (USD)'])
y = df['Purchase Amount (USD)']

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 9. Model
class RetailModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = RetailModel(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 10. Training
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

for epoch in range(20):
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.2f}")

# 11. Evaluate
with torch.no_grad():
    preds = model(X_test_tensor)
    test_loss = criterion(preds, y_test_tensor)
    print(f"\nTest Loss (MSE): {test_loss.item():.2f}")