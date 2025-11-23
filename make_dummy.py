# FILE: make_dummy.py
import torch
import torch.nn as nn

# 1. Define the exact model structure used in app.py
class Classifier(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.fc1 = nn.Linear(dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, classes)
    def forward(self, x): return self.fc2(self.relu(self.fc1(x)))

# 2. Define Dummy Data 
dim = 808 
classes = ['andhra_pradesh', 'gujrat', 'jharkhand', 'karnataka', 'kerala', 'tamil']
model = Classifier(dim, len(classes))

# 3. Save with the CORRECT KEYS for app.py
print("Saving model with correct keys...")
torch.save({
    "model_state_dict": model.state_dict(), # app.py looks for this
    "input_dim": dim,                       # app.py looks for this
    "num_classes": len(classes),            # app.py looks for this
    "l1_classes": classes                   # app.py looks for this
}, "best_nli_model.pth")

print("✅ Success! 'best_nli_model.pth' created. Now run app.py.")