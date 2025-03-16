

# Neural ODE for MNIST Classification

This document explains how to set up an environment for running the above Python code, provides context on Neural Ordinary Differential Equations (ODEs), and outlines the advantages of this approach for digit classification on the MNIST dataset.

---

## 1. Installation and Setup

To replicate the experiments, you need Python 3.7+ and the following packages:

1. **PyTorch**  
2. **torchdiffeq** (for ODE integration)
3. **torchvision** (for common datasets and transforms)
4. **numpy** (commonly included with scientific Python distributions)

You can install them using `pip`:

```bash
pip install torch torchvision torchdiffeq
```

Alternatively, if you are using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), you may install packages such as:

```bash
conda install pytorch torchvision -c pytorch
pip install torchdiffeq
```

Ensure your environment recognizes a CUDA-capable GPU if you intend to train on GPU.

---

## 2. Complete Code

Below is the full script that performs MNIST classification via a Neural ODE model:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchdiffeq import odeint

# Define ODE
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.Tanh(),
            nn.Linear(256, 784)
        )

    def forward(self, t, h):
        return self.net(h)

# Define Neural ODE model
class NeuralODEModel(nn.Module):
    def __init__(self, ode_func):
        super(NeuralODEModel, self).__init__()
        self.ode_func = ode_func
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        # Flatten the 28x28 image to 784
        h0 = x.view(x.size(0), -1)
        # Time range for integration
        t = torch.tensor([0.0, 1.0]).to(x.device)
        # Solve ODE from t=0 to t=1
        hT = odeint(self.ode_func, h0, t, method='dopri5')[-1]
        # Final linear layer for classification
        return self.fc(hT)

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ode_func = ODEFunc().to(device)
model = NeuralODEModel(ode_func).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

This code:
1. **Defines** an ODE through `ODEFunc`, mapping the hidden state **h** back to itself.
2. **Creates** a `NeuralODEModel` that integrates this hidden state from  t = 0 to t = 1 using `torchdiffeq.odeint`.
3. **Performs** a linear transform on the final hidden state for classification into 10 digit classes.
4. **Trains** on the MNIST dataset with a standard cross-entropy loss and an Adam optimizer.

---

## 3. Background on Neural ODEs

**Neural Ordinary Differential Equations** (Chen *et al.*, 2018) formulate network transformations as continuous-time flows. Rather than stacking fixed layers, we parameterize the time derivative of h(t):

$$
\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)
$$

This approach allows adaptive computation via ODE solvers (e.g., Runge--Kutta, Dopri5) and can sometimes reduce memory usage by exploiting the **adjoint method** for gradient computation.

---

## 4. What Problem It Solves

Typically, neural networks define discrete transformations between layers. In contrast, a Neural ODE treats a hidden state as continuously evolving from an initial condition **h**(0) to **h**(T) .This paradigm can:
- **Adapt** the number of function evaluations based on input complexity.
- **Provide** a unified framework connecting standard deep networks with physical dynamical systems.
- **Offer** potential memory efficiency gains if the adjoint method is used.

---

## 5. Advantages

1. **Adaptive Computation**:  
   An ODE solver can dynamically control step sizes, using fewer steps for simpler inputs and more for complex ones.

2. **Memory Efficiency**:  
   If implemented with the adjoint method, only the final states are stored, often reducing memory overhead to **O**(1).

3. **Smooth Representations**:  
   Modeling transformations continuously may capture certain data patterns more naturally than discrete layers.

---

## 6. References and Further Reading

- Chen, Ricky T. Q., *et al.* **Neural Ordinary Differential Equations**. *Advances in Neural Information Processing Systems*, 2018.  
- [PyTorch Documentation](https://pytorch.org/docs/stable/)  
