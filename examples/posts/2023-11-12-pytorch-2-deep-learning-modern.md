---
layout: post
title: PyTorch 2.0 - Deep Learning Moderno y Optimizado
tags: [pytorch, deep-learning, neural-networks, ai, machine-learning, python]
---

**PyTorch 2.0** ha revolucionado el desarrollo de modelos de deep learning con mejoras significativas en performance y facilidad de uso. Como framework que domina tanto la investigación como la producción, PyTorch continúa evolucionando para satisfacer las demandas del machine learning moderno.

## Novedades de PyTorch 2.0

### 1. torch.compile() - El Game Changer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout and pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# Modelo tradicional
model_original = ConvNet(num_classes=10)

# Modelo compilado con PyTorch 2.0
model_compiled = torch.compile(ConvNet(num_classes=10))

# Benchmark de performance
def benchmark_model(model, data_loader, device, num_iterations=100):
    model.to(device)
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= 5:  # 5 iteraciones de warm-up
                break
            data = data.to(device)
            _ = model(data)
    
    # Benchmark real
    start_time = time.time()
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_iterations:
                break
            data = data.to(device)
            outputs = model(data)
    
    end_time = time.time()
    return (end_time - start_time) / num_iterations

# Crear datos sintéticos para benchmark
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
synthetic_data = torch.randn(1000, 3, 32, 32)
synthetic_labels = torch.randint(0, 10, (1000,))
dataset = TensorDataset(synthetic_data, synthetic_labels)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Comparación de performance
time_original = benchmark_model(model_original, data_loader, device)
time_compiled = benchmark_model(model_compiled, data_loader, device)

print(f"Modelo original: {time_original:.4f}s por batch")
print(f"Modelo compilado: {time_compiled:.4f}s por batch")
print(f"Speedup: {time_original/time_compiled:.2f}x")

# Resultados típicos:
# Modelo original: 0.0123s por batch
# Modelo compilado: 0.0087s por batch  
# Speedup: 1.41x
```

### 2. Improved Memory Management

```python
import torch
import psutil
import os

def get_memory_usage():
    """Obtener uso de memoria actual"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def memory_efficient_training():
    """Técnicas de optimización de memoria en PyTorch 2.0"""
    
    # 1. Gradient Checkpointing
    from torch.utils.checkpoint import checkpoint
    
    class MemoryEfficientBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
        def forward(self, x):
            # Usar checkpoint para ahorrar memoria
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        
        def _forward_impl(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return x
    
    # 2. Mixed Precision Training
    from torch.cuda.amp import autocast, GradScaler
    
    model = ConvNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    # Training loop con mixed precision
    model.train()
    for epoch in range(5):
        print(f"Epoch {epoch + 1}, Memory usage: {get_memory_usage():.1f} MB")
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass con autocast
            with autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)
            
            # Backward pass con gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}, "
                      f"Memory: {get_memory_usage():.1f} MB")
    
    # 3. Optimized DataLoader
    optimized_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # Nuevo en PyTorch 1.7+
        prefetch_factor=2
    )
    
    return model

# Ejecutar entrenamiento optimizado
trained_model = memory_efficient_training()
```

## Architecturas Modernas con PyTorch

### 1. Vision Transformers (ViT)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, 
                 mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        
        image_height, image_width = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.pool = pool
        self.to_latent = nn.Identity()

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Apply transformer
        x = self.transformer(x)

        # Pooling
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # Classification
        x = self.to_latent(x)
        return self.mlp_head(x)

# Crear y entrenar ViT
vit_model = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
    dropout=0.1,
    emb_dropout=0.1
)

# Compilar para mejor performance
vit_compiled = torch.compile(vit_model)

print(f"ViT Parameters: {sum(p.numel() for p in vit_model.parameters()):,}")
```

### 2. ResNet Moderno con Mejores Prácticas

```python
class ModernResNetBlock(nn.Module):
    """ResNet block optimizado con mejores prácticas 2023"""
    
    def __init__(self, in_channels, out_channels, stride=1, groups=1, 
                 base_width=64, dilation=1, norm_layer=None, activation=None):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)
            
        width = int(out_channels * (base_width / 64.0)) * groups
        
        # Convoluciones con separación de características
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                              padding=dilation, groups=groups, bias=False,
                              dilation=dilation)
        self.bn2 = norm_layer(width)
        
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels)
        
        self.activation = activation
        
        # Skip connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                norm_layer(out_channels),
            )
            
        # Squeeze-and-Excitation block
        self.se_block = SEBlock(out_channels, reduction=16)
        
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # SE attention
        out = self.se_block(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.activation(out)
        
        return out

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ModernResNet(nn.Module):
    """ResNet moderno con últimas optimizaciones"""
    
    def __init__(self, layers=[3, 4, 6, 3], num_classes=1000, groups=1,
                 width_per_group=64, zero_init_residual=True):
        super().__init__()
        
        norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        # Stem más agresivo
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        )
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # Classification head con dropout
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)
        
        # Inicialización moderna
        self._initialize_weights(zero_init_residual)
        
    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                norm_layer(planes),
            )
            
        layers = []
        layers.append(ModernResNetBlock(
            self.inplanes, planes, stride, self.groups,
            self.base_width, previous_dilation, norm_layer
        ))
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(ModernResNetBlock(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer
            ))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize residual connections
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ModernResNetBlock):
                    nn.init.constant_(m.bn3.weight, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Crear modelo moderno
modern_resnet = ModernResNet(layers=[3, 4, 6, 3], num_classes=10)
modern_resnet_compiled = torch.compile(modern_resnet)

print(f"Modern ResNet Parameters: {sum(p.numel() for p in modern_resnet.parameters()):,}")
```

## Training Loop Optimizado

### 1. Training con Best Practices 2024

```python
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import wandb  # Para logging de experimentos

class ModernTrainer:
    """Trainer moderno con todas las optimizaciones actuales"""
    
    def __init__(self, model, train_loader, val_loader, num_classes, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # Compilar modelo para PyTorch 2.0
        self.model = torch.compile(self.model)
        
        # Optimizer con mejores hiperparámetros
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.05,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler moderno
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=100,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Loss function con label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Métricas
        self.best_acc = 0.0
        self.train_losses = []
        self.val_accs = []
        
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass con mixed precision
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Backward pass con gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Scheduler step (para OneCycleLR)
            self.scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Logging
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                accuracy = 100. * correct / total
                
                print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {running_loss/(batch_idx+1):.6f}, '
                      f'Acc: {accuracy:.2f}%, LR: {current_lr:.6f}')
                
                # Log a wandb si está configurado
                if wandb.run is not None:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/accuracy': accuracy,
                        'train/lr': current_lr,
                        'epoch': epoch
                    })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        self.train_losses.append(epoch_loss)
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(self.val_loader)
        accuracy = 100. * correct / total
        self.val_accs.append(accuracy)
        
        # Save best model
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_acc': self.best_acc,
            }, 'best_model.pth')
            print(f'New best accuracy: {accuracy:.2f}%')
        
        # Log a wandb
        if wandb.run is not None:
            wandb.log({
                'val/loss': test_loss,
                'val/accuracy': accuracy,
                'val/best_accuracy': self.best_acc,
                'epoch': epoch
            })
        
        return test_loss, accuracy
    
    def train(self, epochs):
        """Training loop completo"""
        print("Iniciando entrenamiento...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc = self.validate(epoch)
            
            print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")
            print(f"Best Val Acc: {self.best_acc:.2f}%")
            
        print(f"\nEntrenamiento completado. Mejor accuracy: {self.best_acc:.2f}%")

# Ejemplo de uso
def run_training_example():
    # Inicializar wandb (opcional)
    # wandb.init(project="pytorch2-example", name="modern-resnet")
    
    # Crear datasets sintéticos para el ejemplo
    train_dataset = TensorDataset(
        torch.randn(1000, 3, 32, 32),
        torch.randint(0, 10, (1000,))
    )
    val_dataset = TensorDataset(
        torch.randn(200, 3, 32, 32), 
        torch.randint(0, 10, (200,))
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Crear modelo
    model = ModernResNet(layers=[2, 2, 2, 2], num_classes=10)
    
    # Crear trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModernTrainer(model, train_loader, val_loader, 10, device)
    
    # Entrenar
    trainer.train(epochs=10)

# Ejecutar ejemplo
if __name__ == "__main__":
    run_training_example()
```

## Optimización Avanzada

### 1. Custom CUDA Kernels con Triton

```python
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    Q, K, V, output,
    M, N, d,
    BLOCK_SIZE: tl.constexpr
):
    """Kernel optimizado para attention mechanism"""
    
    # Indices del bloque actual
    row_idx = tl.program_id(0)
    
    # Cargar Q para esta fila
    q_ptr = Q + row_idx * d
    q = tl.load(q_ptr + tl.arange(0, d))
    
    # Inicializar acumuladores
    acc = tl.zeros([d], dtype=tl.float32)
    max_score = float('-inf')
    sum_exp = 0.0
    
    # Iterar sobre todas las keys
    for k_block_start in range(0, N, BLOCK_SIZE):
        k_block_end = min(k_block_start + BLOCK_SIZE, N)
        block_size = k_block_end - k_block_start
        
        # Cargar K y V para este bloque
        k_ptrs = K + k_block_start * d + tl.arange(0, d)[None, :]
        v_ptrs = V + k_block_start * d + tl.arange(0, d)[None, :]
        
        k = tl.load(k_ptrs + tl.arange(0, block_size)[:, None])
        v = tl.load(v_ptrs + tl.arange(0, block_size)[:, None])
        
        # Compute attention scores
        scores = tl.sum(q[None, :] * k, axis=1)
        
        # Softmax estable
        block_max = tl.max(scores)
        new_max = tl.maximum(max_score, block_max)
        
        # Reescalar acumuladores previos
        scale_factor = tl.exp(max_score - new_max)
        acc = acc * scale_factor
        sum_exp = sum_exp * scale_factor
        
        # Agregar contribución del bloque actual
        block_exp = tl.exp(scores - new_max)
        weighted_v = tl.sum(v * block_exp[:, None], axis=0)
        
        acc = acc + weighted_v
        sum_exp = sum_exp + tl.sum(block_exp)
        max_score = new_max
    
    # Normalizar resultado final
    output_ptr = output + row_idx * d
    final_output = acc / sum_exp
    tl.store(output_ptr + tl.arange(0, d), final_output)

class OptimizedAttention(nn.Module):
    """Attention optimizada con Triton"""
    
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        b, n, d = x.shape
        
        # Generar Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), qkv)
        
        # Usar kernel Triton optimizado
        batch_heads, seq_len, head_dim = q.shape
        output = torch.empty_like(q)
        
        # Lanzar kernel
        grid = (batch_heads,)
        fused_attention_kernel[grid](
            q, k, v, output,
            seq_len, seq_len, head_dim,
            BLOCK_SIZE=64
        )
        
        # Reshape y proyección final
        output = rearrange(output, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(output)
```

### 2. Distributed Training con DDP

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

def setup_distributed(rank, world_size):
    """Configurar entrenamiento distribuido"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Inicializar proceso group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Configurar CUDA device
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Limpiar proceso group"""
    dist.destroy_process_group()

class DistributedTrainer(ModernTrainer):
    """Trainer distribuido para multi-GPU"""
    
    def __init__(self, model, train_loader, val_loader, num_classes, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        
        # Setup distribuido
        setup_distributed(rank, world_size)
        
        # Mover modelo a GPU específica
        device = torch.device(f'cuda:{rank}')
        model = model.to(device)
        
        # Wrap con DDP
        model = DDP(model, device_ids=[rank])
        
        super().__init__(model, train_loader, val_loader, num_classes, device)
        
    def train_epoch(self, epoch):
        # Configurar sampler para esta época
        self.train_loader.sampler.set_epoch(epoch)
        
        return super().train_epoch(epoch)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Solo guardar desde el proceso principal"""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_acc': self.best_acc,
            }
            
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
            if is_best:
                torch.save(checkpoint, 'best_model.pth')

def run_distributed_training(rank, world_size):
    """Función para ejecutar en cada proceso"""
    
    # Crear datasets distribuidos
    train_dataset = TensorDataset(
        torch.randn(10000, 3, 32, 32),
        torch.randint(0, 10, (10000,))
    )
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataset = TensorDataset(
        torch.randn(1000, 3, 32, 32),
        torch.randint(0, 10, (1000,))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Crear modelo
    model = ModernResNet(layers=[2, 2, 2, 2], num_classes=10)
    
    # Crear trainer distribuido
    trainer = DistributedTrainer(
        model, train_loader, val_loader, 10, rank, world_size
    )
    
    # Entrenar
    trainer.train(epochs=50)
    
    # Cleanup
    cleanup_distributed()

# Para ejecutar con torchrun:
# torchrun --nproc_per_node=4 script.py
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        # Multi-GPU training
        import multiprocessing as mp
        mp.spawn(run_distributed_training, args=(world_size,), nprocs=world_size)
    else:
        # Single GPU training
        run_training_example()
```

## Conclusiones PyTorch 2.0

Las mejoras clave incluyen:

🚀 **torch.compile()**: 20-50% speedup con zero code changes  
💾 **Memory efficiency**: Gradient checkpointing y mixed precision  
🏗️ **Modern architectures**: ViT, optimized ResNet, custom kernels  
📊 **Better training**: OneCycleLR, label smoothing, gradient clipping  
🌐 **Distributed training**: DDP simplificado y eficiente  
🔧 **Production ready**: TorchScript, ONNX export, quantization  

PyTorch 2.0 solidifica su posición como el framework de elección tanto para investigación como para producción en deep learning.

---
*¿Has migrado a PyTorch 2.0? Comparte tu experiencia con las nuevas características en los comentarios.*