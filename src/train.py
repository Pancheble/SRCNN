import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model import SRCNN
from dataset import CustomDataset
from config import config, device
from utils import visualize_results

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Batch {batch}: Loss = {loss.item():.6f}")

def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    print(f"Test Loss: {test_loss/len(dataloader):.6f}")

def main():
    # Load dataset
    train_dataset = CustomDataset(
        img_paths=config['img_paths'], 
        input_size=config['input_size'], 
        output_size=config['input_size'] - config['f1'] - config['f2'] - config['f3'] + 3, 
        stride=config['stride'], 
        upscale_factor=config['upscale_factor']
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Load model
    model = SRCNN(kernel_list=[config['f1'], config['f2'], config['f3']],
                  filters_list=[config['n1'], config['n2'], config['n3']]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        train(train_dataloader, model, loss_fn, optimizer)
        test(train_dataloader, model, loss_fn)

    # Save the model
    torch.save(model.state_dict(), config['save_path'])
    print(f"Model saved to {config['save_path']}")

if __name__ == '__main__':
    main()
