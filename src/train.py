import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model import SRCNN
from dataset import CustomDataset
from config import config, device
from utils import visualize_results

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % 100 == 0:
            print(f"Batch {batch}: Loss = {loss.item():.6f}")
    
    # Return the average loss for the epoch
    return running_loss / len(dataloader)

def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    
    avg_test_loss = test_loss / len(dataloader)
    print(f"Test Loss: {avg_test_loss:.6f}")
    return avg_test_loss

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

    # Track the best loss and initialize it to a large value
    best_loss = float('inf')

    # Training loop
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")

        # Train and get the average training loss
        train_loss = train(train_dataloader, model, loss_fn, optimizer)

        # Only proceed if the current train loss is better than the best recorded loss
        if train_loss < best_loss:

            # Test and get the average test loss
            test_loss = test(train_dataloader, model, loss_fn)

            # Update the best loss and save the model
            best_loss = train_loss
            torch.save(model.state_dict(), config['save_path'])
        else:
            print(f"Train Loss {train_loss:.6f} is higher than best loss {best_loss:.6f}, skipping model update.")

if __name__ == '__main__':
    main()