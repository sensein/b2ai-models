import torch
import torch.nn as nn

def train_model(model, dataloader, epochs=3, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(dataloader):.4f}")

    # Save Hugging Face style
    model.save_pretrained("adult-detector-wavlm")
    print("Model saved to adult-detector-wavlm/")
