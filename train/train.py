import torch
import torch.nn as nn

def train_model(model, dataloader, epochs=6, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(dataloader):.4f}")

    # Save Hugging Face style
    #model.save_pretrained("adult-detector-wavlm")
    torch.save(model.state_dict(), "adult-detector-wavlm.pt")
    print("Model saved to adult-detector-wavlm/")
