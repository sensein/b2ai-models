import torch
import torch.nn as nn


def compute_pos_weight(dataloader, device="cuda"):
    pos = 0
    total = 0
    for _, y in dataloader:
        y = y.float()
        pos += y.sum().item()
        total += y.numel()
    neg = total - pos
    return torch.tensor(neg / pos, device=device)


def train_model(model, dataloader, epochs=6, lr=1e-4, device="cuda", save_path="adult-detector-wavlm.pt"):
    model.to(device)


    pos_weight = compute_pos_weight(dataloader, device)
    print("Computed pos_weight:", pos_weight.item())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )


    model.eval()
    x_batch, _ = next(iter(dataloader))
    x_batch = x_batch.to(device)
    with torch.no_grad():
        logits = model(x_batch)
        print("BEFORE training | mean:", logits.mean().item(), " std:", logits.std().item())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # debug after epoch

        model.eval()
        x_batch, _ = next(iter(dataloader))
        x_batch = x_batch.to(device)
        with torch.no_grad():
            logits = model(x_batch)
            print(
                f"Epoch {epoch+1} | Loss={avg_loss:.4f} | logit_mean={logits.mean().item():.3f} | logit_std={logits.std().item():.3f}"
            )

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
