import matplotlib.pyplot as plt
import torch


def visualize(model, dataloader, class_names=None, device="cuda", num_images=6):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, 6))
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    break
                plt.subplot(2, num_images//2, images_shown + 1)
                img = inputs[i].cpu().squeeze().numpy()
                plt.imshow(img, cmap="gray")
                title = f"Pred: {preds[i].item()}"
                if class_names:
                    title = f"Pred: {class_names[preds[i].item()]}"
                plt.title(title)
                plt.axis("off")
                images_shown += 1
            if images_shown >= num_images:
                break
    plt.tight_layout()
    plt.show()
