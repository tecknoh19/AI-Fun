# Lora trainer - Stable Diffusion
# This trainer works best when you have at least 150 images.  As of present I have not trained past 500 images so I am not sure what the max count is before over-fitting and other errors occur
# I use my image-to-text.py script found in this repo to generate teh captions for this training script.
# I hope you find this useful

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTokenizer
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms
from peft import LoraConfig, get_peft_model

torch.cuda.empty_cache()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, caption_dir, transform=None):
        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        caption_path = os.path.join(self.caption_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        with open(caption_path, 'r') as f:
            caption = f.read().strip()
        
        return image, caption


def train_lora(model_name, train_dataset, output_dir, num_epochs=5, batch_size=4, learning_rate=0.0005, continue_training=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Stable Diffusion Pipeline and extract components
    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    vae = pipe.vae.to(device)  # Variational Autoencoder
    unet = pipe.unet.to(device)  # U-Net
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = pipe.text_encoder.to(device)

    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["proj_in", "proj_out"], lora_dropout=0.1, bias="none"
    )
    unet = get_peft_model(unet, lora_config)

    # Continue from the last epoch if requested
    start_epoch = 0
    if continue_training:
        epoch_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith("lora_epoch_")], key=lambda x: int(x.split("_")[-1]))
        if epoch_dirs:
            last_epoch_dir = os.path.join(output_dir, epoch_dirs[-1])
            unet.load_adapter(last_epoch_dir, "default")  # Added adapter name 'default'
            start_epoch = int(epoch_dirs[-1].split("_")[-1])
            print(f"Continuing training from epoch {start_epoch}")

    # Adjust total number of epochs to include the ones already completed
    total_epochs = start_epoch + num_epochs

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, total_epochs):
        unet.train()
        running_loss = 0.0
        for i, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            captions = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

            optimizer.zero_grad()

            # Forward pass through the text encoder
            text_embeddings = text_encoder(captions)[0].to(device)

            # Convert RGB images to latent representations using the VAE encoder
            latents = vae.encode(images).latent_dist.sample() * 0.18215  # Latent scaling factor used by Stable Diffusion

            # Add noise to latent representations
            noise = torch.randn_like(latents).to(device)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()

            # Prepare noisy latent representations
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Pass the noisy latents and text embeddings to the U-Net
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Calculate loss (MSE between predicted and actual noise)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:  # Print every 10 batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10}")
                running_loss = 0.0

        print(f"Epoch {epoch + 1} complete. Saving LoRA weights...")
        output_path = os.path.join(output_dir, f"lora_epoch_{epoch + 1}")
        unet.save_pretrained(output_path)

    print("Training complete. LoRA weights saved to:", output_dir)



if __name__ == "__main__":
    import argparse

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train LoRA with images and captions.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with training images")
    parser.add_argument("--caption_dir", type=str, required=True, help="Directory with text captions")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained weights")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for training")
    parser.add_argument("--continue_training", action="store_true", help="Continue training from the last saved epoch")

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    # Create dataset and start training
    train_dataset = CustomDataset(args.image_dir, args.caption_dir, transform)
    train_lora("CompVis/stable-diffusion-v1-4", train_dataset, args.output_dir, args.num_epochs, args.batch_size, args.learning_rate, args.continue_training)
