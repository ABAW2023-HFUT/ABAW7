from torchvision import transforms

def aug1(output_image_size = 224):
    if output_image_size == 224:
        oversize = 256
    else:
        oversize = 128
    
    data_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.Resize((oversize, oversize)),
        transforms.RandomCrop(output_image_size),
        
    ])
    
    return data_aug
    
    
def aug2(output_image_size = 224):
    if output_image_size == 224:
        oversize = 256
    else:
        oversize = 128
    
    data_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.GaussianBlur(kernel_size=(23, 23), sigma=(0.15, 1.)),
        transforms.Resize((oversize, oversize)),
        transforms.RandomCrop(output_image_size),
    ])
    
    return data_aug


def aug3(output_image_size = 224):
    if output_image_size == 224:
        oversize = 256
    else:
        oversize = 128
    data_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((oversize, oversize)),
        transforms.RandomCrop(output_image_size),
    ])
    data_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((output_image_size, output_image_size)),
        # transforms.RandomCrop(output_image_size),
    ])
    return data_aug