import torchvision.transforms as transforms
from model import SegAE
from dataset import Clip_Rescale, crop_slices

with open('label_embedding.pkl', 'rb') as file:
    embedding_dict = pickle.load(file)

transform_ct = transforms.Compose([
        Clip_Rescale(min_val=-200, max_val=200),
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
])

transform_mask = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

model = SegAE(hidden_dim=50, backbone=model_name, embedding='text_embedding')
model.load_state_dict(torch.load("best_resnet50_model_40_samples.pth"))
model.to(device)
model.eval()

def segae_inference(ct_data, mask_this_class, _class):
    slice_dices = []

    valid_slices = np.where(np.any(mask_this_class > 0, axis=(1, 2)))[0]

    if len(valid_slices) > 10:
        valid_slices = [valid_slices[int(len(valid_slices)/10*idx)] for idx in range(10)]
    
    if len(valid_slices) == 0:
        print('0 slice in mask.')
        return 0

    for slice_idx in valid_slices:
            crop_start = time.time()
            ct_slice, pred_mask_slice = crop_slices(
                [ct_data[slice_idx, :, :], mask_this_class[slice_idx, :, :]],
                mask_this_class[slice_idx, :, :]
            )
            crop_time += time.time() - crop_start
            
            aug_start = time.time()
            ct_slice = transform_ct(ct_slice).unsqueeze(0)
            pred_mask_slice = transform_mask(pred_mask_slice).unsqueeze(0)
            aug_time += time.time() - aug_start
            text_embedding = embedding_dict[_class]
            
            image_tensor = torch.cat((ct_slice, pred_mask_slice), dim=1).to(device)
            embedding_tensor = text_embedding.to(device)
            
            GPU_start = time.time()
            predicted_dice = model(image_tensor, embedding_tensor)
            GPU_time += time.time() - GPU_start
            
            slice_dices.append(predicted_dice.detach().cpu().item())

    return np.mean(slice_dices)