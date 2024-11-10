import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from itertools import combinations
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


# loss_dct = {
#     'channels_distinctive': channels_distinctive_loss,
#     'gradient_smoothness': calculate_gradient_smoothness_loss_2d,
#     'informative': informative_loss,
#     'background_regularization': background_regularization_loss,
#     'recon_l1': nn.L1Loss(),
#     'recon_l2': nn.MSELoss(),
#     'pyramid': Pyramid3dLoss(add_base_loss=True),
#     'pt': PyramidTemporalLoss(),
#     'gan': self.GAN,
#     'compensation': CompensationLoss(),
#     'kernel': calculate_gradient_smoothness_loss_2d,
# }
#!TODO: add perceptual loss and SSIM loss
# def channels_distinctive_loss(pred):
#     """ Calculate distinctiveness loss between channels of the generated tensor
#     Args:
#         pred: generated tensor, BxCxHxW
#     Returns:
#         loss: distinctiveness loss
#     """
#     B, C, H, W = pred.shape
#     loss = 0
#     for i in range(C-1):
#         for j in range(i + 1, C):
#             loss += F.mse_loss(pred[:,i], pred[:,j])

#     # Normalizing distinctiveness loss over the number of comparisons
#     num_channel_pairs = C * (C - 1) / 2
#     loss /= num_channel_pairs
#     return loss
def vgg_perceptual_loss(rgb_like, gray_scale, feature_layers=[3, 8, 15, 22], scales=[1, 2, 4]):
    """
    Calculate the VGG perceptual loss between an RGB-like image and two grayscale images.

    Parameters:
    - rgb_like (torch.Tensor): The RGB-like image tensor.
    - gray_scale (torch.Tensor): The grayscale image tensor with at least two channels.
    - feature_layers (list): Indices of VGG layers to be used for feature extraction.

    Returns:
    - loss (torch.Tensor): The computed perceptual loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the VGG16 model
    vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()

    # No gradients are required for the VGG parameters
    for param in vgg.parameters():
        param.requires_grad = False
    
    loss = 0.0
    rgb_like = rgb_like.to(device)
    gray_scale = gray_scale.to(device)
    
    for scale in scales:
        # Downsample images by the current scale factor
        if scale > 1:
            downsample = nn.AvgPool2d(scale)
            scaled_rgb_like = downsample(rgb_like)
            scaled_gray_1 = downsample(gray_scale[:,0,:,:].unsqueeze(1).repeat(1,3,1,1))
            scaled_gray_2 = downsample(gray_scale[:,1,:,:].unsqueeze(1).repeat(1,3,1,1))
        else:
            scaled_rgb_like = rgb_like
            scaled_gray_1 = gray_scale[:,0,:,:].unsqueeze(1).repeat(1,3,1,1)
            scaled_gray_2 = gray_scale[:,1,:,:].unsqueeze(1).repeat(1,3,1,1)

        # Compute the perceptual loss using specified layers for the current scale
        for layer_index in feature_layers:
            vgg_partial = nn.Sequential(*list(vgg[:layer_index + 1]))
            rgb_features = vgg_partial(scaled_rgb_like)
            gray_features_1 = vgg_partial(scaled_gray_1)
            gray_features_2 = vgg_partial(scaled_gray_2)
            
            # Calculate loss between RGB-like and grayscale images
            loss += torch.nn.functional.mse_loss(rgb_features, gray_features_1)
            loss += torch.nn.functional.mse_loss(rgb_features, gray_features_2)
    
    return loss

# def channel_loss(rgb):
#     """
#     Combined loss to ensure channels are not empty, distributions are similar, and activity is balanced.
    
#     Args:
#     images (torch.Tensor): The image tensor with shape (batch, channel, h, w).
#     alpha (float): Weight for the channel similarity loss component.
#     beta (float): Weight for the balanced channel activity loss component.
    
#     Returns:
#     torch.Tensor: The computed combined loss.
#     """
#     # Calculate the mean absolute activation for each channel
#     channel_means = torch.mean(torch.abs(rgb), dim=(0, 2, 3))
    
#     # Apply a penalty for low activation. The negative log of the channel means
#     # is used to heavily penalize channels that are close to zero.
#     channel_penalty = -torch.sum(torch.log(channel_means + 1e-8))


#     return channel_penalty

def channels_distinctive_loss(pred, data_range=1.0, size_average=True):
    # Ensure the tensor has the shape [batch, channels, height, width]
    # Get the number of channels
    num_channels = pred.size(1)

    # Initialize a list to store SSIM for each channel pair
    ssim_values = []

    # Calculate SSIM for each pair of channels
    for (i, j) in combinations(range(num_channels), 2):
        channel_i = pred[:, i, :, :].unsqueeze(1)
        channel_j = pred[:, j, :, :].unsqueeze(1)
        
        # Compute SSIM for the pair of channels and append to the list
        ssim_value = ssim(channel_i, channel_j, data_range=data_range, size_average=size_average)
        ssim_values.append(ssim_value)

    # Average the SSIM scores across all pairs
    avg_ssim = torch.mean(torch.stack(ssim_values))

    return avg_ssim

# def decorrelation_loss(rgb_like):
#     """
#     Calculate the decorrelation loss for a batch of features.
    
#     Args:
#     features (torch.Tensor): A tensor of shape (batch_size, num_channels, height, width)
#                              where each channel's features should be decorrelated from others.
                             
#     Returns:
#     torch.Tensor: A scalar tensor representing the decorrelation loss.
#     """
#     # Reshape features to (batch_size, num_channels, -1) to flatten height and width dimensions
#     batch_size, num_channels, height, width = rgb_like.shape
#     features_flat = rgb_like.view(batch_size, num_channels, -1)
    
#     # Compute the mean across the batches and subtract to center the features
#     mean = features_flat.mean(dim=2, keepdim=True)
#     features_centered = features_flat - mean
    
#     # Compute covariance matrix. Shape: (batch_size, num_channels, num_channels)
#     covariance_matrix = torch.bmm(features_centered, features_centered.transpose(1, 2)) / (height * width - 1)
    
#     # Zero out the diagonal elements and sum the absolute values of the off-diagonal elements
#     off_diagonal_elements = covariance_matrix - torch.diag_embed(torch.diagonal(covariance_matrix, dim1=-2, dim2=-1))
#     loss = off_diagonal_elements.abs().sum() / batch_size
    
#     return loss

#! Notice that this writing method may push all values to 0! consider make dx -> dx/(x0+eps)
def calculate_gradient_smoothness_loss_2d(event_volume):
    # Calculate gradients by finding the difference between adjacent pixels
    dx = torch.abs(event_volume[:, :, :, :-1] - event_volume[:, :, :, 1:])
    dy = torch.abs(event_volume[:, :, :-1, :] - event_volume[:, :, 1:, :])

    # Sum up the norms of differences to compute the smoothness loss
    gradient_dx_loss = dx.norm(p=2)
    gradient_dy_loss = dy.norm(p=2)

    # Combine the losses from x and y directions
    total_smoothness_loss = gradient_dx_loss + gradient_dy_loss 

    # Average the smoothness loss over all channels
    average_smoothness_loss = total_smoothness_loss / event_volume.shape[1]

    return average_smoothness_loss

# def calculate_gradient_smoothness_loss_2d(event_volume):
#     # Move the channel to the batch dimension
#     # [batch, channels, height, width] -> [batch * channels, height, width]
#     event_volume = event_volume.transpose(1, 0).reshape(-1, event_volume.shape[2], event_volume.shape[3])

#     # Calculate gradients by finding the difference between adjacent pixels
#     dx = torch.abs(event_volume[:, :, :-1] - event_volume[:, :, 1:])
#     dy = torch.abs(event_volume[:, :-1, :] - event_volume[:, 1:, :])

#     # Sum up the norms of differences to compute the smoothness loss for each pseudo-batch (which is actually each channel)
#     gradient_dx_loss = dx.norm(p=2, dim=[1, 2])  # Sum over height and width for each batch
#     gradient_dy_loss = dy.norm(p=2, dim=[1, 2])  # Sum over height and width for each batch

#     # Combine the losses from x and y directions
#     total_smoothness_loss = gradient_dx_loss + gradient_dy_loss

#     # Average the smoothness loss over all pseudo-batches (channels)
#     average_smoothness_loss = total_smoothness_loss.mean()

#     return average_smoothness_loss

# def informative_loss(event_volume):
#     """ Calculates the informative loss for each channel in the event volume.
#     Args:
#         event_volume (torch.Tensor): The event volume tensor of shape [batch, channel, height, width].
#     Returns:
#         loss: The calculated informative loss.
#     """
#     # Compute the mean and informative for each channel across spatial dimensions
#     # Keep dimensions for broadcasting the mean correctly
#     mean = event_volume.mean(dim=[2, 3], keepdim=True)
#     informatives = ((event_volume - mean) ** 2).mean(dim=[2, 3])
#     # Sum informatives across all channels and average over the batch
#     loss = -informatives.sum(dim=1).mean()
#     return loss

# def variance_loss(event_volume):

#     mean = event_volume.mean(dim=[1], keepdim=True)
#     variance = ((event_volume - mean) ** 2).mean(dim=[1])
#     # Sum informatives across all channels and average over the batch
#     loss = variance.sum(dim=[1, 2]).mean()
#     return loss


# # 可能在中後期才應該用。前期直接推向0可能不妥。另外該loss的weight應該逐漸變小。
# def background_regularization_loss(rgb):
#     # Identify pixels where the intensity is below a certain threshold
#     background_mask = (rgb.abs() < 0.1).float()
#     # Apply L1 regularization on the background (encouraging values to be zero)
#     reg_loss = torch.sum(torch.abs(rgb * background_mask))
#     return reg_loss

def calculate_loss(rgb_like, random_rgb, images, loss_strs, loss_weights, loss_functions, mode):
    loss_dict = {}
    loss = 0

    ## Voxel-Related Loss
    # Event Frame loss
    # if 'ef' in loss_strs:
    #     ef_gt = torch.sum(gt, dim=1)
    #     ef_pred = torch.sum(pred, dim=1)
    #     ef_loss = loss_functions['ef'](ef_pred, ef_gt)
        
    #     loss += loss_weights['alpha_ef'] * ef_loss
    #     loss_dict['ef_loss'] = ef_loss.detach()

    # # Pyramid Loss
    # if 'pyramid' in loss_strs:
    #     pyramid_loss = loss_functions['pyramid'](pred, gt)
        
    #     loss += loss_weights['alpha_pyramid'] * pyramid_loss
    #     loss_dict['pyramid_loss'] = pyramid_loss.detach()

    # # Pyramid Temporal Loss
    # if 'pt' in loss_strs:
    #     pt_loss =  loss_functions['pt'](pred, gt)
    #     loss += loss_weights['alpha_pt'] * pt_loss
    #     loss_dict['pt_loss'] = pt_loss.detach()
    
    # if 'compensation' in loss_strs:
    #     compensation_loss = loss_functions['compensation'](pred, gt)
    #     loss += loss_weights['alpha_compensation'] * compensation_loss
    #     loss_dict['compensation'] = compensation_loss.detach()

    # if 'gan' in loss_strs:
    #     gan_loss, gan_dis_loss = loss_functions['gan'](rgb_like, images)
    #     loss += loss_weights['alpha_gan'] * gan_loss
    #     loss_dict['gan_loss'] = gan_loss.detach()
    #     loss_dict['gan_dis_loss'] = gan_dis_loss

    if 'rgb_gan' in loss_strs:
        rgb_gan_loss, rgb_gan_dis_loss = loss_functions['rgb_gan'](rgb_like, random_rgb, mode)
        loss += loss_weights['alpha_rgb_gan'] * rgb_gan_loss
        loss_dict['rgb_gan_loss'] = rgb_gan_loss.detach()
        loss_dict['rgb_gan_dis_loss'] = rgb_gan_dis_loss

    if 'kernel' in loss_strs:
        kernel_loss=loss_functions['kernel'](rgb_like)
        loss += loss_weights['alpha_kernel'] * kernel_loss
        loss_dict['kernel_loss'] = kernel_loss.detach()

    # if 'informative' in loss_strs:
    #     informative_loss=loss_functions['informative'](rgb_like)
    #     loss += loss_weights['alpha_informative'] * informative_loss
    #     loss_dict['informative_loss'] = informative_loss.detach()

    # if 'recon_l1' in loss_strs:
    #     recon_loss_l1 = loss_functions['l1'](pred, gt)
    #     loss += loss_weights['alpha_recon_l1'] * recon_loss_l1
    #     loss_dict['recon_loss_l1'] = recon_loss_l1.detach()

    # if 'recon_l2' in loss_strs:
    #     recon_loss_l2 = loss_functions['l2'](pred, gt)
    #     loss += loss_weights['alpha_recon_l2'] * recon_loss_l2
    #     loss_dict['recon_loss_l2'] = recon_loss_l2.detach()
    
    # if 'background' in loss_strs:
    #     background_loss = loss_functions['background'](rgb_like)
    #     # alpha = loss_weights['alpha_background'] / 9000
    #     # if index <= 1000:
    #     #     background_weight = 0
    #     # else:
    #     #     background_weight = loss_weights['alpha_background'] - alpha * (index-1000)         
    #     # loss += background_weight * _background_loss
    #     loss += loss_weights['alpha_background'] * background_loss
    #     loss_dict['background_loss'] = background_loss.detach()

    if 'channels_distinctive' in loss_strs:
        channels_distinctive_loss = loss_functions['channels_distinctive'](rgb_like)
        loss += loss_weights['alpha_distinct'] * channels_distinctive_loss
        loss_dict['channels_distinctive_loss'] = channels_distinctive_loss.detach()
    
    # if 'channel_loss' in loss_strs:
    #     non_empty_loss = loss_functions['channel_loss'](rgb_like)
    #     loss += loss_weights['alpha_channel_nonempty'] * non_empty_loss
    #     loss_dict['nonempty_loss'] = non_empty_loss.detach()
    
    if 'perceptual' in loss_strs:
        perceptual_loss = loss_functions['perceptual'](rgb_like, images)
        loss += loss_weights['alpha_perceptual'] * perceptual_loss
        loss_dict['perceptual_loss'] = perceptual_loss.detach()
    
    # if 'decorrelation' in loss_strs:
    #     decorrelation_loss = loss_functions['decorrelation'](rgb_like)
    #     loss += loss_weights['alpha_decorrelation'] * decorrelation_loss
    #     loss_dict['decorrelation_loss'] = decorrelation_loss.detach()

    # if 'background_recon' in loss_strs:
    #     background_recon_loss = loss_functions['background'](pred)
    #     loss += loss_weights['alpha_background_recon'] * background_recon_loss
    #     loss_dict['background_recon_loss'] = background_recon_loss.detach()
        
    return loss, loss_dict
