import torch
import torch.fft


def augment_patches_freq_domain(patches, n=1, noise_level=0.01):
    """
    Augment time series patches by adding noise in the frequency domain, keeping the frequency domain stable.

    Parameters:
    patches (torch.Tensor): Input tensor of shape (batch_size, num_patch, nvars, patch_len)
    n (int): The factor by which to augment the number of patches.
    noise_level (float): The standard deviation of the Gaussian noise to add in the frequency domain.

    Returns:
    torch.Tensor: Augmented tensor with shape (batch_size, num_patch * n, nvars, patch_len)
    """

    # Repeat patches n times
    augmented_patches = patches.repeat (1, n, 1, 1)

    # Perform FFT to convert to frequency domain
    freq_domain_patches = torch.fft.fft (augmented_patches, dim=-1)

    # Generate noise and add to the frequency domain
    noise = torch.randn_like (freq_domain_patches) * noise_level
    freq_domain_patches += noise

    # Perform inverse FFT to convert back to time domain
    augmented_patches = torch.fft.ifft (freq_domain_patches, dim=-1).real

    return augmented_patches