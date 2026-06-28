def downsample_steps(image_size: int, latent_size: int) -> int:
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if latent_size <= 0:
        raise ValueError("latent_size must be positive.")
    if image_size <= latent_size:
        raise ValueError("image_size must be greater than latent_size.")
    if image_size % latent_size != 0:
        raise ValueError("image_size must be divisible by latent_size.")

    factor = image_size // latent_size
    if factor & (factor - 1) != 0:
        raise ValueError("image_size / latent_size must be a power of two.")
    return factor.bit_length() - 1
