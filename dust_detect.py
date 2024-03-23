import numpy as np


def detect_blur_fft(image, size=60, thresh=10):
    """
    dust is treated as a blured image
    :param image:
    :param size: size of the square filter window to remove the center fgs from the FT image
    :param thresh: to determine whether an image is blurry or not.
    :return:
    """
    (h, w) = image.shape
    (cX, cY) = (w // 2, h // 2)
    # cropped before performing the FFT to reduce the size of fft computation
    # so the entire image is not passed though the fft
    cropped_image = image[cY - size:cY + size, cX - size:cX + size]
    fft = np.fft.fft2(cropped_image)
    fftShift = np.fft.fftshift(fft)
    # zeroing out a smaller region within the shifted FFT spectrum, removing noise
    fftShift[size - 10:size + 10, size - 10:size + 10] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    m = np.mean(magnitude)

    return m, m <= thresh
