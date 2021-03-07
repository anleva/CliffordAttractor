import numpy as np
from numba import jit
import os
import imageio


"""
Standard Clifford Attractor 
xn+1 = sin(a * yn) + c * cos(a xn)
yn+1 = sin(b * xn) + d * cos(b yn)

Generalized 3d Clifford Attractor
xn+1 = a_00 * sin(b_00 * xn + c_00 * pi) +  a_01 * sin(b_01 * yn + c_01 * pi) + a_02 * sin(b_02 * zn + c_02 * pi)
yn+1 = a_10 * sin(b_10 * xn + c_10 * pi) +  a_11 * sin(b_11 * yn + c_11 * pi) + a_12 * sin(b_12 * zn + c_12 * pi)
zn+1 = a_20 * sin(b_20 * xn + c_20 * pi) +  a_21 * sin(b_21 * yn + c_21 * pi) + a_22 * sin(b_22 * zn + c_22 * pi)
"""


@jit(nopython=True)
def sample_histogram_3d(alpha, beta, gamma, h, samples=50000., random_count=1000.):
    """
    :param alpha: NumPy array of size (n, n).
    :param beta: NumPy array of size (n, n).
    :param gamma: NumPy array of size (n, n).
    :param h: A 3 dimensional NumPy array where the Clifford Attractor density is estimated.
    :param samples: Number of samples. Defaults to 50000.
    :param random_count: Number of iterations between randomizing. Defaults to 1000.
    :return: h, an updated NumPy array.
    """

    dim = 3
    x = np.zeros((dim, 1)).astype(np.float32)
    x_amplitude = np.sum(np.abs(alpha), axis=1).reshape(x.shape)
    out_dim = np.array(h.shape).reshape([3, 1])

    # Split sample in batches if too big
    while samples > 1e9:
        h = sample_histogram_3d(alpha, beta, gamma, h, out_dim, 1e9, random_count)
        samples -= 1e9

    # Sample
    for i in range(samples):

        # Randomize slightly every random_count iterations
        if i % random_count == 0:
            print(np.round(100 * i / samples))
            for j in range(dim):
                x[j, 0] += 0.1 * np.random.normal()

        # Update x
        x_sin = np.sin(beta + gamma * x.T)
        x = np.sum(alpha * x_sin, axis=1).reshape(x.shape)

        # Get bucket for x
        bx = np.floor(out_dim * (0.03 + 0.94 * (x_amplitude + x) / (2 * x_amplitude))).astype(np.int32)
        bx_tuple = (int(bx[0][0]), int(bx[1][0]), int(bx[2][0]))

        # Update histogram's bucket
        h[bx_tuple] += 1

    return h


def build_bw_image(h, file_path, clear=0.1, exp=0.5, cap_pct=99.5, start=0, end=0.9):
    # Clear near-empty buckets, scale to 100
    raw = np.copy(h).astype(np.float32)
    raw_mean = np.mean(h)
    raw[raw < clear * raw_mean] = 0
    # raw = cv2.GaussianBlur(raw, (blur, blur), 0)
    raw_max = np.max(raw)
    raw = raw * (100 / raw_max)

    # Cap, soften and interpolate
    img = raw ** exp
    cap = np.percentile(img, cap_pct)
    img[img > cap] = cap
    multiplier = 1 / cap
    img = start + (multiplier * (end - start)) * img

    img = np.round(255 * (1 - img)).astype('uint8')
    img2 = np.zeros((img.shape[0], img.shape[0], 3))
    img2[:, :, 0] = img
    img2[:, :, 1] = img
    img2[:, :, 2] = img

    imageio.imwrite(file_path, img)


if __name__ == "__main__":

    print('INITIATING...')

    alpha = np.array([
        [0.3, 0.7, 0.2],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0]
    ]).astype(np.float32)

    beta = np.pi * np.array([
        [0.0, 0.5, 1.0],
        [0.5, 1.0, 0.0],
        [1.0, 0.0, 0.5],
    ]).astype(np.float32)

    gamma = np.array([
        [-1.4, -1.4, -1.4],
        [0.7, 0.7, 0.7],
        [2.8, 2.8, 2.8]
    ]).astype(np.float32)

    h = np.zeros((600, 600, 600)).astype(np.float32)
    out_dim = np.array(h.shape).reshape([3, 1])

    print('SAMPLING...')
    h = sample_histogram_3d(alpha, beta, gamma, h, out_dim, samples=1e9)

    print('BUILDING IMAGES')

    h_0 = np.sum(h, axis=0)
    h_1 = np.sum(h, axis=1)
    h_2 = np.sum(h, axis=2)

    output_folder = r'output'
    sample_run = 5

    build_bw_image(h_0, os.path.join(output_folder, 'clifford_3d_smoke_{}_0.jpg'.format(sample_run)))
    build_bw_image(h_1, os.path.join(output_folder, 'clifford_3d_smoke_{}_1.jpg'.format(sample_run)))
    build_bw_image(h_2, os.path.join(output_folder, 'clifford_3d_smoke_{}_2.jpg'.format(sample_run)))

