import numpy as np
from PIL import Image


def read_image(file_name):
    im = np.asarray(Image.open(file_name))
    return im[:, :, 0:3]


def initialize_means0(img,
                      num_clusters_radical3):  # num of clusters is considered sth to the power of 3 for simplicity.
    num_clusters = num_clusters_radical3 ** 3
    points = img.reshape((-1, img.shape[2]))
    means = []
    step = 255 // num_clusters_radical3
    for r in range(num_clusters_radical3):
        for g in range(num_clusters_radical3):
            for b in range(num_clusters_radical3):
                mean = [r * step, g * step, b * step]
                means.append(mean)
    means = np.asarray(means)
    return points, means


def initialize_means1(img, clusters):
    points = img.reshape((-1, img.shape[2]))
    m, n = points.shape
    means = np.zeros((clusters, n))
    for i in range(clusters):
        rand_indices = np.random.choice(m, size=10, replace=False)
        means[i] = np.mean(points[rand_indices], axis=0)

    return points, means


def distance(point1, point2):
    dist = np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]) + np.square(point1[2] - point2[2])
    dist = np.sqrt(dist)
    return dist


def k_means(points, means, num_clusters):
    m, n = points.shape
    index = np.zeros(m)
    old_means = np.copy(means)
    while True:
        for j in range(m):
            min_dist = float('inf')
            for k in range(num_clusters):
                if distance(points[j], means[k]) <= min_dist:
                    min_dist = distance(points[j], means[k])
                    index[j] = k

        for k in range(num_clusters):
            cluster_points = points[index == k]
            if len(cluster_points) > 0:
                means[k] = np.mean(cluster_points, axis=0)

        max_diff = np.max(np.abs(old_means - means))
        print(max_diff)
        if max_diff < 5:
            break
        old_means = np.copy(means)

    return means, index


def save_recovered_image(means, index, img_shape):
    recovered = means[index.astype(int), :]
    # getting back the 3d matrix (row, col, rgb(3))
    recovered = recovered.reshape(img_shape)
    recovered = recovered.astype(np.uint8)
    im = Image.fromarray(recovered)
    im.save("cluster_compressed1.png")


im = read_image("image.jpg")
point, means = initialize_means1(im, 10)
means, index = k_means(point, means,
                       )  # index is the compressed data to be saved. but inorder to be able to show the image, we save the recovered image.
save_recovered_image(means, index, im.shape)
