import numpy as np

# Function to sort groups by size (largest first)
def sort_groups(groups, neighbors):
    """
    Sorts groups by their size in descending order.

    Parameters:
        groups (list of lists): Groups of connected pixels.
        neighbors (list of lists): Neighbors of the groups.

    Returns:
        new_groups (list of lists): Sorted groups.
        new_neighbors (list of lists): Sorted neighbors.
    """
    sizes = np.array([len(s) for s in groups])
    if len(sizes) > 1:
        new_groups = []
        new_neighbors = []
        for s in np.arange(np.max(sizes), np.min(sizes) - 1, -1):
            for i in np.where(sizes == s)[0]:
                new_groups.append(groups[i])
                new_neighbors.append(neighbors[i])
    else: 
        new_groups = groups
        new_neighbors = neighbors
    return new_groups, new_neighbors


class Topology:
    def __init__(self, image):
        """
        Initializes the topology class, processing the image and setting up pixel neighbors.
        
        Parameters:
            image (numpy array): Grayscale image array.
        """
        print('Running...')
        self.img = np.array(image)
        self.y, self.x = image.shape
        self.n_pixel_flat = np.arange(self.x * self.y)  # Assign a value to each pixel
        self.n_pixel_img = np.array(self.n_pixel_flat.reshape(self.y, self.x), dtype=int)

        # Identify boundary pixels
        self.limits = list(self.n_pixel_img[:, 0]) + list(self.n_pixel_img[:, -1]) + \
                      list(self.n_pixel_img[0, :]) + list(self.n_pixel_img[-1, :])
        
        self.max = np.max(self.img)

        # Add padding to handle border cases
        mat = np.full((self.y + 2, self.x + 2), -1)
        mat[1:-1, 1:-1] = self.n_pixel_img

        self.neighbors_0_pixels = []
        self.neighbors_1_pixels = []

        # Create neighbors for groups and holes
        for j in range(1, self.y + 1):
            for i in range(1, self.x + 1):
                # 8-connectivity for homology 0 groups
                mat_neighbors_0 = np.array([
                    mat[j-1, i-1], mat[j-1, i], mat[j-1, i+1],
                    mat[j, i-1], mat[j, i+1],
                    mat[j+1, i-1], mat[j+1, i], mat[j+1, i+1]
                ], dtype=int)
                mat_neighbors_0 = list(mat_neighbors_0[mat_neighbors_0 != -1])
                self.neighbors_0_pixels.append([int(n) for n in mat_neighbors_0])

                # 4-connectivity for homology 1 holes
                mat_neighbors_1 = np.array([
                    mat[j-1, i], mat[j, i-1],
                    mat[j, i+1], mat[j+1, i]
                ], dtype=int)
                mat_neighbors_1 = list(mat_neighbors_1[mat_neighbors_1 != -1])
                self.neighbors_1_pixels.append([int(n) for n in mat_neighbors_1])

        self.calculate_groups_0()
        self.calculate_groups_1()

    def calculate_groups_0(self):
        """Calculate homology 0 groups (connected components)."""
        self.groups_0 = []
        self.neighbors_0 = []
        self.lifetime_0 = []

        for level in range(0, self.max + 1):
            pixels = self.n_pixel_img[self.img == level]
            group_pixels, neighbor_pixels = self.connected_pixels(pixels, class_type=0)
            group_pixels, neighbor_pixels = sort_groups(group_pixels, neighbor_pixels)

            for i_g, group in enumerate(group_pixels):
                group_in_neighbors = []
                for i_v, neighbor in enumerate(self.neighbors_0):
                    for pixel in group:
                        if pixel in neighbor:
                            group_in_neighbors.append(i_v)
                            break

                if len(group_in_neighbors) == 0:
                    # Birth of a new group 0
                    self.groups_0.append(group)
                    self.neighbors_0.append(neighbor_pixels[i_g])
                    self.lifetime_0.append(list(np.zeros(self.max + 1, dtype=int)))
                else:
                    # Merging groups
                    _1st = group_in_neighbors[0]
                    self.groups_0[_1st] += group
                    self.neighbors_0[_1st] += neighbor_pixels[i_g]

                    for n_g in group_in_neighbors[1:]:
                        self.groups_0[_1st] += self.groups_0[n_g]
                        self.neighbors_0[_1st] += self.neighbors_0[n_g]
                        self.groups_0[n_g] = []  
                        self.neighbors_0[n_g] = []

            for i_g, group in enumerate(self.groups_0):
                if len(group) > 0:
                    self.lifetime_0[i_g][level] = 1

        # Store the lifetime of each group
        self.lifetime_0 = [[np.where(life)[0][0], np.where(life)[0][-1] + 1] for life in self.lifetime_0]

    def calculate_groups_1(self):
        """Calculate homology 1 groups (holes)."""
        self.groups_1 = []
        self.neighbors_1 = []
        self.lifetime_1 = []

        for level in range(self.max, -1, -1):
            pixels = self.n_pixel_img[(self.img > level) & (self.img < level + 2)]
            group_pixels, neighbor_pixels = self.connected_pixels(pixels, class_type=1)
            group_pixels, neighbor_pixels = sort_groups(group_pixels, neighbor_pixels)

            for i_g, group in enumerate(group_pixels):
                group_in_neighbors = []
                for i_v, neighbor in enumerate(self.neighbors_1):
                    for pixel in group:
                        if pixel in neighbor:
                            group_in_neighbors.append(i_v)
                            break

                if len(group_in_neighbors) == 0:
                    self.groups_1.append(group)
                    self.neighbors_1.append(neighbor_pixels[i_g])
                    self.lifetime_1.append(list(np.zeros(self.max + 1, dtype=int)))
                else:
                    _1st = group_in_neighbors[0]
                    self.groups_1[_1st] += group
                    self.neighbors_1[_1st] += neighbor_pixels[i_g]

                    for n_g in group_in_neighbors[1:]:
                        self.groups_1[_1st] += self.groups_1[n_g]
                        self.neighbors_1[_1st] += self.neighbors_1[n_g]
                        self.groups_1[n_g] = []
                        self.neighbors_1[n_g] = []

            for i_g, group in enumerate(self.groups_1):
                if len(group) > 0 and len([px for px in group if px in self.limits]) == 0:
                    self.lifetime_1[i_g][level] = 1

        self.lifetime_1 = [[np.where(life)[0][0], np.where(life)[0][-1] + 1] for life in self.lifetime_1 if 1 in life]

    def connected_pixels(self, pixels, class_type=1):
        """Find connected pixels (group formation)."""
        pixels = [int(p) for p in pixels]
        neighbors = [self.neighbors_0_pixels[p] if class_type == 0 else self.neighbors_1_pixels[p] for p in pixels]

        groups = []
        groups_neighbors = []

        while pixels:
            group = [pixels.pop(0)]
            neighbor = neighbors.pop(0)

            i = 0
            while i < len(pixels):
                if pixels[i] in neighbor:
                    group.append(pixels.pop(i))
                    neighbor += neighbors.pop(i)
                else:
                    i += 1

            groups.append(group)
            groups_neighbors.append(list(set(neighbor)))

        return groups, groups_neighbors

