import torch

# calculate the magnitude of multiple vectors
def magnitude(coordinates):
    return torch.sqrt(torch.sum(coordinates ** 2, dim=-1)).view(coordinates.shape[0], 1)
# normalize the length of the vector, so that the magnitude of each vector is 1
def normalized(coordinates):
    mag = magnitude(coordinates)
    return coordinates / mag
# calculate the projection of the vector
def component_parallel_to(coordinates, basis):
    u = normalized(basis)
    weight = torch.sum(coordinates * u, dim=-1).view(coordinates.shape[0], 1)
    return u * weight
# calculate the orthogonal vector
def component_orthogonal_to(coordinates, basis):
    projection = component_parallel_to(coordinates, basis)
    return coordinates - projection

def project_algorithm(original_feature, trivial_feature):
    d = component_orthogonal_to(original_feature, trivial_feature)
    f = component_parallel_to(original_feature, d)
    return f