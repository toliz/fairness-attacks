import numpy as np
import cvxpy as cvx
import torch

from datamodules import Dataset
from torch import Tensor
from typing import Union


def project_point(point: Tensor, point_class: int, beta: dict, minimization_problem: cvx.Problem) -> Tensor:
    """
    Project point onto feasible set
    :param point: Point to project
    :param beta: Defense parameters
    :param minimization_problem: Minimization problem
    :param point_class: Class of point
    :return: Projected point
    """
    # assert all(x'sphere_radii', 'slab_radii', 'centroids', 'centroid_vec') in \
    #           beta.keys(), "['sphere_radii', 'slab_radii', 'centroids', 'centroid_vec'] not in beta"
    point = point.detach().clone()
    sphere_radii = beta['sphere_radii'][point_class]
    slab_radii = beta['slab_radii'][point_class]
    center = beta['centroids'][point_class]
    centroid_vec = beta['centroid_vec']
    # Assign the value of the parameters
    # cvxpy shenanigans to set the value of the parameters
    parameters = minimization_problem.parameters()
    param_index_map = {
        v: k
        for k, v in dict(enumerate(map(lambda l: l.name(),
                                       parameters))).items()
    }
    # assert ('sphere_radius', 'slab_radius', 'center', 'centroid_vec', 'x_bar') in param_index_map.keys(), \
    #     "['sphere_radius', 'slab_radius', 'center', 'centroid_vec', 'x_bar'] params not in the minimization problem"
    # cvxpy shenanigans to get the optimal value of the variable
    variables = minimization_problem.variables()
    variable_index_map = {
        v: k
        for k, v in dict(enumerate(map(lambda l: l.name(),
                                       variables))).items()
    }
    # assert ('x',) in variable_index_map.keys(), \
    #     "['x'] variable not in the minimization problem"
    minimization_problem.parameters()[
        param_index_map['sphere_radius']].value = np.array([sphere_radii])
    minimization_problem.parameters()[
        param_index_map['slab_radius']].value = np.array([slab_radii])
    minimization_problem.parameters()[param_index_map['center']].value = center
    minimization_problem.parameters()[
        param_index_map['centroid_vec']].value = centroid_vec
    minimization_problem.parameters()[
        param_index_map['x_bar']].value = np.array(point)
    # Solve the minimization problem
    minimization_problem.solve()
    # Get the projected point
    projected_point = minimization_problem.variables()[
        variable_index_map['x']].value

    return Tensor(projected_point)


def project_dataset(dataset: Dataset, beta: dict, minimization_problem: cvx.Problem) -> Dataset:
    num_features = dataset.X.shape[1]

    XY = torch.hstack((dataset.X, dataset.Y.unsqueeze(1)))
    unique_XY, inverse_map = torch.unique(XY, return_inverse=True, dim=0)

    unique_X_proj = torch.empty((len(unique_XY), num_features))
    for i, xy in enumerate(unique_XY):
        x, y = torch.split(xy, num_features)
        unique_X_proj[i] = project_point(x, int(y), beta, minimization_problem)

    X_proj = torch.empty_like(dataset.X, dtype=torch.float)
    assert isinstance(X_proj, torch.Tensor)

    for i, idx in enumerate(inverse_map):
        X_proj[i] = unique_X_proj[idx]

    return Dataset(X_proj, dataset.Y, dataset.adv_mask)

def cvx_dot(a: Union[cvx.Parameter, cvx.Variable],
            b: Union[cvx.Parameter, cvx.Variable]) -> cvx.Variable:
    """
    Returns the dot product of two variables (maybe? I don't know)
    :param a: First variable
    :param b: Second variable
    :return: Dot product of the two variables
    """
    return cvx.sum(cvx.multiply(a, b))


def get_minimization_problem(dataset: Dataset) -> cvx.Problem:
    """
    Build a minimization problem for projecting points onto the feasible set
    :param dataset: Dataset
    :return: Minimization problem
    """
    if type(dataset.X) is torch.Tensor:
        X = dataset.X.detach().clone().numpy()
    else:
        X = dataset.X
    # Build the minimization problem
    """
    Quick recap of cvxpy:
    - cvx.Variable(shape): The variable to be minimized.
    The shape is the shape of the variable.
    - cvx.Parameter(shape): Parameters of the problem.
    Initializing these objects doesn't assign any values to them.
    - cvx.Parameter.value: The value of the parameter.
    We can assign this later.
    - cvx.Minimize: The objective function.
    - cvx.Problem: The problem to be solved.
    - cvx.Problem.solve(): Solves the problem.
    Assigns to all cvx.Variables of the problem
    their optimal value if problem.status == 'optimal'.
    """
    num_features = X.shape[1]

    # cvx_x is the vector of the point to project
    cvx_x = cvx.Variable(num_features, name='x')

    # cvx_x_bar is the point we want to project onto the feasible set
    # We want the closest point to cvx_x_bar that is feasible
    cvx_x_bar = cvx.Parameter(num_features, name='x_bar')

    #cvx_center is the center of the sphere
    cvx_center = cvx.Parameter(num_features, name='center')

    #cvx_radius is the radius of the sphere
    cvx_sphere_radius = cvx.Parameter(1, name='sphere_radius')

    #cvx_slab_radius is the radius of the slab
    cvx_slab_radius = cvx.Parameter(1, name='slab_radius')

    #cvx_centroid_vec is the vector of the centroid
    cvx_centroid_vec = cvx.Parameter(num_features, name='centroid_vec')

    cvx_x_signed_dist_from_center = cvx_x - cvx_center

    # Our objective is to minimize the distance between the point to project and the point we want to project onto the feasible set
    cvx_objective = cvx.Minimize(cvx.pnorm(cvx_x - cvx_x_bar, 2)**2)

    # We want to make sure that the point to project is feasible
    # By feasible, we mean we want it to conform to the sphere and slab
    # constraints, plus use the LP relaxation technique to make sure after
    # rounding, the point is still feasible
    cvx_constraints = []

    # For the LP relaxation technique, we need to find the maximum
    # values of the features in the clean dataset
    X_max = np.max(X, axis=0).reshape(-1)
    # We set the lowest maximum value of the features to 1
    X_max[X_max < 1] = 1
    # We set the highest maximum value of the features to 50
    X_max[X_max > 50] = 50
    # We add the constraint that our projected point has to have features
    # within the range of the features in the clean dataset
    cvx_constraints.append(cvx_x <= X_max)

    # Additionally, we adaptively add constraints for each feature
    # as described in https://arxiv.org/pdf/1811.00741.pdf section 3.3
    k_max = int(np.ceil(np.max(X_max)))
    # Initialize the expected value E||x_bar||^2
    cvx_expected_value = cvx.Variable(num_features, name='expected_value')

    for k in range(1, k_max + 1):
        # Create mask for features with max value k
        X_k_max = k <= X_max
        if any(X_k_max):
            cvx_constraints.append(
                cvx_expected_value[X_k_max] >= cvx_x[X_k_max] *
                (2 * k - 1) - k * (k - 1))

    # Append the constraint that the expected value of the distance
    # between the projected point and the center of the sphere is less than
    # the radius of the sphere
    cvx_constraints.append(
        cvx.sum(cvx_expected_value) - 2 * cvx_dot(cvx_center, cvx_x) + \
            cvx.sum_squares(cvx_center) <= cvx_sphere_radius ** 2
    )

    # Append the slab constraints
    # TODO: test if instead of these two we could instead have
    # cvx_x_dist_from_center = cvx.pnorm(cvx_x - cvx_center, 2)
    # and cvx_dot(cvx_x_dist_from_center, cvx_centroid_vec) <= cvx_slab_radius

    cvx_constraints.append(
        cvx_dot(cvx_centroid_vec, cvx_x_signed_dist_from_center) <=
        cvx_slab_radius)
    cvx_constraints.append(
        -cvx_dot(cvx_centroid_vec, cvx_x_signed_dist_from_center) <=
        cvx_slab_radius)

    problem = cvx.Problem(cvx_objective, cvx_constraints)
    # This is still an abstract problem. We assign values to the parameters later
    # depending on the point (which class it belongs to) before solving it.
    return problem


def defense(dataset: Dataset, beta: dict) -> Dataset:
    """
    Prunes the dataset according to the feasible set.
    :param dataset: Dataset (D_c \cup D_p)
    :param beta: Dictionary of beta values for the feasible set
    :return: Pruned dataset = (D_c \cup D_p) \cap F_b
    """
    if type(dataset.X) is torch.Tensor:
        X, y = dataset.X.detach().clone().numpy(), dataset.Y.detach().clone().numpy()
    else:
        X, y = dataset.X, dataset.Y

    classes = set(list(y))
    sphere_radii = beta['sphere_radii']
    slab_radii = beta['slab_radii']
    centroids = beta['centroids']
    centroid_vec = beta['centroid_vec']
    masks = {}
    for c in classes:
        center = centroids[c]
        sphere_radius = sphere_radii[c]
        slab_radius = slab_radii[c]
        shifts_from_center = X - center
        dists_from_center = np.linalg.norm(shifts_from_center, axis=1)
        dists_from_slab = np.abs(X @ centroid_vec - centroids[c] @ centroid_vec)
        # Prune points that are too far from the sphere center
        masks[c] = (dists_from_center < sphere_radius)
        # Prune points that are too far from the slab
        masks[c] &= (dists_from_slab < slab_radius)
        masks[c] &= y == c

    # Create mask applying logical OR to all masks
    mask = np.logical_or.reduce(list(masks.values()))
    # Calculate new advantaged mask
    old_adv_mask = dataset.adv_mask.detach().clone().numpy()
    new_adv_mask = old_adv_mask[mask]
    # Create new dataset
    new_dataset = Dataset(X[mask], y[mask], new_adv_mask)
    return new_dataset


def get_defense_params(dataset: Dataset) -> dict:
    """
    Get the parameters for the defense
    :param dataset: Dataset
    :return: dictionary of parameters
    """
    PERCENTILE = 90
    if type(dataset.X) is torch.Tensor:
        X, y = dataset.X.detach().clone().numpy(), dataset.Y.detach().clone().numpy()
    else:
        X, y = dataset.X, dataset.Y
    classes = set(list(y))
    centroids = get_centroids(dataset=dataset)
    centroid_vec = get_centroid_vec(centroids=centroids)
    sphere_radii = dict()
    slab_radii = dict()
    for c in classes:
        center = centroids[c]
        shifts_from_center = X[y == c] - center
        dists_from_center = np.linalg.norm(shifts_from_center, axis=1)
        sphere_radii[c] = np.percentile(dists_from_center, PERCENTILE)
        dists_from_slab = np.abs(X[y == c] @ centroid_vec -
                                 centroids[c] @ centroid_vec)
        slab_radii[c] = np.percentile(dists_from_slab, PERCENTILE)
    return {
        'sphere_radii': sphere_radii,
        'slab_radii': slab_radii,
        'centroids': centroids,
        'centroid_vec': centroid_vec
    }


def get_centroids(dataset: Dataset) -> dict:
    """
    Returns the centroids of the training data
    :param dataset: Dataset
    :return: dictionary of centroids with class as key
    and centroid as value
    """
    if type(dataset.X) is torch.Tensor:
        X, y = dataset.X.detach().clone().numpy(), dataset.Y.detach().clone().numpy()
    else:
        X, y = dataset.X, dataset.Y
    classes = set(list(y))
    centroids = dict()
    for c in classes:
        centroids[c] = np.mean(X[y == c], axis=0)
    return centroids


def get_centroid_vec(centroids: dict) -> np.ndarray:
    """
    Returns the centroid vector of the dataset
    :param centroids: dictionary of centroids with class as key
    and centroid as value
    :return: centroid vector
    """
    centroids_vec = centroids[1] - centroids[0]
    #Normalize the centroid vector
    centroids_vec /= np.linalg.norm(centroids_vec)
    centroids_vec.reshape(1, -1)
    return centroids_vec
