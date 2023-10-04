from __future__ import division
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree, distance
import torch
from svgpath2mpl import parse_path
import xml.etree.ElementTree as etree
import requests
from io import StringIO
import pickle
import os
from pymoo.indicators.hv import HV
import csv
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm.autonotebook import trange
from pymoo.util.display.display import Display
import pymoo
import numpy as np
from lib2to3.pgen2 import driver
from math import sin, cos, acos, pi
import warnings
warnings.filterwarnings("ignore")


def get_oriented(curve):
    """ Orient a curve
    Parameters
    ----------
    curve:  numpy array [n_points,2]
            The curve to be oriented
    Returns
    -------
    out:    numpy array [n_points,2]
            The oriented curve
    """

    ds = squareform(pdist(curve))
    pi, t = np.unravel_index(np.argmax(ds), ds.shape)

    d = curve[pi] - curve[t]

    if d[1] == 0:
        theta = 0
    else:
        d = d * np.sign(d[1])
        theta = -np.arctan(d[1]/d[0])

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    out = np.matmul(rot, curve.T).T
    out = out - np.min(out, 0)

    rot2 = np.array([[np.cos(theta+np.pi), -np.sin(theta+np.pi)],
                    [np.sin(theta+np.pi), np.cos(theta+np.pi)]])
    out2 = np.matmul(rot2, curve.T).T
    out2 = out2 - np.min(out2, 0)

    if out2[:, 0].sum() > out[:, 0].sum():
        return out2

    return out


def get_oriented_both(curve):
    """ Orient a curve
    Parameters
    ----------
    curve:  numpy array [n_points,2]
            The curve to be oriented
    Returns
    -------
    out:    numpy array [n_points,2]
            The oriented curve
    """

    ds = squareform(pdist(curve))
    pi, t = np.unravel_index(np.argmax(ds), ds.shape)

    d = curve[pi] - curve[t]

    if d[1] == 0:
        theta = 0
    else:
        d = d * np.sign(d[1])
        theta = -np.arctan(d[1]/d[0])

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    out = np.matmul(rot, curve.T).T
    out = out - np.min(out, 0)

    rot2 = np.array([[np.cos(theta+np.pi), -np.sin(theta+np.pi)],
                    [np.sin(theta+np.pi), np.cos(theta+np.pi)]])
    out2 = np.matmul(rot2, curve.T).T
    out2 = out2 - np.min(out2, 0)

    return out, out2


def get_oriented_angle(curve):
    """ Calculate the orientation of a curve
    Parameters
    ----------
    curve:  numpy array [n_points,2]
            The curve to be oriented
    Returns
    -------
    theta:  float
            The orientation of the curve
    """

    ds = squareform(pdist(curve))
    pi, t = np.unravel_index(np.argmax(ds), ds.shape)

    d = curve[pi] - curve[t]

    if d[1] == 0:
        theta = 0
    else:
        d = d * np.sign(d[1])
        theta = -np.arctan(d[1]/d[0])

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    out = np.matmul(rot, curve.T).T
    out = out - np.min(out, 0)

    rot2 = np.array([[np.cos(theta+np.pi), -np.sin(theta+np.pi)],
                    [np.sin(theta+np.pi), np.cos(theta+np.pi)]])
    out2 = np.matmul(rot2, curve.T).T
    out2 = out2 - np.min(out2, 0)

    if out2[:, 0].sum() > out[:, 0].sum():
        return theta+np.pi

    return theta


def solve_rev_vectorized_batch_wds(As, x0s, node_types, thetas):
    """ Solve a vectorized batch of mechanisms
    Parameters
    ----------
    As:     torch tensor [batch_size,N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0s:    torch tensor [batch_size,N,2]
            The initial positions of the nodes
    node_types: torch tensor [batch_size,N,1]
            A list of the nodes that are grounded/fixed.
    thetas: torch tensor [n_thetas]
            The angles to solve the mechanism at.
    Returns
    -------
    x:     torch tensor [batch_size,N,n_thetas,2]
            The positions of the nodes at each angle.

    """
    Gs = torch.cdist(x0s, x0s)

    x = torch.zeros([x0s.shape[0], x0s.shape[1],
                    thetas.shape[0], 2]).to(As.device)

    x = x + torch.unsqueeze(node_types * x0s, 2)

    m = x[:, 0] + torch.tile(torch.unsqueeze(torch.transpose(torch.cat([torch.unsqueeze(torch.cos(thetas), 0), torch.unsqueeze(
        torch.sin(thetas), 0)], 0), 0, 1), 0), [x0s.shape[0], 1, 1]) * torch.unsqueeze(torch.unsqueeze(Gs[:, 0, 1], -1), -1)

    x[:, 1, :, :] = m

    cos_list = []

    for k in range(3, x0s.shape[1]):
        inds = torch.argsort(As[:, k, 0:k])[:, -2:].numpy()

        l_ijs = torch.linalg.norm(x[np.arange(
            x0s.shape[0]), inds[:, 0]] - x[np.arange(x0s.shape[0]), inds[:, 1]], dim=-1)

        gik = torch.unsqueeze(
            Gs[np.arange(x0s.shape[0]), inds[:, 0], np.ones(shape=[x0s.shape[0]])*k], -1)
        gjk = torch.unsqueeze(
            Gs[np.arange(x0s.shape[0]), inds[:, 1], np.ones(shape=[x0s.shape[0]])*k], -1)

        cosphis = (torch.square(l_ijs) + torch.square(gik) -
                   torch.square(gjk))/(2 * l_ijs * gik)

        cosphis = torch.where(torch.tile(node_types[:, k], [
                              1, thetas.shape[0]]) == 0.0, cosphis, torch.zeros_like(cosphis))

        cos_list.append(cosphis.unsqueeze(1))

        x0i1 = x0s[np.arange(x0s.shape[0]), inds[:, 0],
                   np.ones(shape=[x0s.shape[0]]).astype(int)]
        x0i0 = x0s[np.arange(x0s.shape[0]), inds[:, 0],
                   np.zeros(shape=[x0s.shape[0]]).astype(int)]

        x0j1 = x0s[np.arange(x0s.shape[0]), inds[:, 1],
                   np.ones(shape=[x0s.shape[0]]).astype(int)]
        x0j0 = x0s[np.arange(x0s.shape[0]), inds[:, 1],
                   np.zeros(shape=[x0s.shape[0]]).astype(int)]

        x0k1 = x0s[:, k, 1]
        x0k0 = x0s[:, k, 0]

        s = torch.unsqueeze(torch.sign(
            (x0i1-x0k1)*(x0i0-x0j0) - (x0i1-x0j1)*(x0i0-x0k0)), -1)

        phi = s * torch.arccos(cosphis)

        a = torch.permute(torch.cat([torch.unsqueeze(
            torch.cos(phi), 0), torch.unsqueeze(-torch.sin(phi), 0)], 0), dims=[1, 2, 0])
        b = torch.permute(torch.cat([torch.unsqueeze(
            torch.sin(phi), 0), torch.unsqueeze(torch.cos(phi), 0)], 0), dims=[1, 2, 0])

        R = torch.einsum(
            "ijk...->jki...", torch.cat([torch.unsqueeze(a, 0), torch.unsqueeze(b, 0)], 0))

        xi = x[np.arange(x0s.shape[0]), inds[:, 0]]
        xj = x[np.arange(x0s.shape[0]), inds[:, 1]]

        scaled_ij = (xj-xi)/torch.unsqueeze(l_ijs, -1) * \
            torch.unsqueeze(gik, -1)

        x_k = torch.squeeze(torch.matmul(
            R, torch.unsqueeze(scaled_ij, -1))) + xi
        x_k = torch.where(torch.tile(torch.unsqueeze(
            node_types[:, k], -1), [1, thetas.shape[0], 2]) == 0.0, x_k, torch.zeros_like(x_k))
        x[:, k, :, :] += x_k
    return x, torch.cat(cos_list, dim=1)


def Transformation(curve):
    """Get transformation tuple for curve
    Parameters
    ----------
    curve:  numpy array [n_points,2]
            The curve to be transformed
    Returns
    -------
    tr:     tuple
            Transformation tuple (theta,scale,translation)
    """

    theta = -get_oriented_angle(curve)
    or_c = get_oriented(curve)
    scale = or_c.max()
    # scale = total_length(np.array([or_c]))[0]
    translation = curve.min(0)
    # translation = curve.mean(0)

    return theta, scale, translation


def apply_transformation(tr, curve):
    """ Apply transformation to curve
    Parameters
    ----------
    tr:     tuple
            Transformation tuple (theta,scale,translation)
    curve:  numpy array [n_points,2]
            The curve to be transformed
    Returns
    -------
    out:    numpy array [n_points,2]
            The transformed curve
    """

    theta, scale, translation = tr

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    out = np.matmul(rot, curve.T).T
    out = scale * out
    out = out - np.min(out, 0)
    # out = out - out.mean(0)
    out = out + translation

    return out


def batch_chamfer_distance(c1, c2):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    c1: torch tensor [n_points_x, n_dims]
        first point cloud
    c2: torch tensor [n_points_y, n_dims]
        second point cloud
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    d = torch.cdist(c1, c2)
    d1 = d.min(dim=1)[0].mean(dim=1)
    d2 = d.min(dim=2)[0].mean(dim=1)
    chamfer_dist = d1+d2
    return chamfer_dist


def get_mat(x0, C):
    """Get total link length of mechanism
    Parameters
    ----------
    x0:    torch tensor [N,2]
        The initial positions of the
    C:     torch tensor [N,N]
              Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.

    Returns
    -------
    total:     float
              total length of all links in mechanism.
    """

    N = C.shape[0]
    G = torch.zeros_like(C, dtype=float)
    for i in range(N):
        for j in range(i):
            if C[i, j]:
                G[i, j] = G[j, i] = torch.norm(x0[i]-x0[j])
            else:
                G[i, j] = G[j, i] = 0
    total = torch.sum(G)
    return total


def sort_mech(A, x0, motor, fixed_nodes):
    if motor[1] in fixed_nodes:
        return None
    nt = np.zeros([A.shape[0], 1])
    nt[fixed_nodes] = 1

    if motor[-1] in fixed_nodes:
        t = motor[0]
        motor[0] = motor[1]
        motor[1] = t

    motor_first_order = np.arange(A.shape[0])
    motor_first_order = motor_first_order[motor_first_order != motor[0]]
    motor_first_order = motor_first_order[motor_first_order != motor[1]]
    motor_first_order = np.concatenate([motor, motor_first_order])

    A = A[motor_first_order, :][:, motor_first_order]
    x0 = x0[motor_first_order]
    nt = nt[motor_first_order]
    p, check = find_path(A, [0, 1], np.where(nt)[0])

    if check and len(p.shape) > 1:
        fixed_nodes = np.where(nt[1:])[0]+1
        sorted_order = np.concatenate([[0, 1], fixed_nodes, p[:, 0]])
        A = A[sorted_order, :][:, sorted_order]
        x0 = x0[sorted_order]
        nt = nt[sorted_order]
        return A, x0, np.where(nt)[0], motor_first_order[sorted_order]
    else:
        return None


def functions_and_gradients(C, x0, fixed_nodes, target_pc, motor, idx=None, device='cpu', timesteps=2000):
    """Simulate, then return functions and gradients for the mechanism
    Parameters
    ----------
    C:     numpy array [N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [N,2]
            The positions of the nodes
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    target_pc: numpy array [n_points,2]
            The target point cloud that the mechanism should match.

    Returns
    -------
    validity: bool
            bool indicating whether the mechanism is valid
    CD_fn:     function
            function that returns the Chamfer distance between the target point cloud and the mechanism.
    mat_fn:     function
            function that returns the total length of the mechanism.
    CD_grad:     function
            function that returns the gradient of the Chamfer distance between the target point cloud and the mechanism.
    mat_grad:     function
            function that returns the gradient of the total length of the mechanism.
    matched_curve: numpy array [n_points,2]
            The curve that the mechanism matches to the target point cloud.

    """
    # target_pc = target_pc/500
    target_pc = get_oriented(target_pc)
    scale = target_pc.max()

    if idx is None:
        idx = C.shape[0]-1

    xc = x0.copy()
    res = sort_mech(C, x0, motor, fixed_nodes)
    if res:
        C, x0, fixed_nodes, sorted_order = res

        inverse_order = np.argsort(sorted_order)
    else:
        return False, None, None, None, None, None

    A = torch.Tensor(np.expand_dims(C, 0)).to(device)
    X = torch.Tensor(np.expand_dims(x0, 0)).to(device)
    node_types = np.zeros([1, C.shape[0], 1])
    node_types[0, fixed_nodes, :] = 1
    node_types = torch.Tensor(node_types).to(device)
    thetas = torch.Tensor(np.linspace(
        0, np.pi*2, timesteps+1)[0:timesteps]).to(device)

    x_sol, cos = solve_rev_vectorized_batch_wds(A, X, node_types, thetas)
    best_match = x_sol[0, idx].detach().cpu().numpy()
    best_tree = KDTree(best_match)

    tr = Transformation(best_match)
    matched_curve = apply_transformation(tr, target_pc/scale)
    matched_curve_180 = apply_transformation(
        (tr[0]+np.pi, tr[1], tr[2]), target_pc/scale)

    m_tree = KDTree(matched_curve)
    m180_tree = KDTree(matched_curve_180)

    multiplier = scale/tr[1]

    cd_tr = np.array([best_tree.query(matched_curve)[0].mean()+m_tree.query(best_match)[0].mean(),
                     best_tree.query(matched_curve_180)[0].mean()+m180_tree.query(best_match)[0].mean()])

    if cd_tr[1] < cd_tr[0]:
        tr = (tr[0]+np.pi, tr[1], tr[2])
        matched_curve = matched_curve_180

    target = torch.Tensor(matched_curve.astype(float)).to(device).unsqueeze(0)

    def CD_fn(x0_inp):
        x0_in = np.reshape(x0_inp, x0.shape)[sorted_order]/multiplier

        current_x0 = torch.nn.Parameter(torch.Tensor(
            np.expand_dims(x0_in, 0)), requires_grad=True).to(device)
        with torch.no_grad():

            sol, cos = solve_rev_vectorized_batch_wds(
                A, current_x0, node_types, thetas)

            ds = torch.square(
                torch.log(torch.pow(1 - torch.square(cos[0]), 0.25))).mean()

            current_sol = sol[0, idx]
            CD = batch_chamfer_distance(
                current_sol.unsqueeze(0)/tr[1], target/tr[1])[0]

        # if torch.isnan(current_sol).sum()>0 or CD<=thCD:
        #     break

        # final_loss = CD + pen*ds

        if torch.isnan(CD):
            return 1e6

        return CD.detach().cpu().numpy()*multiplier

    def mat_fn(x0_inp):
        x0_in = np.reshape(x0_inp, x0.shape)[sorted_order]
        x0_in = torch.from_numpy(x0_in)
        material = get_mat(x0_in, A[0])
        return material.detach().cpu().numpy()

    def CD_grad(x0_inp):
        x0_in = np.reshape(x0_inp, x0.shape)[sorted_order]/multiplier
        current_x0 = torch.nn.Parameter(torch.Tensor(
            np.expand_dims(x0_in, 0)), requires_grad=True).to(device)

        sol, cos = solve_rev_vectorized_batch_wds(
            A, current_x0, node_types, thetas)

        ds = torch.square(
            torch.log(torch.pow(1 - torch.square(cos[0]), 0.25))).mean()

        current_sol = sol[0, idx]
        CD = batch_chamfer_distance(
            current_sol.unsqueeze(0)/tr[1], target/tr[1])[0]

        CD.backward()

        if torch.isnan(CD):
            return np.zeros_like(x0_inp)
        return current_x0.grad.detach().cpu().numpy()[:, inverse_order].reshape(x0_inp.shape)*multiplier

    def mat_grad(x0_inp):
        x0_in = np.reshape(x0_inp, x0.shape)[sorted_order]
        current_x0 = torch.nn.Parameter(
            torch.Tensor(x0_in), requires_grad=True).to(device)
        material = get_mat(current_x0, A[0])
        material.backward()
        if torch.isnan(material):
            return np.zeros_like(x0_inp)

        return current_x0.grad.detach().cpu().numpy()[inverse_order].reshape(x0_inp.shape)

    return True, CD_fn, mat_fn, CD_grad, mat_grad, matched_curve*multiplier


def solve_mechanism(C, x0, fixed_nodes, motor, device='cpu', timesteps=2000):
    """Calculates all curves traced by a mechanism's nodes. Also returns validity and material use.
    Parameters
    ----------
    C:     numpy array [N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [N,2]
            The positions of the nodes
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    motor: numpy array [2]
            Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end).
    Returns
    -------
    validity: bool
            bool indicating whether the mechanism is valid
    sol:     numpy array [N,2]
            The positions of the nodes at each angle.
    cos:     numpy array [N,2]
            The cosines of the angles at each node. (Can be larger than 1 for locking mechanisms)
    material: float
            total length of all links in mechanism.
    """

    res = sort_mech(C, x0, motor, fixed_nodes)
    if res:
        C, x0, fixed_nodes, sorted_order = res

        inverse_order = np.argsort(sorted_order)
    else:
        return False, None, None, None

    A = torch.Tensor(np.expand_dims(C, 0)).to(device)
    X = torch.Tensor(np.expand_dims(x0, 0)).to(device)
    node_types = np.zeros([1, C.shape[0], 1])
    node_types[0, fixed_nodes, :] = 1
    node_types = torch.Tensor(node_types).to(device)
    thetas = torch.Tensor(np.linspace(
        0, np.pi*2, timesteps+1)[0:timesteps]).to(device)

    current_x0 = torch.Tensor(np.expand_dims(x0, 0)).to(device)
    sol, cos = solve_rev_vectorized_batch_wds(
        A, current_x0, node_types, thetas)
    material = get_mat(torch.Tensor(x0), A[0])
    if torch.isnan(sol).any():
        return False, None, None, None
    else:
        return True, sol, cos, material


def evaluate_mechanism(C, x0, fixed_nodes, motor, target_pc, idx=None, device='cpu', timesteps=2000):
    """Calculate the validity, chamfer distance, and material use of a mechanism with respect to a target curve.
    Also returns the curve that the mechanism traces.

    Parameters
    ----------
    C:     numpy array [N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [N,2]
            The positions of the nodes
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    motor: numpy array [2]
            Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end).
    target_pc: numpy array [n_points,2]
            The target point cloud that the mechanism should match.
    Returns
    -------
    validity: bool
            bool indicating whether the mechanism is valid
    CD:     float
            The chamfer distance between the target point cloud and the mechanism.
    material: float
            total length of all links in mechanism.
    sol:     numpy array [N,2]
            The positions of the nodes at each angle.
    """

    if idx is None:
        idx = C.shape[0]-1
    target_pc = get_oriented(target_pc)
    valid, sol, cos, material = solve_mechanism(
        C, x0, fixed_nodes, motor, device, timesteps)
    if not valid:
        return False, None, None, None
    else:
        sol = sol.detach().numpy()[0, idx, :, :]
        sol1, sol2 = get_oriented_both(sol)
        CD1 = batch_chamfer_distance(torch.tensor(sol1, dtype=float).unsqueeze(
            0), torch.tensor(target_pc, dtype=float).unsqueeze(0))[0]
        CD2 = batch_chamfer_distance(torch.tensor(sol2, dtype=float).unsqueeze(
            0), torch.tensor(target_pc, dtype=float).unsqueeze(0))[0]
        if CD1 < CD2:
            CD = CD1
            sol = sol1
        else:
            CD = CD2
            sol = sol2

        return True, float(CD), float(material), sol


def batch_random_generator(N, g_prob=0.15, n=None, N_min=8, N_max=20, strategy='rand', scale=1, show_progress=True):
    """Fast generate a batch of random mechanisms that are not locking or invalid.
    Parameters
    ----------
    N:      int
            The number of mechanims to generate
    g_prob: float
            Probability of a node being assigned as ground. Default: 0.15
    n:     int
            Size of mechanism. Default: None (variable size)
    N_min: int
            Minimum size of mechanims if n is not set. Default: 8
    N_max: int
            Maximum size of mechanims if n is not set. Default: 20
    strategy: str
            Either rand or srand. 'rand': fully random, 'srand': sequentially built random. Default: 'rand'
    scale: float
            Scale of the mechanism. Default: 1
    show_progress: Boolean
            If true will display progress bar. Deafault: True

    Returns
    -------
    list of: [N,...]

        C:     numpy array [n,n]
                Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
        x0:    numpy array [n,2]
                    The initial positions of the
        fixed_nodes: numpy array [n_fixed_nodes]
                A list of the nodes that are grounded/fixed.
        motor: numpy array [2]
                Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end).
    """

    args = [(g_prob, n, N_min, N_max, strategy, scale)]*N
    return run_imap_multiprocessing(auxilary, args, show_prog=show_progress)

# Auxiary function for batch generator


def auxilary(intermidiate):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    return random_generator_ns(g_prob=intermidiate[0], n=intermidiate[1], N_min=intermidiate[2], N_max=intermidiate[3], strategy=intermidiate[4])


def random_generator_ns(g_prob=0.15, n=None, N_min=8, N_max=20, scale=1, strategy='rand'):
    """Generate random mechanism that is not locking or invalid.
    Parameters
    ----------
    g_prob: float
            Probability of a node being assigned as ground. Default: 0.15
    n:     int
            Size of mechanism. Default: None (variable size)
    N_min: int
            Minimum size of mechanims if n is not set. Default: 8
    N_max: int
            Maximum size of mechanims if n is not set. Default: 20
    strategy: str
            Either rand or srand. 'rand': fully random, 'srand': sequentially built random. Default: 'rand'
    scale: float
            Scale of the mechanism. Default: 1

    Returns
    -------
    C:     numpy array [n,n]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [n,2]
                The initial positions of the
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    motor: numpy array [2]
            Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end).
    """
    if not n:
        n = int(np.round(np.random.uniform(low=N_min, high=N_max)))

    edges = [[0, 1], [1, 3], [2, 3]]

    fixed_nodes = [0, 2]
    motor = [0, 1]

    node_types = np.random.binomial(1, g_prob, n-4)

    for i in range(4, n):
        if node_types[i-4] == 1:
            fixed_nodes.append(i)
        else:
            picks = np.random.choice(i, size=2, replace=False)

            while picks[0] in fixed_nodes and picks[1] in fixed_nodes:
                picks = np.random.choice(i, size=2, replace=False)

            edges.append([picks[0], i])
            edges.append([picks[1], i])

    C = np.zeros([n, n], dtype=bool)

    for edge in edges:
        C[edge[0], edge[1]] = True
        C[edge[1], edge[0]] = True

    fixed_nodes = np.array(fixed_nodes)

    if strategy == 'srand':
        x = np.random.uniform(low=0.0, high=scale, size=[n, 2])

        for i in range(4, n+1):

            sub_size = i
            valid, _, _, _ = solve_mechanism(
                C, x, fixed_nodes, motor, device="cpu", timesteps=50)
            invalid = not valid

            co = 0

            while invalid:
                if sub_size > 4:
                    x[sub_size -
                        1:sub_size] = np.random.uniform(low=0.0, high=1.0, size=[1, 2])
                else:
                    x[0:sub_size] = np.random.uniform(
                        low=0.0, high=1.0, size=[sub_size, 2])

                valid, _, _, _ = solve_mechanism(C[0:sub_size, 0:sub_size], x[0:sub_size], fixed_nodes[np.where(
                    fixed_nodes < sub_size)], motor, device="cpu", timesteps=50)
                invalid = not valid
                co += 1

                if co == 100:
                    neighbours = np.where(C[sub_size-1])[0]
                    relavent_neighbours = neighbours[np.where(
                        neighbours < sub_size-1)]
                    C[relavent_neighbours[0], sub_size-1] = 0
                    C[relavent_neighbours[1], sub_size-1] = 0
                    C[sub_size-1, relavent_neighbours[0]] = 0
                    C[sub_size-1, relavent_neighbours[1]] = 0
                    picks = np.random.choice(sub_size-1, size=2, replace=False)
                    C[picks[0], sub_size-1] = True
                    C[picks[1], sub_size-1] = True
                    C[sub_size-1, picks[0]] = True
                    C[sub_size-1, picks[1]] = True
                    co = 0
    else:
        co = 0
        x = np.random.uniform(low=0.05*scale, high=0.95*scale, size=[n, 2])
        valid, _, _, _ = solve_mechanism(
            C, x, fixed_nodes, motor, device="cpu", timesteps=50)
        invalid = not valid

        res = sort_mech(C, x, motor, fixed_nodes)
        if res:
            C, x, fixed_nodes, sorted_order = res

            inverse_order = np.argsort(sorted_order)
        else:
            invalid = True
        while invalid:
            x = np.random.uniform(low=0.1+0.25*co/1000,
                                  high=0.9-0.25*co/1000, size=[n, 2])
            valid, _, _, _ = solve_mechanism(
                C, x, fixed_nodes, motor, device="cpu", timesteps=50)
            invalid = not valid
            co += 1

            if co >= 1000:
                return random_generator_ns(g_prob, n, N_min, N_max, strategy)

    return C, x, fixed_nodes, [0, 1]


def run_imap_multiprocessing(func, argument_list, show_prog=True):
    """Run function in parallel
    Parameters
    ----------
    func:          function
                    Python function to run in parallel.
    argument_list: list [N]
                    List of arguments to be passed to the function in each parallel run.

    show_prog:     boolean
                    If true a progress bas will be displayed to show progress. Default: True.
    Returns
    -------
    output:        list [N,]
                    outputs of the function for the given arguments.
    """
    pool = mp.Pool(processes=mp.cpu_count())

    if show_prog:
        result_list_tqdm = []
        for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list), position=0, leave=True):
            result_list_tqdm.append(result)
    else:
        result_list_tqdm = []
        for result in pool.imap(func=func, iterable=argument_list):
            result_list_tqdm.append(result)

    return result_list_tqdm


class best(Display):
    """Pymoo display to show the best performance of each generation during optimization
    """

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        lowest = np.min(algorithm.pop.get("F"), 0)
        for i in range(lowest.shape[0]):
            self.output.append(
                "Lowest Memeber for Objective %i" % (i), lowest[i])


def hyper_volume(F, ref):
    """Get Hypervolume of population
    Parameters
    ----------
    F:          numpy array [N,m]
                    Perfromance of the paretor front/population.
    ref:        numpy array [m]
                    Reference point for hypervolume calculations.

    Returns
    -------
    hypervolume:  float
                    Hyper volume of the selected population with respect to the reference point.
    """
    ind = HV(ref_point=np.array(ref))
    return ind(F)


def evaluate_submission():
    """Evaluate CSVs in the results folder
    Returns
    -------
    score:  float
            Score of the submission
    """

    scores = []

    target_curves = []

    # Read every file separately and append to the list
    for i in range(6):
        if not os.path.exists('./data/%i.csv' % (i)):
            raise IOError('Could not find %i.csv in the data folder' % (i))
        target_curves.append(np.loadtxt('./data/%i.csv' % (i), delimiter=','))

    for i in trange(6):
        if os.path.exists('./results/%i.csv' % i):

            mechanisms = get_population_csv('./results/%i.csv' % i)
            print(i)
            F = []
            for m in mechanisms:
                C, x0, fixed_nodes, motor, target = from_1D_representation(m)

                # Solve
                valid, CD, material, _ = evaluate_mechanism(
                    C, x0, fixed_nodes, motor, target_curves[i], target, device='cpu', timesteps=2000)

                if valid:
                    if CD <= 0.1 and material <= 10.0:
                        print(f'CD{CD} mat{material}')
                        F.append([CD, material])
            if len(F):
                if len(F) > 1000:
                    print(
                        "Over 1000 linkages submitted! Truncating submission to first 1000.")
                    F = F[:1000]
                
                scores.append(hyper_volume(np.array(F), [0.1, 10.0]))
            else:
                scores.append(0)
        else:
            scores.append(0)

    print('Score Break Down:')
    for i in range(6):
        print('Curve %i: %f' % (i, scores[i]))

    print('Overall Score: %f' % (np.mean(scores)))

    return np.mean(scores)


def to_final_representation(C, x0, fixed_nodes, motor, target):
    """Get 1D representation of mechanism
    ----------
    C:     numpy array [N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [N,2]
            The initial positions of the
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    motor: numpy array [2]
            Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end).
    target: int
            The index of the node which is to be evaluated as the target node.
    Returns
    -------
    1D representation: numpy array [N^2+3*N+3]
            1D representation of mechanims
    """
    C, x0, fixed_nodes, motor, target = np.array(C), np.array(
        x0), np.array(fixed_nodes), np.array(motor), np.array(target)

    # Get the size of the mechanism in this case 5
    N = C.shape[0]

    # Make the list of node types (set all to zero or ordinary)
    node_types = np.zeros([N])

    # Set the ground nodes
    node_types[fixed_nodes] = 1

    # Set the target
    target_node = target

    # Concatenate to make the final representation
    final_representation = np.concatenate(
        [[N], C.reshape(-1), x0.reshape(-1), node_types, motor, [target_node]])

    return final_representation


def from_1D_representation(mechanism):
    """Get python representation of mechanism from 1D representation
    ----------
    mechanism: numpy array [N^2+3*N+3]
                1D representation of mechanims

    Returns
    -------
    C:     numpy array [N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [N,2]
            The initial positions of the
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    motor: numpy array [2]
            Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end).
    target: int
            The index of the node which is to be evaluated as the target node.
    """
    mechanism = np.array(mechanism)

    # Get size
    N = int(mechanism[0])

    mechanism = mechanism[1:]

    # Extract mechanism components
    C = mechanism[0:N**2].reshape([N, N])
    x0 = mechanism[N**2:N**2 + 2*N].reshape([N, 2])
    fixed_nodes = np.where(mechanism[N**2 + 2*N:N**2 + 3*N])[0].astype(int)
    motor = mechanism[N**2 + 3*N:N**2 + 3*N+2].astype(int)
    target = mechanism[-1].astype(int)

    return C, x0, fixed_nodes, motor, target


def save_population_csv(file_name, population):
    """Save a population of mechanims as csv for submission/evaluation
    ----------
    file_name: str
                File name and path to save the population in.
    population: list [N]
                A list of 1D mechanims representations to be saved in the CSV.
    """

    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        for m in population:
            m = np.array(m)
            writer.writerow(m.tolist())


def get_population_csv(file_name):
    """Load a population of mechanims from csv of submission/evaluation
    ----------
    file_name: str
                File name and path to save the population in.

    Returns
    -------
    population: list [N]
                A list of 1D mechanims representations from the saved CSV.
    """
    population = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            population.append(np.array(row).astype(float))
    return population


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    Parameters
    ----------
    costs:      numpy array [N,n_objectives]
                Objective-wise performance values
    return_mask:bool
                If True, the function returns a mask of dominated points, else it returns the indices of efficient points.
    Returns
    -------
    is_efficient:   numpy array [N]
                    If return_mask is True, this is an array boolean values indicating whether each point in the input `costs` is Pareto efficient.
                    If return_mask is False, this is an array of indices of the points in the input `costs` array that are Pareto efficient.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(
            costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def visualize_pareto_front(mechanisms, F, target_curve):
    """Draw Pareto Front in population and visualize mechanisms
    Parameters
    ----------
    mechanisms: numpy array [N,...]
                    List of 1D representations of the mechanims in the population.
    F:          numpy array [N,n_objectives]
                    Perfromance of the paretor population ([[chamfer distance, material]*N]).
    target_curve:  numpy array [n,2]
                    point cloud of target.

    """
    ind = is_pareto_efficient(F)
    X_p = mechanisms[ind]
    F_p = F[ind]

    ind = np.argsort(F_p[:, 0])

    X_p = X_p[ind]
    F_p = F_p[ind]

    fig, axs = plt.subplots(X_p.shape[0], 3, figsize=(15, 5*X_p.shape[0]))

    for i in trange(X_p.shape[0]):
        C, x0, fixed_nodes, motor, target = from_1D_representation(X_p[i])
        draw_mechanism_on_ax(C, x0, fixed_nodes, motor, axs[i, 0])

        # Solve
        valid, CD, mat, sol = evaluate_mechanism(
            C, x0, fixed_nodes, motor, target_curve, idx=target)
        target_curve = get_oriented(target_curve)
        # Plot
        axs[i, 1].scatter(target_curve[:, 0], target_curve[:, 1], s=2)
        axs[i, 1].scatter(sol[:, 0], sol[:, 1], s=2)
        axs[i, 1].axis('equal')

        axs[i, 1].set_title('Chamfer Distance: %f' % (CD))

        axs[i, 2].scatter(F_p[:, 1], F_p[:, 0])
        axs[i, 2].set_xlabel('Material Use')
        axs[i, 2].set_ylabel('Chamfer Distance')
        axs[i, 2].scatter([F_p[i, 1]], [F_p[i, 0]], color="red")


def find_path(A, motor=[0, 1], fixed_nodes=[0, 1]):
    """
    Deduce node heirarchy of a mechanism
    Parameters
    ----------
    A:          numpy array [N,N]
                Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    motor:      numpy array [2]
                Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end).
    fixed_nodes: numpy array [n_fixed_nodes]
                A list of the nodes that are grounded/fixed.
    Returns
    -------
    path:       numpy array [N,3]
                Path of the mechanism
    valid:      bool
                If the mechanism is valid
    """

    path = []

    A, fixed_nodes, motor = np.array(A), np.array(fixed_nodes), np.array(motor)

    if (A != A.T).any():
        return [], False

    unknowns = np.array(list(range(A.shape[0])))

    if motor[-1] in fixed_nodes:
        driven = motor[0]
        driving = motor[-1]
    else:
        driven = motor[-1]
        driving = motor[0]

    if motor[-1] not in fixed_nodes and motor[0] not in fixed_nodes:
        return [], False

    for item in fixed_nodes:
        if item != driving:
            if A[driven, item]:
                return [], False

    knowns = np.concatenate([fixed_nodes, [driven]])

    unknowns = unknowns[np.logical_not(np.isin(unknowns, knowns))]

    counter = 0
    while unknowns.shape[0] != 0:
        if counter == unknowns.shape[0]:
            # Non dyadic or DOF larger than 1
            return [], False
        n = unknowns[counter]
        ne = np.where(A[n])[0]

        kne = knowns[np.isin(knowns, ne)]
#         print(kne.shape[0])

        if kne.shape[0] == 2:

            path.append([n, kne[0], kne[1]])
            counter = 0
            knowns = np.concatenate([knowns, [n]])
            unknowns = unknowns[unknowns != n]
        elif kne.shape[0] > 2:
            #redundant or overconstraint
            return [], False
        else:
            counter += 1
    return np.array(path), True


def get_G(x0):
    return (np.linalg.norm(np.tile([x0], [x0.shape[0], 1, 1]) - np.tile(np.expand_dims(x0, 1), [1, x0.shape[0], 1]), axis=-1))


def solve_rev_vectorized(path, x0, G, motor, fixed_nodes, thetas):

    path, x0, G, motor, fixed_nodes = np.array(path), np.array(
        x0), np.array(G), np.array(motor), np.array(fixed_nodes)

    x = np.zeros([x0.shape[0], thetas.shape[0], 2])

    x[fixed_nodes] = np.expand_dims(x0[fixed_nodes], 1)

    if motor[-1] in fixed_nodes:
        driven = motor[0]
        motor_joint = motor[-1]
    else:
        driven = motor[-1]
        motor_joint = motor[0]

    x[driven] = x[motor_joint] + G[motor[0], motor[1]] * \
        np.concatenate([[np.cos(thetas)], [np.sin(thetas)]]).T

    state = np.zeros(thetas.shape[0])
    flag = True
    kk = np.zeros(thetas.shape[0]) - 1.0

    for step in path:
        i = step[1]
        j = step[2]
        k = step[0]

        l_ij = np.linalg.norm(x[j]-x[i], axis=-1)
        cosphi = (l_ij ** 2 + G[i, k]**2 - G[j, k]**2)/(2 * l_ij * G[i, k])

        state += np.logical_or(cosphi < -1.0, cosphi > 1.0)

        kk = state * k * (kk == -1.0) + kk

        s = np.sign((x0[i, 1]-x0[k, 1])*(x0[i, 0]-x0[j, 0]) -
                    (x0[i, 1]-x0[j, 1])*(x0[i, 0]-x0[k, 0]))

        phi = s * np.arccos(cosphi)

        a = np.concatenate([[np.cos(phi)], [-np.sin(phi)]]).T
        b = np.concatenate([[np.sin(phi)], [np.cos(phi)]]).T

        R = np.swapaxes(np.concatenate([[a], [b]]), 0, 1)

        scaled_ij = (x[j]-x[i])/np.expand_dims(l_ij, -1) * G[i, k]
        x[k] = np.squeeze(R @ np.expand_dims(scaled_ij, -1)) + x[i]

    kk = (kk != -1.0) + kk
    return x, state == 0.0, kk.astype(int)


def draw_mechanism(A, x0, fixed_nodes, motor, highlight=100, solve=True, thetas=np.linspace(0, np.pi*2, 200), def_alpha=1.0, h_alfa=1.0, h_c="#f15a24"):

    valid, _, _, _ = solve_mechanism(
        A, x0, fixed_nodes, motor, device="cpu", timesteps=2000)
    if not valid:
        print("Mechanism is invalid!")
        return

    fig = plt.figure(figsize=(12, 12))

    def fetch_path():
        root = etree.parse(StringIO(
            '<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 338"><defs><style>.cls-1{fill:#1a1a1a;stroke:#1a1a1a;stroke-linecap:round;stroke-miterlimit:10;stroke-width:20px;}</style></defs><path class="cls-1" d="M45.5,358.5l70.71-70.71M46,287.5H644m-507.61,71,70.72-70.71M223,358.5l70.71-70.71m20.18,70.72,70.71-70.71m13.67,70.7,70.71-70.71m20.19,70.72,70.71-70.71m15.84,70.71,70.71-70.71M345,39.62A121.38,121.38,0,1,1,223.62,161,121.38,121.38,0,0,1,345,39.62Z" transform="translate(-35.5 -29.62)"/></svg>')).getroot()
        view_box = root.attrib.get('viewBox')
        if view_box is not None:
            view_box = [int(x) for x in view_box.split()]
            xlim = (view_box[0], view_box[0] + view_box[2])
            ylim = (view_box[1] + view_box[3], view_box[1])
        else:
            xlim = (0, 500)
            ylim = (500, 0)
        path_elem = root.findall('.//{http://www.w3.org/2000/svg}path')[0]
        return xlim, ylim, parse_path(path_elem.attrib['d'])
    _, _, p = fetch_path()
    p.vertices -= p.vertices.mean(axis=0)
    p.vertices = (np.array([[np.cos(np.pi), -np.sin(np.pi)],
                  [np.sin(np.pi), np.cos(np.pi)]])@p.vertices.T).T

    A, x0, fixed_nodes, motor = np.array(A), np.array(
        x0), np.array(fixed_nodes), np.array(motor)

    x = x0

    N = A.shape[0]
    for i in range(N):
        if i in fixed_nodes:
            if i == highlight:
                plt.scatter(x[i, 0], x[i, 1], color=h_c,
                            s=700, zorder=10, marker=p)
            else:
                plt.scatter(x[i, 0], x[i, 1], color="#1a1a1a",
                            s=700, zorder=10, marker=p)
        else:
            if i == highlight:
                plt.scatter(x[i, 0], x[i, 1], color=h_c, s=100,
                            zorder=10, facecolors=h_c, alpha=0.7)
            else:
                plt.scatter(x[i, 0], x[i, 1], color="#1a1a1a", s=100,
                            zorder=10, facecolors='#ffffff', alpha=0.7)

        for j in range(i+1, N):
            if A[i, j]:
                if (motor[0] == i and motor[1] == j) or (motor[0] == j and motor[1] == i):
                    plt.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]],
                             color="#ffc800", linewidth=4.5)
                else:
                    plt.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]],
                             color="#1a1a1a", linewidth=4.5, alpha=0.6)

    if solve:
        path = find_path(A, motor, fixed_nodes)[0]
        G = get_G(x0)
        x, c, k = solve_rev_vectorized(path.astype(
            int), x0, G, motor, fixed_nodes, thetas)
        x = np.swapaxes(x, 0, 1)
        if np.sum(c) == c.shape[0]:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        plt.plot(x[:, i, 0], x[:, i, 1], '-',
                                 color=h_c, linewidth=4.5, alpha=h_alfa)
                    else:
                        plt.plot(x[:, i, 0], x[:, i, 1], '--',
                                 color="#0078a7", linewidth=1.5, alpha=def_alpha)
        else:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        plt.plot(x[:, i, 0], x[:, i, 1], '-',
                                 color=h_c, linewidth=4.5, alpha=h_alfa)
                    else:
                        plt.plot(x[:, i, 0], x[:, i, 1], '--',
                                 color="#0078a7", linewidth=1.5, alpha=def_alpha)
            plt.text(0.5, 0.5, 'Locking Or Under Defined', color='red',
                     horizontalalignment='center', verticalalignment='center')

    plt.axis('equal')
    plt.axis('off')


def draw_mechanism_on_ax(A, x0, fixed_nodes, motor, ax, highlight=100, solve=True, thetas=np.linspace(0, np.pi*2, 200), def_alpha=1.0, h_alfa=1.0, h_c="#f15a24"):
    valid, _, _, _ = solve_mechanism(
        A, x0, fixed_nodes, motor, device="cpu", timesteps=50)
    if not valid:
        print("Mechanism is invalid!")
        return
    # fig = plt.figure(figsize=(12,12))

    def fetch_path():
        root = etree.parse(StringIO(
            '<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 338"><defs><style>.cls-1{fill:#1a1a1a;stroke:#1a1a1a;stroke-linecap:round;stroke-miterlimit:10;stroke-width:20px;}</style></defs><path class="cls-1" d="M45.5,358.5l70.71-70.71M46,287.5H644m-507.61,71,70.72-70.71M223,358.5l70.71-70.71m20.18,70.72,70.71-70.71m13.67,70.7,70.71-70.71m20.19,70.72,70.71-70.71m15.84,70.71,70.71-70.71M345,39.62A121.38,121.38,0,1,1,223.62,161,121.38,121.38,0,0,1,345,39.62Z" transform="translate(-35.5 -29.62)"/></svg>')).getroot()
        view_box = root.attrib.get('viewBox')
        if view_box is not None:
            view_box = [int(x) for x in view_box.split()]
            xlim = (view_box[0], view_box[0] + view_box[2])
            ylim = (view_box[1] + view_box[3], view_box[1])
        else:
            xlim = (0, 500)
            ylim = (500, 0)
        path_elem = root.findall('.//{http://www.w3.org/2000/svg}path')[0]
        return xlim, ylim, parse_path(path_elem.attrib['d'])
    _, _, p = fetch_path()
    p.vertices -= p.vertices.mean(axis=0)
    p.vertices = (np.array([[np.cos(np.pi), -np.sin(np.pi)],
                  [np.sin(np.pi), np.cos(np.pi)]])@p.vertices.T).T

    A, x0, fixed_nodes, motor = np.array(A), np.array(
        x0), np.array(fixed_nodes), np.array(motor)

    x = x0

    N = A.shape[0]
    for i in range(N):
        if i in fixed_nodes:
            if i == highlight:
                ax.scatter(x[i, 0], x[i, 1], color=h_c,
                           s=700, zorder=10, marker=p)
            else:
                ax.scatter(x[i, 0], x[i, 1], color="#1a1a1a",
                           s=700, zorder=10, marker=p)
        else:
            if i == highlight:
                ax.scatter(x[i, 0], x[i, 1], color=h_c, s=100,
                           zorder=10, facecolors=h_c, alpha=0.7)
            else:
                ax.scatter(x[i, 0], x[i, 1], color="#1a1a1a", s=100,
                           zorder=10, facecolors='#ffffff', alpha=0.7)

        for j in range(i+1, N):
            if A[i, j]:
                if (motor[0] == i and motor[1] == j) or (motor[0] == j and motor[1] == i):
                    ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]],
                            color="#ffc800", linewidth=4.5)
                else:
                    ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]],
                            color="#1a1a1a", linewidth=4.5, alpha=0.6)

    if solve:
        path = find_path(A, motor, fixed_nodes)[0]
        G = get_G(x0)
        x, c, k = solve_rev_vectorized(path.astype(
            int), x0, G, motor, fixed_nodes, thetas)
        x = np.swapaxes(x, 0, 1)
        if np.sum(c) == c.shape[0]:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        ax.plot(x[:, i, 0], x[:, i, 1], '-',
                                color=h_c, linewidth=4.5, alpha=h_alfa)
                    else:
                        ax.plot(x[:, i, 0], x[:, i, 1], '--',
                                color="#0078a7", linewidth=1.5, alpha=def_alpha)
        else:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        ax.plot(x[:, i, 0], x[:, i, 1], '-',
                                color=h_c, linewidth=4.5, alpha=h_alfa)
                    else:
                        ax.plot(x[:, i, 0], x[:, i, 1], '--',
                                color="#0078a7", linewidth=1.5, alpha=def_alpha)
            ax.text(0.5, 0.5, 'Locking Or Under Defined', color='red',
                    horizontalalignment='center', verticalalignment='center')

    ax.axis('equal')
    ax.axis('off')
