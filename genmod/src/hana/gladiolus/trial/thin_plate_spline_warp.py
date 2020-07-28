import math

import torch
from torch.nn.functional import grid_sample


def diff_square_matrix(v: torch.Tensor):
    assert v.dim() == 1
    n = v.shape[0]
    diff = v.view(n, 1).expand(-1, n) - v.view(1, n).expand(n, -1)
    return diff ** 2


def landmark_matrix(landmarks: torch.Tensor):
    n = landmarks.shape[0]
    assert landmarks.shape[1] == 2
    return torch.cat([torch.ones(n, 1), landmarks], dim=1)


def thin_plate_spline_matrices(landmarks: torch.Tensor):
    N = landmark_matrix(landmarks)
    NT = N.transpose(0, 1)

    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    distance_squared = diff_square_matrix(landmarks_x) + diff_square_matrix(landmarks_y)
    n = landmarks.shape[0]
    M = distance_squared * torch.log(distance_squared + torch.eye(n)) / 2 / (8 * math.pi)
    Minv = M.inverse()
    Ncross = (NT.mm(Minv).mm(N)).inverse().mm(NT.mm(Minv))

    return Minv, Ncross


def thin_plate_spline_coeffs(landmarks: torch.Tensor, warped_landmarks: torch.Tensor):
    Minv, Ncross = thin_plate_spline_matrices(warped_landmarks)

    N = landmark_matrix(warped_landmarks)

    n = landmarks.shape[0]
    landmarks_x = landmarks[:, 0].view(n, 1)
    landmarks_y = landmarks[:, 1].view(n, 1)

    b_x = Ncross.mm(landmarks_x)
    a_x = Minv.mm(landmarks_x - N.mm(b_x))
    b_x = b_x.squeeze(dim=1)
    a_x = a_x.squeeze(dim=1)

    b_y = Ncross.mm(landmarks_y)
    a_y = Minv.mm(landmarks_y - N.mm(b_y))
    b_y = b_y.squeeze(dim=1)
    a_y = a_y.squeeze(dim=1)

    return a_x, b_x, a_y, b_y

def eval_thin_plate_spline(point: torch.Tensor, warped_landmarks: torch.Tensor, a_x, b_x, a_y, b_y):
    n = warped_landmarks.shape[0]
    epsilon = 1e-16
    output = torch.zeros(2, device=point.device)
    for i in range(n):
        r2 = ((point - warped_landmarks[i])**2).sum()
        r2log = r2 * math.log(r2 + epsilon) / 2 / (8 * math.pi)
        output[0] += a_x[i] * r2log
        output[1] += a_y[i] * r2log
    output[0] += b_x[0] + b_x[1] * point[0] + b_x[2] * point[1]
    output[1] += b_y[0] + b_y[1] * point[0] + b_y[2] * point[1]
    return output

def thin_plate_spline_warp(image: torch.Tensor, landmarks: torch.Tensor, warped_landmarks: torch.Tensor):
    assert image.shape[2] == image.shape[3]
    assert 1 == image.shape[0]

    a_x, b_x, a_y, b_y = thin_plate_spline_coeffs(landmarks, warped_landmarks)

    n = landmarks.shape[0]
    num_pixel = image.shape[2]
    pixel_size = 2.0 / num_pixel
    range_start = -1 + pixel_size / 2
    range_end = 1 - pixel_size / 2
    grid_values = torch.linspace(range_start, range_end, num_pixel)
    grid_u_values = grid_values.view(1, num_pixel)
    grid_u_values = grid_u_values.expand(num_pixel, -1)
    grid_v_values = grid_values.view(num_pixel, 1)
    grid_v_values = grid_v_values.expand(-1, num_pixel)

    grid = torch.zeros(1, num_pixel, num_pixel, 2)
    grid[0, :, :, 0] = b_x[0] + b_x[1] * grid_u_values + b_x[2] * grid_v_values
    grid[0, :, :, 1] = b_y[0] + b_y[1] * grid_u_values + b_y[2] * grid_v_values
    epsilon = 1e-16
    for i in range(n):
    #for i in range(0):
        lx = warped_landmarks[i, 0]
        grid_u_diff2 = (grid_values - lx) ** 2
        grid_u_diff2 = grid_u_diff2.view(1, num_pixel)
        grid_u_diff2 = grid_u_diff2.expand(num_pixel, -1)

        ly = warped_landmarks[i, 1]
        grid_v_diff2 = (grid_values - ly) ** 2
        grid_v_diff2 = grid_v_diff2.view(num_pixel, 1)
        grid_v_diff2 = grid_v_diff2.expand(-1, num_pixel)

        grid_r2 = grid_u_diff2 + grid_v_diff2
        grid_r2_log = grid_r2 * torch.log(grid_r2 + epsilon) / 2 / (8 * math.pi)

        grid[0, :, :, 0] += grid_r2_log * a_x[i]
        grid[0, :, :, 1] += grid_r2_log * a_y[i]

    return grid_sample(image, grid, padding_mode='border', align_corners=False)


if __name__ == "__main__":
    landmarks = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ])

    warped_landmarks = torch.tensor([
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 4.0]
    ])

    a_x, b_x, a_y, b_y = thin_plate_spline_coeffs(landmarks, warped_landmarks)
    n = warped_landmarks.shape[0]
    for i in range(n):
        input = warped_landmarks[i]
        output = eval_thin_plate_spline(input, warped_landmarks, a_x, b_x, a_y, b_y)
        print(input, output - landmarks[i])

    #image = torch.zeros(1, 3, 512, 512)
    # thin_plate_spline_warp(image, landmarks, warped_landmarks)

    #identity = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32).unsqueeze(0)
    #A = affine_grid(identity, [1, 1, 4, 4], align_corners=False)
    #print(A.shape)
    # print(torch.linspace(0.5, 9.5, 10))
