import torch
def thin_plate_spline(src_points, target_points):
    device = src_points.device
    num_ctrl_points = len(src_points)
    # Slices return row vectors
    mesh = target_points[:, 0].repeat((num_ctrl_points, 1))
    rx = torch.abs(torch.t(mesh) - mesh)
    mesh = target_points[:, 1].repeat((num_ctrl_points, 1))
    ry = torch.abs(torch.t(mesh) - mesh)
    mesh = target_points[:, 2].repeat((num_ctrl_points, 1))
    rz = torch.abs(torch.t(mesh) - mesh)

    sqrt_r = torch.sqrt((rx * rx) + (ry * ry) + (rz * rz))
    T = torch.zeros((num_ctrl_points + 4, num_ctrl_points + 4),
                    device=device).double()
    T[:num_ctrl_points, 0] = 1
    T[:num_ctrl_points, 1:4] = target_points
    T[:num_ctrl_points, 4:(num_ctrl_points + 4)] = sqrt_r
    T[num_ctrl_points, 4:(num_ctrl_points + 4)] = 1

    T[(num_ctrl_points + 1):(num_ctrl_points + 4),
      4:(num_ctrl_points+4)] = torch.t(target_points)

    denom_a = torch.zeros((num_ctrl_points + 4, 1), device=device).double()
    denom_b = torch.zeros((num_ctrl_points + 4, 1), device=device).double()
    denom_c = torch.zeros((num_ctrl_points + 4, 1), device=device).double()

    denom_a[:num_ctrl_points, 0] = src_points[:, 0]
    denom_b[:num_ctrl_points, 0] = src_points[:, 1]
    denom_c[:num_ctrl_points, 0] = src_points[:, 2]

    #soln_a, _ = torch.lstsq(denom_a, T)
    #soln_b, _ = torch.lstsq(denom_b, T)
    #soln_c, _ = torch.lstsq(denom_c, T)

    soln_a, _, _, _ = torch.linalg.lstsq(T, denom_a)
    soln_b, _, _, _ = torch.linalg.lstsq(T, denom_b)
    soln_c, _, _, _ = torch.linalg.lstsq(T, denom_c)

    def transformation(points):
        num_points = len(points)
        rx = (points[:, 0].repeat((num_ctrl_points, 1))
              - torch.t(target_points[:, 0].repeat((num_points, 1))))

        ry = (points[:, 1].repeat((num_ctrl_points, 1))
              - torch.t(
                  target_points[:, 1].repeat((num_points, 1))))

        rz = (points[:, 2].repeat((num_ctrl_points, 1))
              - torch.t(
                  target_points[:, 2].repeat((num_points, 1))))

        pmat = torch.ones((num_points, 4 + num_ctrl_points),
                          device=device).double()
        pmat[:, 1:4] = points
        pmat[:, 4:] = torch.t(torch.sqrt((rx * rx) + (ry * ry) + (rz * rz)))

        x = torch.mm(pmat, soln_a)
        y = torch.mm(pmat, soln_b)
        z = torch.mm(pmat, soln_c)
        return torch.squeeze(torch.stack([x, y, z], 1))

    return transformation