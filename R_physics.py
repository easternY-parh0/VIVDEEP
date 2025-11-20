import torch

def compute_physics_residual(model, t, D, MR, M, K, C, device):
    """
    Returns:
      preds: [N,4] (d_x, d_y, h_x, h_y)
      resid: [N,2] (r_x, r_y) where r = M d'' + C d' + K d - h
    """

    t = t.clone().detach().to(device).requires_grad_(True)
    inp = torch.stack([t, D, MR], dim=1).to(device)
    y = model(inp)
    d_x = y[:,0]; d_y = y[:,1]; h_x = y[:,2]; h_y = y[:,3]

    # derivatives wrt normalized time
    d_x_tn = torch.autograd.grad(d_x, t, grad_outputs=torch.ones_like(d_x), create_graph=True)[0]
    d_y_tn = torch.autograd.grad(d_y, t, grad_outputs=torch.ones_like(d_y), create_graph=True)[0]

    d_x_ttn = torch.autograd.grad(d_x_tn, t, grad_outputs=torch.ones_like(d_x_tn), create_graph=True)[0]
    d_y_ttn = torch.autograd.grad(d_y_tn, t, grad_outputs=torch.ones_like(d_y_tn), create_graph=True)[0]

    # convert to physical time derivatives via chain rule
    dx_dt = d_x_tn
    dy_dt = d_y_tn
    dx_ddt = d_x_ttn
    dy_ddt = d_y_ttn

    # physics residual
    r_x = M * dx_ddt + C * dx_dt + K * d_x - h_x
    r_y = M * dy_ddt + C * dy_dt + K * d_y - h_y

    resid = torch.stack([r_x, r_y], dim=1)
    return y, resid
