import numpy as np 
import torch, time
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

def leastsquare(model, body, bdry, params, device, problem, bdry_penalty="False", display="True", norm=True):
    N = body.shape[0]
    u = model(body)[0]
    A_tem = torch.zeros(N,u.shape[1]).to(device)
    if problem.name == "helmholtz1d":
        for i in range(u.shape[1]):
            grad = torch.autograd.grad(u[:,i], body,grad_outputs=torch.ones_like(u[:,i]),create_graph=True)[0]
            grad2 = torch.autograd.grad(grad, body,grad_outputs=torch.ones_like(grad),retain_graph=True)[0] 
            A_tem[:,i] = grad2.detach().squeeze(1) 
        A_body = (A_tem- params["lambda"]*u)
    
    if problem.name == "poisson2d":
        for i in range(u.shape[1]):
            dudxy = torch.autograd.grad(u[:,i],body,grad_outputs=torch.ones_like(u[:,i]),create_graph=True)[0]
            d2udxx2 = torch.autograd.grad(dudxy[:,0],body,grad_outputs=torch.ones_like(dudxy[:,0]),create_graph=True)[0]
            dxx = d2udxx2[:,0]
            d2udyy2 = torch.autograd.grad(dudxy[:,1],body,grad_outputs=torch.ones_like(dudxy[:,1]),create_graph=True)[0]
            dyy = d2udyy2[:,1]
            A_tem[:,i] = (- dxx - dyy).detach()
        A_body = A_tem
    
    if problem.name == 'poisson1d':
        for i in range(u.shape[1]):
            dudxy = torch.autograd.grad(u[:,i],body,grad_outputs=torch.ones_like(u[:,i]),create_graph=True)[0]
            d2udxx2 = torch.autograd.grad(dudxy[:,0],body,grad_outputs=torch.ones_like(dudxy[:,0]),create_graph=True)[0]
            dxx = d2udxx2
            A_tem[:,i] = (- dxx).detach().squeeze(1)
        A_body = A_tem
        
    if problem.name == 'poisson3d':
        for i in range(u.shape[1]):  
            dudxyz = torch.autograd.grad(u[:, i], body,grad_outputs=torch.ones_like(u[:, i]),create_graph=True)[0]  # shape: [N, 3]
            d2udx2 = torch.autograd.grad(dudxyz[:, 0], body,grad_outputs=torch.ones_like(dudxyz[:, 0]),retain_graph=True)[0][:, 0]  # 取第二次对 x 导
            d2udy2 = torch.autograd.grad(dudxyz[:, 1], body,grad_outputs=torch.ones_like(dudxyz[:, 1]),retain_graph=True)[0][:, 1]  # 取第二次对 y 导
            d2udz2 = torch.autograd.grad(dudxyz[:, 2], body,grad_outputs=torch.ones_like(dudxyz[:, 2]),retain_graph=True)[0][:, 2]  # 取第二次对 z 导
            laplace_u = d2udx2 + d2udy2 + d2udz2  # shape: [N]
            A_tem[:, i] = (-laplace_u).detach()
        A_body = A_tem

        
    f_body = problem.interior_condition(body)
    A_bdry = model(bdry)[0]
    f_bdry = problem.boundary_condition(bdry)
    if bdry_penalty == "True":
        A_bdry = A_bdry * body.shape[0]/bdry.shape[0]
        f_bdry = f_bdry * body.shape[0]/bdry.shape[0]
    if f_body is not None:
            if f_body.dim() > 1 and f_body.shape[1] == 1:
                f_body = f_body.squeeze(1)
    if f_bdry is not None:
            if f_bdry.dim() > 1 and f_bdry.shape[1] == 1:
                f_bdry = f_bdry.squeeze(1)
                
    A = torch.cat((A_body, A_bdry), dim=0)
    rhs = torch.cat((f_body, f_bdry), dim=0)
    if norm:
        A, rhs = normalization(A, rhs)
    start_time = time.time()
    A_np = A.cpu().detach().numpy()
    rhs_np = rhs.cpu().detach().numpy()
    w, residuals, rank, s = lstsq(A_np, rhs_np)
    end_time = time.time()
    error = A.cpu().detach().numpy() @ w - rhs.cpu().detach().numpy()
    if display:
        print("the error of least square in problem " + str(problem.name)+": ", np.mean(np.abs(error)))
        print("the condition number of matrix A in problem " + str(problem.name)+": ", np.max(s)/np.min(s))
        print("the rank of matrix A in problem " + str(problem.name)+": ", torch.linalg.matrix_rank(A).cpu().detach().numpy())
    return w


def picard_least_squares(
    basis, data_points, boundary_points, boundary_values_fn,
    nu, max_iter=10, normalization=None, add_const=False
):
    device = data_points.device
    M = data_points.shape[0]
    B = boundary_points.shape[0]
    torch.cuda.empty_cache()  # 清除 GPU 显存缓存（不影响模型/变量）
    data_points = data_points.double().requires_grad_(True)
    boundary_points = boundary_points.double()

    # Basis evaluation
    phi_raw = basis(data_points)        # shape: [M, N]
    phi_bdry_raw = basis(boundary_points)
    N = phi_raw.shape[1]

    if add_const:
        ones_pde = torch.ones((M, 1), dtype=torch.float64, device=device)
        ones_bdry = torch.ones((B, 1), dtype=torch.float64, device=device)
        phi = torch.cat([phi_raw, ones_pde], dim=1)
        phi_bdry = torch.cat([phi_bdry_raw, ones_bdry], dim=1)
        N += 1
    else:
        phi = phi_raw
        phi_bdry = phi_bdry_raw
    

    # Precompute gradient and Laplacian of basis
    grad_phi = torch.zeros(M, N, 2, dtype=torch.float64, device=device)
    lap_phi = torch.zeros(M, N, dtype=torch.float64, device=device)
    for i in range(N):
        dphi = torch.autograd.grad(phi[:, i], data_points,
                                   grad_outputs=torch.ones_like(phi[:, i]),
                                   create_graph=True)[0]
        d2phi_x = torch.autograd.grad(dphi[:, 0], data_points,
                                      grad_outputs=torch.ones_like(dphi[:, 0]),
                                      retain_graph=True)[0][:, 0]
        d2phi_y = torch.autograd.grad(dphi[:, 1], data_points,
                                      grad_outputs=torch.ones_like(dphi[:, 1]),
                                      retain_graph=True)[0][:, 1]
        grad_phi[:, i, 0] = dphi[:, 0]
        grad_phi[:, i, 1] = dphi[:, 1]
        lap_phi[:, i] = d2phi_x + d2phi_y

    # Boundary target values
    u_bdry_target = boundary_values_fn(boundary_points).to(torch.float64)

    # Fixed boundary matrix blocks
    A4 = torch.cat([phi_bdry, torch.zeros_like(phi_bdry), torch.zeros_like(phi_bdry)], dim=1)
    A5 = torch.cat([torch.zeros_like(phi_bdry), phi_bdry, torch.zeros_like(phi_bdry)], dim=1)
    A6 = torch.cat([torch.zeros_like(phi_bdry), torch.zeros_like(phi_bdry), phi_bdry], dim=1)
    A_fixed = torch.cat([A4, A5, A6], dim=0)
    rhs_fixed = torch.cat([
        u_bdry_target[:, 0],
        u_bdry_target[:, 1],
        u_bdry_target[:, 2],
    ], dim=0)

    # Initialize coefficients
    a = torch.zeros(N, dtype=torch.float64, device=device)
    b = torch.zeros(N, dtype=torch.float64, device=device)
    c = torch.zeros(N, dtype=torch.float64, device=device)

    for it in range(max_iter):
        u_k = phi @ a  # shape: [M]
        v_k = phi @ b

        # NS residuals for u equation
        A_u = u_k.unsqueeze(1) * grad_phi[:, :, 0] + v_k.unsqueeze(1) * grad_phi[:, :, 1] - nu * lap_phi

        # NS residuals for v equation
        A_v = u_k.unsqueeze(1) * grad_phi[:, :, 0] + v_k.unsqueeze(1) * grad_phi[:, :, 1] - nu * lap_phi

        # ⛳ NOTE: Although grad_phi same structure used here, semantically A_u, A_v are different residuals.

        # Final block assembly
        Z = torch.zeros_like(A_u)
        A1 = torch.cat([A_u, Z, grad_phi[:, :, 0]], dim=1)
        A2 = torch.cat([Z, A_v, grad_phi[:, :, 1]], dim=1)
        A3 = torch.cat([
            grad_phi[:, :, 0],
            grad_phi[:, :, 1],
            torch.zeros_like(grad_phi[:, :, 0])
        ], dim=1)

        A_dynamic = torch.cat([A1, A2, A3], dim=0)
        rhs_dynamic = torch.zeros(3 * M, dtype=torch.float64, device=device)

        A = torch.cat([A_dynamic, A_fixed], dim=0)
        rhs = torch.cat([rhs_dynamic, rhs_fixed], dim=0)

        if normalization is not None:
            A, rhs = normalization(A, rhs)

        lstsq_result = torch.linalg.lstsq(A, rhs.unsqueeze(1))
        x = lstsq_result.solution.squeeze()

        a, b, c = x[:N], x[N:2 * N], x[2 * N:]
        residual = (A @ x - rhs)
        print(f"[Picard {it}] LS MSE: {torch.mean(residual**2).item():.2e}")

    return a, b, c, torch.mean(residual**2).item()


def normalization(A,b):
    A_norm = torch.max(torch.abs(A), axis=1)
    return A / A_norm.values.unsqueeze(1), b / A_norm.values

def trainnew(model,problem,device,params,optimizer):
    
    problem.generate_points(params["bodyBatch"], params["bdryBatch"], "random")
    int_data = problem.data()["interior_points"]
    bdry_data = problem.data()["boundary_points"]
    int_data.requires_grad = True
    bdry_data.requires_grad = True
    effectiveranks = []
    singular_values = []
    steps = []  
    initialloss = 1
    model.train()
    for step in range(params["trainStep"]):
        u = model(int_data)[1]
        u = u.squeeze(1) if len(u.shape) == 2 else u
        u_second = model(int_data)[0]
        u_bdry = model(bdry_data)[1]
        u_bdry = u_bdry.squeeze(1) if len(u_bdry.shape) == 2 else u_bdry
        model.zero_grad()
        problem.loss_int(u, int_data)
        problem.loss_bdry(u_bdry, bdry_data)
        loss_pde = problem.loss_pde()
        loss_orthogonal = torch.norm(u_second.T @ u_second - torch.eye(params["width"]).to(device).double(), p='fro')
        loss = loss_pde + loss_orthogonal*params["orthogonalpenalty"]
        
        if step == 1:
            initialloss = loss_pde.detach().cpu().numpy()
            # print("Initial pde loss is %s"%(initialloss))
        else:
            if loss_pde.detach().cpu().numpy()/initialloss < params["epsilon"]:
                break

        if step%params["writeStep"] == params["writeStep"]-1:
            effectiverank = torch.linalg.matrix_rank(u_second.T@u_second, tol=0.001)
            effectiveranks.append(effectiverank.cpu().numpy())
            print("Loss at Step %s is %s with pde loss %s , orthogonal loss %s and rank of matrix %s."%(step+1,loss.detach().cpu().numpy(),loss_pde.detach().cpu().numpy(),loss_orthogonal.detach().cpu().numpy(),effectiverank.cpu().numpy()))


        if step%params["plotStep"] == params["plotStep"]-1:
            tem = lambda x: model(x)[1]
            problem.plot(tem)
            # U, S, Vh = torch.linalg.svd(u_second)
            # singular_values.append(S.detach().cpu().numpy())
            # steps.append([step] * len(S))  
        loss.backward()
        optimizer.step()
    print("Final Loss at Step %s is %s with pde loss %s , orthogonal loss %s."%(step,loss.detach().cpu().numpy(),loss_pde.detach().cpu().numpy(),loss_orthogonal.detach().cpu().numpy()))
    print("after least squares")
    if problem.name == "navierstokes2d":
        # basis = lambda x: model(x)[0]
        # a,b,c,residual = picard_least_squares(basis,int_data, bdry_data, problem.boundary_condition, 1/40, params["max_iter"])
        # perdict = lambda x: torch.stack([torch.matmul(model(x)[0], torch.tensor(a).to(device)), torch.matmul(model(x)[0], torch.tensor(b).to(device)), torch.matmul(model(x)[0], torch.tensor(c).to(device))], dim=1)
        # problem.plot(perdict)
        a,b,c,residual = problem.picardleastsquareproblem(model, int_data, bdry_data)
        perdict = lambda x: torch.stack([torch.matmul(model(x)[0], torch.tensor(a).to(device)), torch.matmul(model(x)[0], torch.tensor(b).to(device)), torch.matmul(model(x)[0], torch.tensor(c).to(device))], dim=1)
        problem.plot(perdict)
    else:
        (A, rhs) = problem.leastsquareproblem(model, int_data, bdry_data)
        w, _, _, _ = lstsq(A, rhs)
        U, s, Vh = torch.linalg.svd(u_second.T@u_second)
        error = A @ w - rhs
        threshold = 1e-3
        effective_rank = (s > threshold).sum()
        print("the error of least square in problem " + str(problem.name)+": ", np.mean(np.abs(error)))
        print("the effective rank of basis in problem %s is %s with threshold %s" % (str(problem.name), effective_rank.detach().cpu().numpy(), threshold))
        perdict = lambda x: torch.matmul(model(x)[0], torch.tensor(w).to(device))
        problem.plot(perdict)
        
    if problem.name == "navierstokes2d":
        return effectiveranks, singular_values, residual
    else:
        return effectiveranks, singular_values


def generate_grf(points, length_scale=0.3, sigma=1.0):
    """
    生成一个 Gaussian Random Field。
    points: [N, 2]
    返回: [N]
    """
    N = points.shape[0]
    diff = points.unsqueeze(1) - points.unsqueeze(0)
    dist2 = (diff ** 2).sum(dim=-1)

    K = sigma ** 2 * torch.exp(-dist2 / (2 * length_scale ** 2))
    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(N, device=points.device))
    z = torch.randn(N, device=points.device)
    return L @ z  # shape: [N]
     
     
def evaluate_basis_projection_quality_with_sin(
    basis_matrix: torch.Tensor,
    interior_points: torch.Tensor,
    max_k: int = 10,
    plot: bool = False  # 是否绘制拟合曲线
):
    """
    用 sin(kx) 测试 basis_matrix 的拟合质量

    Parameters:
        basis_matrix: [N, d]，倒数第二层在 interior_points 上的值
        interior_points: [N, 2]，对应的采样点
        max_k: 使用的最大频率 k，k=1 到 max_k

    Returns:
        {
            'errors': list of float,
            'rank': int,
            'mean_error': float,
        }
    """
    torch.set_default_dtype(torch.float64)
    x = interior_points[:, 0]  

    coeff_list = []
    error_list = []
    Φ = basis_matrix.detach()

    for k in range(1, max_k + 1):
        target = torch.sin(torch.pi * (x+1) * k / 2)  
        coeff = torch.linalg.lstsq(Φ, target).solution
        coeff_list.append(coeff)

        projection = Φ @ coeff
        error = torch.mean((projection - target)**2).item()
        error_list.append(error)

        rank = torch.linalg.matrix_rank(Φ, tol=1e-3).item()

    
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, max_k + 1), error_list, marker='o')
        plt.xlabel('k (frequency in sin(kx))')
        plt.ylabel('Mean Squared Error')
        plt.title('Basis Projection Error vs Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    return {
        'errors': error_list,
        'mean_error': sum(error_list) / len(error_list),
        'rank': rank,
    }
