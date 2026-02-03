import numpy as np 
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky


class PD_OFM(torch.nn.Module):
    def __init__(self, params, connection="residual", initialization="xavier", activation="tanh", bias_index=False):
        super(PD_OFM, self).__init__()

        activation_dict = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        self.modelname = "PD_OFM"
        self.connections = connection
        self.act = activation_dict.get(activation.lower(), nn.Tanh())
        self.initialization = initialization
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))
        self.linearOut = nn.Linear(self.params["width"], params.get("output", 1), bias=bias_index)
        self.initialize_weights()
        
    def forward(self, x):
        if self.connections == "direct":
            x = self.act(self.linearIn(x))
            for layer in self.linear:
                x = self.act(layer(x))
            basis = x 
            x = self.linearOut(x)
        else:
            x = self.linearIn(x)
            for layer in self.linear:
                x = F.tanh(layer(x))**3 + x
            basis = x 
            x = self.linearOut(x)
        return basis, x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.initialization == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / np.sqrt(fan_in)
                        nn.init.uniform_(m.bias, -bound, bound)

                elif self.initialization == "xavier":
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

                elif self.initialization == "uniform":
                    # naive uniform initialization over [-1.0, 1.0]
                    nn.init.uniform_(m.weight, -1.0, 1.0)
                    if m.bias is not None:
                        nn.init.uniform_(m.bias, -1.0, 1.0)

    def grad(self, x, j):
        """
        Compute the gradient of the j-th neuron in the penultimate layer of PD_OFM
        with respect to the input x.

        Args:
            x: (N, d) Input tensor.  
            j: int Index of the target neuron.  

        Returns:
            grad: (N, d) Gradient of the selected neuron with respect to the input.  
        """
        x.requires_grad_(True)
        basis, _ = self.forward(x)  
        target_neuron = basis[:, j]  
        grad = torch.autograd.grad(
            target_neuron, x,
            grad_outputs=torch.ones_like(target_neuron),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]  # shape: (N, d)

        return grad
    
    def hessian(self, x, j):
        """
        Compute the Hessian matrix of the j-th neuron in the penultimate layer of PD_OFM
        with respect to the input x.

        Args:
            x: (N, d) Input tensor.  
            j: int Index of the target neuron.  

        Returns:
            hessian: (N, d, d) Hessian matrix of the selected neuron with respect to the input.  
        """

        x.requires_grad_(True)
        org, _ = self.forward(x)
        target_neuron = org[:, j]
        hessians = []
        for i in range(x.shape[1]):
            grad_i = torch.autograd.grad(
                target_neuron, x,
                grad_outputs=torch.ones_like(target_neuron),
                create_graph=True, only_inputs=True
            )[0][:, i]  # (N,)

            second_deriv = []
            for k in range(x.shape[1]):
                grad2 = torch.autograd.grad(
                    grad_i, x,
                    grad_outputs=torch.ones_like(grad_i),
                    create_graph=False, only_inputs=True
                )[0][:, k]  # (N,)
                second_deriv.append(grad2.unsqueeze(1))  # (N, 1)

            hessians.append(torch.cat(second_deriv, dim=1).unsqueeze(1))  # (N, 1, d)

        hessian = torch.cat(hessians, dim=1)  # (N, d, d)
        return hessian
    
    def laplacian(self, x, j):
        """
        Using PyTorch automatic differentiation, directly compute the Laplacian of the j-th neuron 
        in the penultimate layer of PD_OFM with respect to the input x.

        Args:
            x: (N, d) Input tensor.  
            j: int Index of the target neuron.  

        Returns:
            laplacian: (N,) Laplacian value for each sample.  
        """

        x.requires_grad_(True)
        basis, _ = self.forward(x)
        target_neuron = basis[:, j]  
        grads = torch.autograd.grad(
            target_neuron, x,
            grad_outputs=torch.ones_like(target_neuron),
            create_graph=True, only_inputs=True
        )[0]  # (N, d)

        laplacian = 0.0

        for i in range(x.shape[1]):
            grad2 = torch.autograd.grad(
                grads[:, i], x,
                grad_outputs=torch.ones_like(grads[:, i]), only_inputs=True, retain_graph=True
            )[0][:, i]  
            laplacian += grad2  

        return laplacian  

### TransNet
class TransNet(nn.Module):
    def __init__(self, x_dim, basis_num, radius, nlin_type='tanh', eta=0.3, K=5, gammas=torch.linspace(0.2, 5.0, 25), init_mode="default", shape_parameter=None, mesh_size=50):
        """
            x_dim: Input dimension  
            basis_num: Number of basis functions M  
            nlin_type: Nonlinear activation function ('tanh' or 'relu')  
            include_const: Whether to include a constant basis function  
            eta: Correlation length (controls the smoothness of the GRF)  
            K: Number of realizations for GRF sampling  
            gamma_range: Search range for γ (γ_min, γ_max)  
            S: Grid size for γ sampling  
            init_mode: Initialization mode ("default" = standard initialization, "grf" = GRF initialization)  
        """

        super(TransNet, self).__init__()
        self.x_dim = x_dim
        self.basis_num = basis_num
        self.radius = radius
        self.nlin_type = nlin_type
        self.eta = eta
        self.K = K
        self.gammas = gammas
        self.init_mode = init_mode  # 记录初始化模式
        self.shape_parameter = shape_parameter
        self.mesh_size = mesh_size
        self.center = torch.zeros(self.x_dim, dtype=torch.float64)

    
        if self.shape_parameter is None:
            self.init()
            self.shape_parameter = self.optimize_shape_parameter()
            self.init(self.shape_parameter)
        else:
            self.init(self.shape_parameter)

        if self.nlin_type == 'tanh':
            self.nonlinear = torch.tanh
        elif self.nlin_type == 'relu':
            self.nonlinear = torch.relu
        else:
            raise ValueError(f"Unsupported non-linearity: {self.nlin_type}")
        
    def set_center(self, center):
        self.center = center
        
    def init(self, shape_parameter=1.0):
        """ 
            radius: corresponds to the domain of the problem, we require the ball of this radius can fully cover the domain
            shape_parameter: needs to be optimized
        """
        
        weight = np.random.randn(self.basis_num, self.x_dim)
        weight = weight / np.sqrt(np.sum(weight ** 2, axis=1, keepdims=True))  # 归一化
        
        b = np.random.rand(self.basis_num)*self.radius
        
        self.weight = nn.Parameter(torch.tensor(weight * shape_parameter, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(b * shape_parameter, dtype=torch.float64))

    def generate_grf(self, N, correlation_length=1.0, sigma=1.0):
        """
        Generate N Gaussian Random Fields (GRF) on given points using gstools.
        
        Args:
            mesh_size (int): Mesh size of the unit cube
            N (int): Number of GRF samples to generate
            correlation_length (float): Correlation length parameter
            sigma (float): Standard deviation of the Gaussian Random Field
            
        Returns:
            torch.Tensor: GRF samples of shape [N, M]
        """
        import gstools as gs
        
        # Convert points to numpy array
        x_1d = np.linspace(-1, 1, self.mesh_size)
        
        # Create a Gaussian covariance model
        model = gs.Gaussian(dim=self.x_dim, var=sigma**2, len_scale=correlation_length)
        
        # Create a SRF generator
        srf = gs.SRF(model)
        samples = [x_1d for _ in range(self.x_dim)]
        srf.set_pos(samples, "structured")

        # 生成N个实现
        grf_samples = []
        for i in range(N):
            field = srf(seed=i)  # 直接生成场
            grf_samples.append(field.flatten())  # 将2D场展平
            
        grf_samples = torch.tensor(np.array(grf_samples), dtype=torch.float64)
        
        return grf_samples
    
    def optimize_shape_parameter(self, correlation_length=1.0, sigma=1.0):
        """
        Optimize the shape parameter gamma using the algorithm from the image.
        """
        avgMSEs = []
        axes_1d = [np.linspace(-1, 1, self.mesh_size) for _ in range(self.x_dim)]
        mesh = np.meshgrid(*axes_1d, indexing='ij')
        points = np.stack([m.flatten() for m in mesh], axis=1)
        points = torch.tensor(points, dtype=torch.float64)

        for gamma in self.gammas:
            grf_sample = self.generate_grf(1, correlation_length, sigma).squeeze(0) 
            weight = (self.weight * gamma.item()).clone().detach().to(dtype=torch.float64)
            bias = (self.bias * gamma.item()).clone().detach().to(dtype=torch.float64)
            z = torch.matmul(points, weight.T) + bias
            basis_eval = F.tanh(z) 
            alpha = torch.linalg.lstsq(basis_eval, grf_sample.T).solution  # (basis_num, 1)
            projected_grf = basis_eval @ alpha  # (N_samples, 1)
            mse = torch.mean((grf_sample.T - projected_grf) ** 2).item()
            avgMSEs.append(np.mean(mse))

        # Find the gamma with the lowest average MSE
        best_idx = int(np.argmin(avgMSEs))
        best_gamma = self.gammas[best_idx].item()

        plt.figure(figsize=(10, 6))
        plt.semilogy(self.gammas, avgMSEs)
        plt.xlabel('Gamma')
        plt.ylabel('Average Projection Error (log scale)')
        plt.title('Error vs Gamma (Log Scale)')
        plt.grid(True)
        plt.show()

        return best_gamma
    
    def forward(self, x):
        z = torch.matmul(x-self.center, self.weight.T) + self.bias
        return self.nonlinear(z), None
    
    def grad(self, x, j):
        """
        Compute the derivative of the j-th neuron with respect to the input x.

        Args:
            x: (N, x_dim) Input sample points.  
            j: int Index of the target neuron.  

        Returns:
            grad: (N, x_dim)  Derivative of the j-th neuron with respect to x.  
        """

        x_shifted = x - self.center  # (N, x_dim)
        z_j = torch.matmul(x_shifted, self.weight[j].unsqueeze(0).T) + self.bias[j]  # (N, 1)

        if self.nlin_type == 'tanh':
            activation_derivative = 1 - torch.tanh(z_j) ** 2  # (N, 1)
        elif self.nlin_type == 'relu':
            activation_derivative = (z_j > 0).double()  # (N, 1)
        else:
            raise ValueError(f"Unsupported non-linearity: {self.nlin_type}")

        # grad shape: (N, x_dim)
        grad = activation_derivative * self.weight[j]  # 自动广播

        return grad
    
    def hessian(self, x, j):
        """
        Compute the Hessian matrix of the j-th neuron with respect to the input x.

        Args:
            x: (N, x_dim) Input sample points.  
            j: int Index of the target neuron.  

        Returns:
            hessians: (N, x_dim, x_dim) Hessian matrix of the j-th neuron for each sample.  
        """

        x_shifted = x - self.center  # (N, x_dim)
        w_j = self.weight[j].unsqueeze(0)  # (1, x_dim)
        z_j = torch.matmul(x_shifted, w_j.T) + self.bias[j]  # (N, 1)

        if self.nlin_type == 'tanh':
            sigma_second_derivative = -2 * torch.tanh(z_j) * (1 - torch.tanh(z_j) ** 2)  # (N, 1)
        elif self.nlin_type == 'relu':
            sigma_second_derivative = torch.zeros_like(z_j)  # ReLU 的二阶导为 0
        else:
            raise ValueError(f"Unsupported non-linearity: {self.nlin_type}")

        # 外积 w_j w_j^T，结果形状 (x_dim, x_dim)
        w_outer = torch.ger(self.weight[j], self.weight[j])  # (x_dim, x_dim)

        # Hessian: 每个样本的 Hessian 是 σ''(z_j) * (w_j w_j^T)
        hessians = sigma_second_derivative.view(-1, 1, 1) * w_outer.unsqueeze(0)  # (N, x_dim, x_dim)

        return hessians
    
    def laplacian(self, x, j):
        """
        Compute the Laplacian of the j-th neuron with respect to the input x.

        Args:
            x: (N, x_dim) Input sample points.
            j: int Index of the target neuron.

        Returns:
            laplacian: (N,) Laplacian value of the j-th neuron for each sample.  
        """

        x_shifted = x - self.center  # (N, x_dim)
        w_j = self.weight[j]  # (x_dim,)
        z_j = torch.matmul(x_shifted, w_j.unsqueeze(0).T) + self.bias[j]  # (N, 1)

        if self.nlin_type == 'tanh':
            sigma_second_derivative = -2 * torch.tanh(z_j) * (1 - torch.tanh(z_j) ** 2)  # (N, 1)
        elif self.nlin_type == 'relu':
            sigma_second_derivative = torch.zeros_like(z_j)  # ReLU 的二阶导为 0
        else:
            raise ValueError(f"Unsupported non-linearity: {self.nlin_type}")

        # Laplacian: σ''(z_j) * ||w_j||^2
        weight_norm_squared = torch.sum(w_j ** 2)  # scalar

        laplacian = sigma_second_derivative.squeeze(1) * weight_norm_squared  # (N,)

        return laplacian


class PsiB:
    def __init__(self, center: torch.Tensor, radius: torch.Tensor, dtype=torch.float64):
        self.center = center.to(dtype=dtype)
        self.radius = radius.to(dtype=dtype)
        self.dtype = dtype

    def _x_tilde(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center) / self.radius

    def func(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        xtilde = self._x_tilde(x)
        result = torch.ones(x.shape[0], dtype=self.dtype, device=device)

        for d in range(x.shape[1]):
            xi = xtilde[:, d]
            val = torch.zeros_like(xi)

            mask1 = (-1.25 <= xi) & (xi < -0.75)
            val[mask1] = 0.5 * (1 + torch.sin(2 * math.pi * xi[mask1]))

            mask2 = (-0.75 <= xi) & (xi < 0.75)
            val[mask2] = 1.0

            mask3 = (0.75 <= xi) & (xi < 1.25)
            val[mask3] = 0.5 * (1 - torch.sin(2 * math.pi * xi[mask3]))

            result *= val

        return result

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        xtilde = self._x_tilde(x)
        N, d = x.shape

        phi = torch.ones(N, dtype=self.dtype, device=device)
        dphi = torch.zeros(N, d, dtype=self.dtype, device=device)

        for i in range(d):
            xi = xtilde[:, i]
            val = torch.zeros_like(xi)
            dval = torch.zeros_like(xi)

            mask1 = (-1.25 <= xi) & (xi < -0.75)
            val[mask1] = 0.5 * (1 + torch.sin(2 * math.pi * xi[mask1]))
            dval[mask1] = math.pi * torch.cos(2 * math.pi * xi[mask1]) / self.radius[i]

            mask2 = (-0.75 <= xi) & (xi < 0.75)
            val[mask2] = 1.0
            dval[mask2] = 0.0

            mask3 = (0.75 <= xi) & (xi < 1.25)
            val[mask3] = 0.5 * (1 - torch.sin(2 * math.pi * xi[mask3]))
            dval[mask3] = -math.pi * torch.cos(2 * math.pi * xi[mask3]) / self.radius[i]

            phi *= val

            prod_others = torch.ones_like(xi)
            for j in range(d):
                if j != i:
                    xj = xtilde[:, j]
                    temp = torch.zeros_like(xj)

                    mask1 = (-1.25 <= xj) & (xj < -0.75)
                    temp[mask1] = 0.5 * (1 + torch.sin(2 * math.pi * xj[mask1]))

                    mask2 = (-0.75 <= xj) & (xj < 0.75)
                    temp[mask2] = 1.0

                    mask3 = (0.75 <= xj) & (xj < 1.25)
                    temp[mask3] = 0.5 * (1 - torch.sin(2 * math.pi * xj[mask3]))

                    prod_others *= temp

            dphi[:, i] = dval * prod_others

        return dphi

    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        xtilde = self._x_tilde(x)
        N, d = x.shape

        phi = torch.ones(N, dtype=self.dtype, device=device)
        dphi = torch.zeros(N, d, dtype=self.dtype, device=device)
        ddphi = torch.zeros(N, d, d, dtype=self.dtype, device=device)

        for i in range(d):
            xi = xtilde[:, i]
            val = torch.zeros_like(xi)
            dval = torch.zeros_like(xi)
            ddval = torch.zeros_like(xi)

            mask1 = (-1.25 <= xi) & (xi < -0.75)
            val[mask1] = 0.5 * (1 + torch.sin(2 * math.pi * xi[mask1]))
            dval[mask1] = math.pi * torch.cos(2 * math.pi * xi[mask1]) / self.radius[i]
            ddval[mask1] = -2 * math.pi ** 2 * torch.sin(2 * math.pi * xi[mask1]) / (self.radius[i] ** 2)

            mask2 = (-0.75 <= xi) & (xi < 0.75)
            val[mask2] = 1.0
            dval[mask2] = 0.0
            ddval[mask2] = 0.0

            mask3 = (0.75 <= xi) & (xi < 1.25)
            val[mask3] = 0.5 * (1 - torch.sin(2 * math.pi * xi[mask3]))
            dval[mask3] = -math.pi * torch.cos(2 * math.pi * xi[mask3]) / self.radius[i]
            ddval[mask3] = -2 * math.pi ** 2 * torch.sin(2 * math.pi * xi[mask3]) / (self.radius[i] ** 2)

            phi *= val

            prod_others = torch.ones_like(xi)
            for j in range(d):
                if j != i:
                    xj = xtilde[:, j]
                    temp = torch.zeros_like(xj)

                    mask1 = (-1.25 <= xj) & (xj < -0.75)
                    temp[mask1] = 0.5 * (1 + torch.sin(2 * math.pi * xj[mask1]))

                    mask2 = (-0.75 <= xj) & (xj < 0.75)
                    temp[mask2] = 1.0

                    mask3 = (0.75 <= xj) & (xj < 1.25)
                    temp[mask3] = 0.5 * (1 - torch.sin(2 * math.pi * xj[mask3]))

                    prod_others *= temp

            dphi[:, i] = dval * prod_others

            for j in range(d):
                if i == j:
                    ddphi[:, i, j] = ddval * prod_others
                else:
                    xj = xtilde[:, j]
                    temp = torch.zeros_like(xj)

                    mask1 = (-1.25 <= xj) & (xj < -0.75)
                    temp[mask1] = 0.5 * (1 + torch.sin(2 * math.pi * xj[mask1]))

                    mask2 = (-0.75 <= xj) & (xj < 0.75)
                    temp[mask2] = 1.0

                    mask3 = (0.75 <= xj) & (xj < 1.25)
                    temp[mask3] = 0.5 * (1 - torch.sin(2 * math.pi * xj[mask3]))

                    partial = dval * torch.ones_like(xj)
                    ddphi[:, i, j] = partial * dphi[:, j] / temp

        return ddphi

    
class localRFM(torch.nn.Module):
    def __init__(self, params, center, radius, connection="residual", initialization="uniform", activation="tanh", bias_index=False):
        super(localRFM, self).__init__()
        activation_dict = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.connections = connection
        self.act = activation_dict.get(activation.lower())
        self.initialization = initialization
        self.center = center
        self.radius = radius
        self.initialize_weights()
        
    def forward(self, x):
        x = self.act(self.linearIn(self.normalized(x)))
        return x
    
    def normalized(self, x):
        y = (x - self.center)/self.radius
        return y 
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.initialization == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / np.sqrt(fan_in)
                        nn.init.uniform_(m.bias, -bound, bound)

                elif self.initialization == "xavier":
                    gain = nn.init.calculate_gain('tanh')  # 或其他激活函数，比如 'relu'
                    nn.init.xavier_normal_(m.weight, gain=gain)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif self.initialization == "uniform":
                    nn.init.uniform_(m.weight, -1.0, 1.0)
                    if m.bias is not None:
                        nn.init.uniform_(m.bias, -1.0, 1.0)

class RFM(torch.nn.Module):
    def __init__(self, params, centers, radius, initialization="uniform", typeofPoU="a", device=None):
        """
        M: number of basis functions
        P: number of subdomains
        Q: number of basis functions in each subdomain
        """
        
        super(RFM, self).__init__()
        self.params = params
        self.M = params["number_basis"]
        self.P = centers.shape[0]
        self.Q = int(self.M / self.P)
        self.PoU = []
        self.localfeature = []
        self.params["depth"] = 1
        self.params["width"] = self.Q
        self.initialization = initialization
        self.typeofPoU = typeofPoU.lower()
        self.device = device
        for i in range(self.P):
            self.PoU.append(PsiB(center=centers[i], radius=radius[i]))
            self.localfeature.append(localRFM(self.params, center=centers[i], radius=radius[i], initialization=self.initialization).to(self.device))
                
    def forward(self, x):
        # overide the forward function to compute the output
        U = torch.zeros(x.shape[0], self.M).to(self.device)
        phis = [self.PoU[j].func(x) for j in range(self.P)]   
        phis = [phi.view(-1) for phi in phis] 
        row_sums = sum(phis)
        for i in range(self.P):
            phi_i = phis[i]/row_sums # partition of unity requires the coefficients to sum to 1]
            u = self.localfeature[i].forward(x)                 
            U[:, i*self.Q:(i+1)*self.Q] = u * phi_i.unsqueeze(1) 
        return U, None
      
    def grad(self, x, neuron_index):
        """
        Explicitly compute the first-order derivative of the neuron specified by 
        neuron_index with respect to the input x.
        """

        N, d = x.shape
        p_idx = neuron_index // self.Q
        q_idx = neuron_index % self.Q

        # 取 PoU 权重和导数
        phi_p = self.PoU[p_idx].func(x)  # (N,)
        dphi_p = self.PoU[p_idx].grad(x)  # (N, d)

        # 计算所有 PoU 总和及其导数
        phi_list = []
        dphi_list = []
        for j in range(self.P):
            phi_j = self.PoU[j].func(x)  # (N,)
            dphi_j = self.PoU[j].grad(x)  # (N, d)
            phi_list.append(phi_j)
            dphi_list.append(dphi_j)

        phi_sum = sum(phi_list)  # (N,)
        dphi_sum = sum(dphi_list)  # (N, d)

        # PoU 归一化权重及导数
        phi_normalized = phi_p / phi_sum  # (N,)
        dphi_normalized = (dphi_p * phi_sum.unsqueeze(1) - phi_p.unsqueeze(1) * dphi_sum) / (phi_sum.unsqueeze(1) ** 2)  # (N, d)

        # 取局部网络权重
        local_net = self.localfeature[p_idx]
        xtilde = local_net.normalized(x)  # (N, d)

        weight = local_net.linearIn.weight[q_idx]  # (d,)
        bias = local_net.linearIn.bias[q_idx]  # scalar

        z = (xtilde * weight).sum(dim=1) + bias  # (N,)
        sigma = torch.tanh(z)
        sigma_deriv = 1 - torch.tanh(z) ** 2  # (N,)

        grad = dphi_normalized * sigma.unsqueeze(1) + phi_normalized.unsqueeze(1) * sigma_deriv.unsqueeze(1) * weight.unsqueeze(0) / local_net.radius.unsqueeze(0)

        return grad
    
    def hessian(self, x, neuron_index):
        """
        Explicitly compute the second-order derivative (Hessian) of the neuron 
        specified by neuron_index with respect to the input x.
        """
        N, d = x.shape
        p_idx = neuron_index // self.Q
        q_idx = neuron_index % self.Q

        phi_p = self.PoU[p_idx].func(x)  # (N,)
        dphi_p = self.PoU[p_idx].grad(x)  # (N, d)
        ddphi_p = self.PoU[p_idx].hessian(x)  # (N, d, d)

        phi_list = []
        dphi_list = []
        ddphi_list = []
        for j in range(self.P):
            phi_j = self.PoU[j].func(x)  # (N,)
            dphi_j = self.PoU[j].grad(x)  # (N, d)
            ddphi_j = self.PoU[j].hessian(x)  # (N, d, d)

            phi_list.append(phi_j)
            dphi_list.append(dphi_j)
            ddphi_list.append(ddphi_j)

        phi_sum = sum(phi_list)  # (N,)
        dphi_sum = sum(dphi_list)  # (N, d)
        ddphi_sum = sum(ddphi_list)  # (N, d, d)

        phi_normalized = phi_p / phi_sum  # (N,)
        dphi_normalized = (dphi_p * phi_sum.unsqueeze(1) - phi_p.unsqueeze(1) * dphi_sum) / (phi_sum.unsqueeze(1) ** 2)  # (N, d)

        ddphi_normalized = (
            (ddphi_p * phi_sum.unsqueeze(1).unsqueeze(2) - 2 * dphi_p.unsqueeze(2) * dphi_sum.unsqueeze(1) - phi_p.unsqueeze(1).unsqueeze(2) * ddphi_sum)
            / (phi_sum.unsqueeze(1).unsqueeze(2) ** 2)
            + 2 * phi_p.unsqueeze(1).unsqueeze(2) * (dphi_sum.unsqueeze(1) * dphi_sum.unsqueeze(2)) / (phi_sum.unsqueeze(1).unsqueeze(2) ** 3)
        )

        local_net = self.localfeature[p_idx]
        xtilde = local_net.normalized(x)  # (N, d)

        weight = local_net.linearIn.weight[q_idx]  # (d,)
        bias = local_net.linearIn.bias[q_idx]  # scalar

        z = (xtilde * weight).sum(dim=1) + bias  # (N,)
        sigma = torch.tanh(z)
        sigma_deriv = 1 - torch.tanh(z) ** 2  # (N,)
        sigma_second_deriv = -2 * torch.tanh(z) * (1 - torch.tanh(z) ** 2)  # (N,)

        hessian = torch.zeros(N, d, d, device=x.device)
        for i in range(d):
            for j in range(d):
                term1 = ddphi_normalized[:, i, j] * sigma
                term2 = dphi_normalized[:, i] * sigma_deriv * weight[j] / local_net.radius[j]
                term3 = dphi_normalized[:, j] * sigma_deriv * weight[i] / local_net.radius[i]
                term4 = phi_normalized * sigma_second_deriv * weight[i] * weight[j] / (local_net.radius[i] * local_net.radius[j])
                hessian[:, i, j] = term1 + term2 + term3 + term4

        return hessian

    def laplacian(self, x, neuron_index):
        """
        Explicitly compute the Laplacian (diagonal of the Hessian only) of the neuron 
        specified by neuron_index with respect to the input x.
        """

        N, d = x.shape
        p_idx = neuron_index // self.Q
        q_idx = neuron_index % self.Q

        phi_p = self.PoU[p_idx].func(x)  # (N,)
        dphi_p = self.PoU[p_idx].grad(x)  # (N, d)
        ddphi_p = self.PoU[p_idx].hessian(x)  # (N, d, d)

        phi_list = []
        dphi_list = []
        ddphi_list = []
        for j in range(self.P):
            phi_j = self.PoU[j].func(x)  # (N,)
            dphi_j = self.PoU[j].grad(x)  # (N, d)
            ddphi_j = self.PoU[j].hessian(x)  # (N, d, d)

            phi_list.append(phi_j)
            dphi_list.append(dphi_j)
            ddphi_list.append(ddphi_j)

        phi_sum = sum(phi_list)  # (N,)
        dphi_sum = sum(dphi_list)  # (N, d)
        ddphi_sum = sum(ddphi_list)  # (N, d, d)

        phi_normalized = phi_p / phi_sum  # (N,)
        dphi_normalized = (dphi_p * phi_sum.unsqueeze(1) - phi_p.unsqueeze(1) * dphi_sum) / (phi_sum.unsqueeze(1) ** 2)  # (N, d)

        ddphi_normalized = (
            (ddphi_p * phi_sum.unsqueeze(1).unsqueeze(2) - 2 * dphi_p.unsqueeze(2) * dphi_sum.unsqueeze(1) - phi_p.unsqueeze(1).unsqueeze(2) * ddphi_sum)
            / (phi_sum.unsqueeze(1).unsqueeze(2) ** 2)
            + 2 * phi_p.unsqueeze(1).unsqueeze(2) * (dphi_sum.unsqueeze(1) * dphi_sum.unsqueeze(2)) / (phi_sum.unsqueeze(1).unsqueeze(2) ** 3)
        )

        local_net = self.localfeature[p_idx]
        xtilde = local_net.normalized(x)  # (N, d)

        weight = local_net.linearIn.weight[q_idx]  # (d,)
        bias = local_net.linearIn.bias[q_idx]  # scalar

        z = (xtilde * weight).sum(dim=1) + bias  # (N,)
        sigma = torch.tanh(z)
        sigma_deriv = 1 - torch.tanh(z) ** 2  # (N,)
        sigma_second_deriv = -2 * torch.tanh(z) * (1 - torch.tanh(z) ** 2)  # (N,)

        # term1: trace of ddphi_normalized * sigma
        term1 = torch.einsum('nii->n', ddphi_normalized) * sigma  # (N,)

        # term2 + term3: 2 * sum over diagonal of dphi_normalized * sigma' * weight / radius
        term2 = 2 * torch.sum(dphi_normalized * sigma_deriv.unsqueeze(1) * weight.unsqueeze(0) / local_net.radius.unsqueeze(0), dim=1)  # (N,)

        # term4: sum over diagonal of phi_normalized * sigma'' * weight^2 / radius^2
        term4 = phi_normalized * sigma_second_deriv * torch.sum((weight ** 2) / (local_net.radius ** 2))  # (N,)

        laplacian = term1 + term2 + term4  # (N,)

        return laplacian




