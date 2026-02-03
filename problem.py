import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from abc import ABC, abstractmethod
from scipy.linalg import lstsq

class Helmholtz:
    def __init__(self, interior_condition, boundary_condition, exact_solution, lower, upper, wavenumber=10, bdry_penalty=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.interior_condition = interior_condition
        self.boundary_condition = boundary_condition
        self.exact_solution = exact_solution
        self.boundary_points = torch.tensor([[lower], [upper]], device=self.device).double()
        self.name = "helmholtz1d"
        self.wavenumber = wavenumber
        self.lower = lower
        self.upper = upper
        self.bdry_penalty = bdry_penalty
        
    def generate_points(self, num_int, num_bdry=2, point_distribution='uniform'):
        """Generates interior points using either uniform or random distribution."""
        if point_distribution == 'uniform':
            self.interior_points = torch.linspace(self.lower, self.upper, num_int).reshape(-1, 1).to(self.device)
        elif point_distribution == 'random':
            self.interior_points = torch.sort(torch.rand(num_int, 1) * (self.upper - self.lower) + self.lower)[0].to(self.device)
        else:
            raise ValueError("Invalid point distribution method. Choose 'uniform' or 'random'.")
    
    def plot(self, function):
        """Plots the exact solution over the domain."""
        x_values = np.linspace(self.lower, self.upper, 2000).reshape(-1, 1)
        x_values_torch = torch.from_numpy(x_values).to(self.device)
        y_values = self.exact_solution(x_values_torch).squeeze(1)
        y_perdict = function(x_values_torch)
    
        if y_perdict is not None:
            if y_perdict.dim() > 1:
                y_perdict = y_perdict.squeeze(1)
        if y_perdict is not None:
            if y_perdict.dim() > 1:
                y_perdict = y_perdict.squeeze(1)
                
        pred_u = y_perdict.detach().cpu().numpy()
        exact_u = y_values.detach().cpu().numpy()
        error = abs(exact_u - pred_u)
        error_l2 = np.sqrt(np.mean(error**2))
        error_l2_rel = error_l2 / np.sqrt(np.mean(exact_u**2))
        print("the maximum pointwise error in problem " + str(self.name)+": ", np.max(error))
        print("the l2 relative error in problem " + str(self.name)+": ", error_l2_rel)
        plt.figure(figsize=(6, 4))
        plt.plot(x_values, y_values.detach().cpu().numpy(), label='Exact Solution', color='blue')
        plt.plot(x_values, y_perdict.detach().cpu().numpy(), label='Model Prediction', color='red')
        plt.xlabel("x")
        plt.ylabel("Solution")
        plt.title("Helmholtz Equation")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(6, 4))
        plt.plot(x_values, np.abs(y_values.detach().cpu().numpy()-y_perdict.detach().cpu().numpy()), label='Error', color='blue')
        plt.xlabel("x")
        plt.ylabel("Error")
        plt.title("Absolute Error")
        plt.legend()
        plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.show()
        
        return error_l2, error_l2_rel
    
    def data(self):
        """Returns the generated interior and boundary data points."""
        return {
            'interior_points': self.interior_points,
            'boundary_points': self.boundary_points
        }

    def loss_int(self, u, data):
        dudx = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),create_graph=True,only_inputs=True,allow_unused=True)[0]
        d2udxx2 = torch.autograd.grad(dudx,data,grad_outputs=torch.ones_like(dudx),create_graph=True,only_inputs=True,allow_unused=True)[0]
        loss_int = torch.mean((d2udxx2.squeeze(1) - self.wavenumber*u - self.interior_condition(data).squeeze(1))**2)
        self.loss_int_value = loss_int
        return loss_int
    
    def loss_bdry(self, u, data):
        loss_bdry = torch.mean((u - self.boundary_condition(data))**2)
        self.loss_bdry_value = loss_bdry
        return loss_bdry
    
    def loss_pde(self):
        return self.loss_int_value + self.loss_bdry_value*self.bdry_penalty
    
    def leastsquareproblem(self, model, body, bdry):
        basis = lambda x: model(x)[0]
        u_body = basis(body)
        Nb, Mb = u_body.shape[0], u_body.shape[1]
        A_body = torch.zeros(Nb, Mb, device=self.device)
        for i in range(Mb):
            laplacian = model.laplacian(body, i).unsqueeze(-1)
            A_body[:, i] = (laplacian - self.wavenumber * u_body[:, i].unsqueeze(-1)).detach().squeeze(1)
        f_body = self.interior_condition(body)
        A_bdry = basis(bdry)
        f_bdry = self.boundary_condition(bdry)
        if f_body is not None:
            if f_body.dim() > 1 and f_body.shape[1] == 1:
                f_body = f_body.squeeze(1)
        if f_bdry is not None:
                if f_bdry.dim() > 1 and f_bdry.shape[1] == 1:
                    f_bdry = f_bdry.squeeze(1)
                    
        A = torch.cat((A_body, A_bdry), dim=0)
        rhs = torch.cat((f_body, f_bdry), dim=0)
        A_np = A.cpu().detach().numpy()
        rhs_np = rhs.cpu().detach().numpy()
        c = 100.0
        for i in range(len(A_np)):
            ratio = c/abs(A_np[i,:]).max()
            A_np[i,:] = A_np[i,:]*ratio
            rhs_np[i] = rhs_np[i]*ratio
            
        return (A_np, rhs_np)


class PoissonBase(ABC):
    def __init__(self, interior_condition, boundary_condition, exact_solution, domain_size, bdry_penalty, device):
        self.device = device
        self.interior_condition = interior_condition
        self.boundary_condition = boundary_condition
        self.exact_solution = exact_solution
        self.domain_size = domain_size
        self.interior_points = None
        self.boundary_points = None
        self.bdry_penalty = bdry_penalty

    @abstractmethod
    def generate_points(self, *args, **kwargs):
        pass

    @abstractmethod
    def plot(self, function):
        pass

    def data(self):
        return {
            'interior_points': self.interior_points,
            'boundary_points': self.boundary_points
        }

    def loss_int(self, u, data):
        if data.shape[1] == 1:
            dudx = torch.autograd.grad(u, data, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True, allow_unused=True)[0]
            d2udxx2 = torch.autograd.grad(dudx, data, grad_outputs=torch.ones_like(dudx), create_graph=True, only_inputs=True, allow_unused=True)[0]
            loss_int = torch.mean((d2udxx2 + self.interior_condition(data))**2)
        elif data.shape[1] == 2:
            dudxy = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),create_graph=True)[0]
            d2udxx2 = torch.autograd.grad(dudxy,data,grad_outputs=torch.ones_like(dudxy),create_graph=True)[0]
            dxx = d2udxx2[:,0]
            d2udyy2 = torch.autograd.grad(dudxy[:,1],data,grad_outputs=torch.ones_like(dudxy[:,1]),create_graph=True)[0]
            dyy = d2udyy2[:,1]
            loss_int = torch.mean((dxx + dyy + self.interior_condition(data))**2)
        elif data.shape[1] == 3:
            dudxyz = torch.autograd.grad(u, data, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True, allow_unused=True)[0]
            d2udxx2 = torch.autograd.grad(dudxyz[:,0], data, grad_outputs=torch.ones_like(dudxyz[:,0]), create_graph=True, only_inputs=True, allow_unused=True)[0][:, 0]
            d2udyy2 = torch.autograd.grad(dudxyz[:,1], data, grad_outputs=torch.ones_like(dudxyz[:,1]), create_graph=True, only_inputs=True, allow_unused=True)[0][:, 1]
            d2udzz2 = torch.autograd.grad(dudxyz[:,2], data, grad_outputs=torch.ones_like(dudxyz[:,2]), create_graph=True, only_inputs=True, allow_unused=True)[0][:, 2]
            loss_int = torch.mean((d2udxx2 + d2udyy2 + d2udzz2 + self.interior_condition(data))**2)
        self.loss_int_value = loss_int
        return loss_int
    
    def loss_bdry(self, u, data):
        loss_bdry = torch.mean((u - self.boundary_condition(data))**2)
        self.loss_bdry_value = loss_bdry
        return loss_bdry
    
    def loss_pde(self):
        return self.loss_int_value + self.loss_bdry_value*self.bdry_penalty
    
    def leastsquareproblem(self, model, body, bdry):
        basis = lambda x: model(x)[0]
        u_body = basis(body)
        Nb, Mb = u_body.shape[0], u_body.shape[1]
        A_body = torch.zeros(Nb, Mb, device=self.device)
        
        for i in range(Mb):
            laplace = model.laplacian(body, i)  
            A_body[:, i] = (-laplace).detach()
            
        f_body = self.interior_condition(body)
        A_bdry = basis(bdry)
        f_bdry = self.boundary_condition(bdry)
        if f_body is not None:
            if f_body.dim() > 1 and f_body.shape[1] == 1:
                f_body = f_body.squeeze(1)
        if f_bdry is not None:
                if f_bdry.dim() > 1 and f_bdry.shape[1] == 1:
                    f_bdry = f_bdry.squeeze(1)
        
        A = torch.cat((A_body, A_bdry), dim=0)
        rhs = torch.cat((f_body, f_bdry), dim=0)
        A_np = A.cpu().detach().numpy()
        rhs_np = rhs.cpu().detach().numpy()
        c = 1 if body.shape[1] == 3 else 100
        for i in range(len(A_np)):
            ratio = c/abs(A_np[i,:]).max()
            A_np[i,:] = A_np[i,:]*ratio
            rhs_np[i] = rhs_np[i]*ratio
            
        return (A_np, rhs_np)
        
            

class Poisson1D(PoissonBase):
    def __init__(self, interior_condition, boundary_condition, exact_solution, domain_size=1.0, bdry_penalty=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(interior_condition, boundary_condition, exact_solution, domain_size, bdry_penalty, device)
        self.name = "poisson1d"
        
    def generate_points(self, num_int, num_bdry, point_distribution='uniform'):
        """在 [-domain_size, domain_size] 内部生成采样点"""
        if point_distribution == 'uniform':
            x = torch.linspace(-self.domain_size, self.domain_size, num_int)
        elif point_distribution == 'random':
            x = torch.rand(num_int) * 2 * self.domain_size - self.domain_size
        else:
            raise ValueError("Invalid point distribution method. Choose 'uniform' or 'random'.")
        self.boundary_points = torch.tensor([[-self.domain_size], [self.domain_size]], dtype=torch.float64).to(self.device)
        self.interior_points = x.reshape(-1, 1).to(self.device)

    def plot(self, function, plot=True):
        """绘制真实解 vs 预测解 vs 误差"""
        x_plot = np.linspace(-self.domain_size, self.domain_size, 2000).reshape(-1, 1)
        x_plot_torch = torch.from_numpy(x_plot).to(self.device)
        exact_u = self.exact_solution(x_plot_torch).squeeze(1).detach().cpu().numpy()
        pred_u = function(x_plot_torch)
        if pred_u is not None:
            if pred_u.dim() > 1 and pred_u.shape[1] == 1:
                pred_u = pred_u.squeeze(1)
        pred_u = pred_u.detach().cpu().numpy()
        error = abs(exact_u - pred_u)
        error_l2 = np.sqrt(np.mean(error**2))
        error_l2_rel = error_l2 / np.sqrt(np.mean(exact_u**2))
        
        if plot:
        # Absolute Error
            plt.figure(figsize=(6, 4))
            plt.plot(x_plot, error, label="|Error|", color="blue")
            plt.title("Absolute Error")
            plt.xlabel("x")
            plt.ylabel("Error")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        return error_l2, error_l2_rel
        
class PoissonBox(PoissonBase):
    def __init__(self, interior_condition, boundary_condition, exact_solution, domain_size=1.0, bdry_penalty=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(interior_condition, boundary_condition, exact_solution, domain_size, bdry_penalty, device)
        self.name = "poisson2d"
        
    def generate_points(self, num_interior, num_boundary, point_distribution='uniform'):
        """
        生成内部点和边界点。
        内部点生成方式可选：'uniform' 或 'random'
        """
        if point_distribution == 'uniform':
            n = int(num_interior ** 0.5)
            x = torch.linspace(-1, 1, n) * self.domain_size
            y = torch.linspace(-1, 1, n) * self.domain_size
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            self.interior_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(self.device)
        elif point_distribution == 'random':
            self.interior_points = (torch.rand(num_interior, 2) * 2 * self.domain_size - self.domain_size).to(self.device)
        else:
            raise ValueError("Invalid point distribution method. Choose 'uniform' or 'random'.")

        per_edge = num_boundary // 4
        x_b = torch.linspace(-1, 1, per_edge) * self.domain_size
        y_b = torch.linspace(-1, 1, per_edge) * self.domain_size

        left   = torch.stack([torch.full_like(y_b, -self.domain_size), y_b], dim=-1)
        right  = torch.stack([torch.full_like(y_b,  self.domain_size), y_b], dim=-1)
        bottom = torch.stack([x_b, torch.full_like(x_b, -self.domain_size)], dim=-1)
        top    = torch.stack([x_b, torch.full_like(x_b,  self.domain_size)], dim=-1)

        self.boundary_points = torch.cat([left, right, bottom, top], dim=0).to(self.device)


    def plot(self, function):
        
        n_plot = 100  # 画图用 50x50 采样点
        x = np.linspace(-1, 1, n_plot)
        y = np.linspace(-1, 1, n_plot)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        grid_points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        grid_points_torch = torch.from_numpy(grid_points).to(self.device)

        # 计算真实解 & 预测解
        Exact_u = self.exact_solution(grid_points_torch).reshape(n_plot, n_plot)
        Train_u = function(grid_points_torch)
        Train_u = Train_u.reshape(n_plot, n_plot)
        Error = torch.abs(Exact_u - Train_u)
        Exact_u = Exact_u.detach().cpu().numpy()
        Train_u = Train_u.detach().cpu().numpy()
        Error = Error.detach().cpu().numpy()
        l2_error = np.sqrt(np.mean(Error**2))
        l2_error_rel = l2_error / np.sqrt(np.mean(Exact_u**2))
        print("the l2 relative error in problem " + str(self.name)+": ", l2_error_rel)
        print("the maximum pointwise error in problem " + str(self.name)+": ", np.max(np.abs(Error)))
        
        
        plt.figure(figsize=(6, 5))
        cf1 = plt.contourf(xx, yy, Error, cmap="viridis", levels=100)
        plt.title("Absolute Error")
        cbar = plt.colorbar(cf1, label="Absolute Error")

        # === 调整科学计数法 "1e−6" 位置 ===
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_x(1.0)   # 控制左右位置（越大越往右）
        offset_text.set_y(2.5)  # 控制上下位置（>1 往上移动，默认大约是 1.0）

        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


        # === 第二张图：Exact Solution ===
        plt.figure(figsize=(6, 5))
        contour2 = plt.contourf(xx, yy, Exact_u, levels=100, cmap="viridis")
        plt.colorbar(contour2, label="Exact")
        plt.title("Exact Solution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # === 第三张图：Model Prediction ===
        plt.figure(figsize=(6, 5))
        contour3 = plt.contourf(xx, yy, Train_u, levels=100, cmap="viridis")
        plt.colorbar(contour3, label="Prediction")
        plt.title("Model Prediction")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    
    
class PoissonL(PoissonBase):
    def __init__(self, interior_condition, boundary_condition, exact_solution, domain_size=1.0, bdry_penalty=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(interior_condition, boundary_condition, exact_solution, domain_size, bdry_penalty, device)
        self.name = "poisson2d"

        
    def generate_points(self, num_int, num_boundary, point_distribution='uniform'):
        """在 L 形区域 [-1,1]^2 - [0,1]^2 内部生成采样点"""
        if point_distribution == 'uniform':
            n = int(num_int ** 0.5)  # 生成 n x n 个点
            x = np.linspace(-1, 1, n) * self.domain_size
            y = np.linspace(-1, 1, n) * self.domain_size
            xx, yy = np.meshgrid(x, y, indexing='ij')
            points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
            points_torch = torch.from_numpy(points).to(self.device)

            # **过滤掉 [0,1] × [0,1] 区域**
            mask = ~((points[:, 0] >= 0) & (points[:, 1] >= 0))
            self.interior_points = points_torch[mask].to(self.device)

        elif point_distribution == 'random':
            # 先多采样一些点，避免删除后数量不够
            points = (np.random.rand(num_int * 2, 2) * 2 - 1) * self.domain_size
            points_torch = torch.from_numpy(points).to(self.device)
            mask = ~((points[:, 0] >= 0) & (points[:, 1] >= 0))
            self.interior_points = points_torch[mask][:num_int]  # 只取前 num 个点，确保数量合适

        else:
            raise ValueError("Invalid point distribution method. Choose 'uniform' or 'random'.")
        
        x = torch.linspace(-1, 1, num_boundary).to(self.device) * self.domain_size
        y = torch.linspace(-1, 1, num_boundary).to(self.device) * self.domain_size
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # **完整的 L 形边界**
        left = torch.stack([xx[:, 0], yy[:, 0]], dim=-1)  # 左边界 (x=-1, 全部保留)
        right = torch.stack([xx[:num_boundary//2, -1], yy[:num_boundary//2, -1]], dim=-1)  # 右边界 (x=1, 但只保留 y ≤ 0)
        bottom = torch.stack([xx[0, :], yy[0, :]], dim=-1)  # 下边界 (y=-1, 全部保留)
        top = torch.stack([xx[-1, :num_boundary//2], yy[-1, :num_boundary//2]], dim=-1)  # 上边界 (y=1, 但只保留 x ≤ 0)
        mid_vertical = torch.stack([torch.zeros(num_boundary//2), torch.linspace(0, 1, num_boundary//2)], dim=-1).to(device=self.device)
        mid_horizontal = torch.stack([torch.linspace(0, 1, num_boundary//2), torch.zeros(num_boundary//2)], dim=-1).to(device=self.device)
        self.boundary_points = torch.cat([left, right, bottom, top, mid_vertical, mid_horizontal])
        self.boundary_points.requires_grad = True

    def plot(self, function):
        n_plot = 50  # 画图用 50x50 采样点
        x = np.linspace(-1, 1, n_plot) * self.domain_size
        y = np.linspace(-1, 1, n_plot) * self.domain_size
        xx, yy = np.meshgrid(x, y, indexing='ij')
        grid_points = np.stack([xx.flatten(), yy.flatten()], axis=-1)

        # **去掉 L 形之外的点**
        mask = ~((grid_points[:, 0] >= 0) & (grid_points[:, 1] >= 0))
        valid_points = grid_points[mask]
        valid_points_torch = torch.from_numpy(valid_points).to(self.device)

        # 计算真实解 & 预测解
        Exact_u = self.exact_solution(valid_points_torch)
        Train_u = function(valid_points_torch).view(-1)
        Error = torch.abs(Exact_u - Train_u)

        # **重塑成网格**
        Exact_u_full = torch.full((n_plot * n_plot,), float("nan"), device=self.device)
        Train_u_full = torch.full((n_plot * n_plot,), float("nan"), device=self.device)
        Error_full = torch.full((n_plot * n_plot,), float("nan"), device=self.device)

        Exact_u_full[mask] = Exact_u
        Train_u_full[mask] = Train_u
        Error_full[mask] = Error

        Exact_u_full = Exact_u_full.reshape(n_plot, n_plot).cpu().detach().numpy()
        Train_u_full = Train_u_full.reshape(n_plot, n_plot).detach().cpu().numpy()
        Error_full = Error_full.reshape(n_plot, n_plot).detach().cpu().numpy()
        l2_error = np.sqrt(np.nanmean(Error_full**2))
        l2_error_rel = l2_error / np.sqrt(np.nanmean(Exact_u_full**2))
        print("the l2 relative error in problem " + str(self.name)+": ", l2_error_rel)
        print("the maximum pointwise error in problem " + str(self.name)+": ", np.nanmax(np.abs(Error_full)))
        
        # === 第一张图：Absolute Error ===
        plt.figure(figsize=(6, 5))
        cf1 = plt.contourf(xx, yy, Error_full, cmap="viridis", levels=100)
        plt.title("Absolute Error")
        cbar = plt.colorbar(cf1, label="Absolute Error")

        # 只移动 colorbar 的 “1e−10”
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_x(1.0)   # 数值越大越往右移
        offset_text.set_y(2.5)  # 数值越大越往上移

        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # === 第二张图：Exact Solution ===
        plt.figure(figsize=(6, 5))
        contour2 = plt.contourf(xx, yy, Exact_u_full, levels=100, cmap="viridis")
        plt.colorbar(contour2, label="Exact")
        plt.title("Exact Solution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # === 第三张图：Model Prediction ===
        plt.figure(figsize=(6, 5))
        contour3 = plt.contourf(xx, yy, Train_u_full, levels=100, cmap="viridis")
        plt.colorbar(contour3, label="Prediction")
        plt.title("Model Prediction")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()



class PoissonAnnulus(PoissonBase):
    def __init__(self, interior_condition, boundary_condition, exact_solution, R_in=0.5, R_out=1.0, bdry_penalty=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(interior_condition, boundary_condition, exact_solution, None, bdry_penalty, device)
        self.name = "poisson2d"
        self.R_in = R_in
        self.R_out = R_out

    def generate_points(self, num_int, num_boundary, point_distribution='uniform'):
        """ 在环形区域 R_in ≤ r ≤ R_out 生成内部采样点 """
        if point_distribution == 'uniform':
            max_trials = int(num_int * 20)  # 多生成一些保证采样成功
            
            x = torch.rand(max_trials) * 2 - 1
            y = torch.rand(max_trials) * 2 - 1
            r = torch.sqrt(x ** 2 + y ** 2)
            mask = (r >= self.R_in) & (r <= self.R_out)
            valid_points = torch.stack([x[mask], y[mask]], dim=-1)
            if valid_points.shape[0] < num_int:
                raise RuntimeError("Not enough valid points sampled, try increasing max_trials or check R_in/R_out.")
            
            self.interior_points = valid_points[:num_int].to(self.device)


        elif point_distribution == 'random':
            points = []
            while len(points) < num_int:
                r = torch.sqrt(torch.rand(num_int * 2) * (self.R_out ** 2 - self.R_in ** 2) + self.R_in ** 2)
                theta = torch.rand(num_int * 2) * 2 * np.pi
                x = r * torch.cos(theta)
                y = r * torch.sin(theta)
                points.append(torch.stack([x, y], dim=-1))

            self.interior_points = torch.cat(points)[:num_int].to(self.device)

        else:
            raise ValueError("Invalid point distribution method. Choose 'uniform' or 'random'.")
        
        # **生成环形边界点**
        theta = torch.linspace(0, 2 * np.pi, num_boundary).to(self.device)
        x_outer = self.R_out * torch.cos(theta)
        y_outer = self.R_out * torch.sin(theta)
        x_inner = self.R_in * torch.cos(theta)
        y_inner = self.R_in * torch.sin(theta)

        # **边界点合并**
        self.boundary_points = torch.cat([
            torch.stack([x_outer, y_outer], dim=-1),
            torch.stack([x_inner, y_inner], dim=-1)
        ]).to(self.device)
        

    def plot(self, function):
        n_plot = 100  # 画图用 100x100 采样点
        x = np.linspace(-self.R_out, self.R_out, n_plot)
        y = np.linspace(-self.R_out, self.R_out, n_plot)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        grid_points = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        grid_points_torch = torch.from_numpy(grid_points).to(self.device)

        # **去掉环形外的点**
        r = np.sqrt(grid_points[:, 0] ** 2 + grid_points[:, 1] ** 2)
        mask = (r >= self.R_in) & (r <= self.R_out)  # 只保留在环形区域内的点
        valid_points = grid_points[mask]
        valid_points_torch = torch.from_numpy(valid_points).to(self.device)

        # 计算真实解 & 预测解
        Exact_u = self.exact_solution(valid_points_torch)
        Train_u = function(valid_points_torch  ).view(-1)
        Error = torch.abs(Exact_u - Train_u)

        # **填充到完整网格**
        Exact_u_full = torch.full((n_plot * n_plot,), float("nan"), device=self.device)
        Train_u_full = torch.full((n_plot * n_plot,), float("nan"), device=self.device)
        Error_full = torch.full((n_plot * n_plot,), float("nan"), device=self.device)

        Exact_u_full[mask] = Exact_u
        Train_u_full[mask] = Train_u
        Error_full[mask] = Error

        Exact_u_full = Exact_u_full.reshape(n_plot, n_plot).detach().cpu().numpy()
        Train_u_full = Train_u_full.reshape(n_plot, n_plot).detach().cpu().numpy()
        Error_full = Error_full.reshape(n_plot, n_plot).detach().cpu().numpy()

        l2_error = np.sqrt(np.nanmean(Error_full**2))
        l2_error_rel = l2_error / np.sqrt(np.nanmean(Exact_u_full**2))
        print("the l2 relative error in problem " + str(self.name)+": ", l2_error_rel)
        print("the maximum pointwise error in problem " + str(self.name)+": ", np.nanmax(np.abs(Error_full)))

        # === 第一张图：Absolute Error ===
        plt.figure(figsize=(6, 5))
        cf1 = plt.contourf(xx, yy, Error_full, cmap="viridis", levels=100)
        plt.title("Absolute Error")
        cbar = plt.colorbar(cf1, label="Absolute Error")

        # 只移动 colorbar 的 “1e−10”
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_x(1.0)   # 数值越大越往右移
        offset_text.set_y(2.5)  # 数值越大越往上移

        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # === 第二张图：Exact Solution ===
        plt.figure(figsize=(6, 5))
        contour2 = plt.contourf(xx, yy, Exact_u_full, levels=100, cmap="viridis")
        plt.colorbar(contour2, label="Exact")
        plt.title("Exact Solution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # === 第三张图：Model Prediction ===
        plt.figure(figsize=(6, 5))
        contour3 = plt.contourf(xx, yy, Train_u_full, levels=100, cmap="viridis")
        plt.colorbar(contour3, label="Prediction")
        plt.title("Model Prediction")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


    
class NavierStokes:
    def __init__(self, interior_condition, boundary_condition, exact_solution, nu=1/40, bdry_penalty=1.0,
                 x_range=[-0.5, 1], y_range=[-0.5, 1.5], device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.interior_condition = interior_condition
        self.boundary_condition = boundary_condition
        self.exact_solution = exact_solution
        self.x_range = x_range
        self.y_range = y_range
        self.name = "navierstokes2d"
        self.nu = nu
        self.bdry_penalty = bdry_penalty

    def generate_points(self, num_int, num_boundary, mode='uniform'):
        if mode == 'uniform':
            # 在 x 和 y 方向上生成等间距的点，形成规则网格
            num_per_axis = int(num_int ** 0.5)
            x_lin = torch.linspace(self.x_range[0], self.x_range[1], num_per_axis)
            y_lin = torch.linspace(self.y_range[0], self.y_range[1], num_per_axis)
            xx, yy = torch.meshgrid(x_lin, y_lin, indexing='ij')  # 笛卡尔积
            points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
            self.interior_points = points

        elif mode == 'random':
            # 在矩形区域中均匀随机采样
            x = torch.rand(num_int) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
            y = torch.rand(num_int) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
            self.interior_points = torch.stack([x, y], dim=-1).to(self.device)

        else:
            raise ValueError(f"Sampling mode '{mode}' not supported. Use 'uniform' or 'random'.")
        
        xb = torch.linspace(self.x_range[0], self.x_range[1], num_boundary//4).to(self.device)
        yb = torch.linspace(self.y_range[0], self.y_range[1], num_boundary//4).to(self.device)

        # 四条边上的点
        bottom = torch.stack([xb, torch.full_like(xb, self.y_range[0])], dim=-1)
        top    = torch.stack([xb, torch.full_like(xb, self.y_range[1])], dim=-1)
        left   = torch.stack([torch.full_like(yb, self.x_range[0]), yb], dim=-1)
        right  = torch.stack([torch.full_like(yb, self.x_range[1]), yb], dim=-1)

        self.boundary_points = torch.cat([bottom, top, left, right], dim=0).to(self.device)


    def plot(self, function, individually_show=False):
        n_plot = 100
        x = torch.linspace(self.x_range[0], self.x_range[1], n_plot).to(self.device)
        y = torch.linspace(self.y_range[0], self.y_range[1], n_plot).to(self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(self.device)

        # 获取真实解 & 预测值
        Exact_uvp = self.exact_solution(grid_points)  # [N, 3]
        Pred_uvp = function(grid_points)              # [N, 3]
        Error_uvp = torch.abs(Exact_uvp - Pred_uvp)   # [N, 3]

        def fill_to_grid(data_col):
            full = torch.full((n_plot * n_plot,), float("nan"), device=self.device)
            full[:] = data_col.squeeze()
            return full.reshape(n_plot, n_plot).cpu().detach().numpy()

        u_exact = fill_to_grid(Exact_uvp[:, 0])
        v_exact = fill_to_grid(Exact_uvp[:, 1])
        p_exact = fill_to_grid(Exact_uvp[:, 2])

        u_pred = fill_to_grid(Pred_uvp[:, 0])
        v_pred = fill_to_grid(Pred_uvp[:, 1])
        p_pred = fill_to_grid(Pred_uvp[:, 2])

        u_err = fill_to_grid(Error_uvp[:, 0])
        v_err = fill_to_grid(Error_uvp[:, 1])
        p_err = fill_to_grid(Error_uvp[:, 2])

        xx = xx.cpu().detach().numpy()
        yy = yy.cpu().detach().numpy()

        titles = [
            ["u - Exact", "u - Prediction", "u - Error"],
            ["v - Exact", "v - Prediction", "v - Error"],
            ["p - Exact", "p - Prediction", "p - Error"],
        ]
        data = [
            [u_exact, u_pred, u_err],
            [v_exact, v_pred, v_err],
            [p_exact, p_pred, p_err],
        ]

        # --------- 整体大图展示 ----------
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        for i in range(3):
            for j in range(3):
                ax = axes[i][j]
                cf = ax.contourf(xx, yy, data[i][j], cmap="viridis")
                ax.set_title(titles[i][j])
                plt.colorbar(cf, ax=ax)

        plt.tight_layout()
        plt.show()

        # --------- 单独展示每一个小图 ----------
        if individually_show:
            for i in range(3):
                for j in range(3):
                    fig_single, ax_single = plt.subplots(figsize=(5, 4))
                    cf_single = ax_single.contourf(xx, yy, data[i][j], cmap="viridis")
                    ax_single.set_title(titles[i][j])
                    plt.colorbar(cf_single, ax=ax_single)
                    plt.tight_layout()
                    plt.show()


    def data(self):
        """返回内部点和边界点"""
        return {
            'interior_points': self.interior_points,
            'boundary_points': self.boundary_points
        }

    def loss_int(self, uvp, data):
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]

        # 一阶导数
        grads = torch.autograd.grad(u, data, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]

        grads = torch.autograd.grad(v, data, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = grads[:, 0:1]
        v_y = grads[:, 1:2]

        grads = torch.autograd.grad(p, data, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_x = grads[:, 0:1]
        p_y = grads[:, 1:2]

        # 二阶导数
        u_xx = torch.autograd.grad(u_x, data, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, data, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x, data, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, data, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

        # Navier–Stokes 残差项
        momentum_u = u * u_x + v * u_y + p_x - 1/40 * (u_xx + u_yy)
        momentum_v = u * v_x + v * v_y + p_y - 1/40 * (v_xx + v_yy)
        continuity = u_x + v_y

        # Loss（每个残差平方平均）
        loss_int = torch.mean(momentum_u**2 + momentum_v**2 + continuity**2)
        self.loss_int_value = loss_int
        return loss_int
    
    def loss_bdry(self, uvp, data): 
        loss_bdry = torch.mean((uvp - self.exact_solution(data)) ** 2)
        self.loss_bdry_value = loss_bdry
        return loss_bdry
    
    def loss_pde(self):
        return self.loss_int_value + self.bdry_penalty * self.loss_bdry_value
    
    def picardleastsquareproblem(self, model, body, bdry, max_iter=20):
        device = body.device
        M = body.shape[0]
        B = bdry.shape[0]
        
        # Basis evaluation
        basis = lambda x: model(x)[0]
        phi= basis(body)        # shape: [M, N]
        phi_bdry = basis(bdry)
        N = phi.shape[1]

        grad_phi = torch.zeros(M, N, 2, dtype=torch.float64, device=device)
        lap_phi = torch.zeros(M, N, dtype=torch.float64, device=device)
        for i in range(N):
            dphi = model.grad(body, i)
            # dphi = torch.autograd.grad(phi[:, i], body,
            #                         grad_outputs=torch.ones_like(phi[:, i]),
            #                         create_graph=True)[0]
            d2phi_x = torch.autograd.grad(dphi[:, 0], body,
                                        grad_outputs=torch.ones_like(dphi[:, 0]),
                                        retain_graph=True)[0][:, 0]
            d2phi_y = torch.autograd.grad(dphi[:, 1], body,
                                        grad_outputs=torch.ones_like(dphi[:, 1]),
                                        retain_graph=True)[0][:, 1]
            
            grad_phi[:, i, 0] = dphi[:, 0]
            grad_phi[:, i, 1] = dphi[:, 1]
            lap_phi[:, i] = d2phi_x + d2phi_y

        # Boundary target values
        u_bdry_target = self.boundary_condition(bdry).to(torch.float64)

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
        
        a = torch.zeros(N, dtype=torch.float64, device=device)
        b = torch.zeros(N, dtype=torch.float64, device=device)
        c = torch.zeros(N, dtype=torch.float64, device=device)
        for it in range(max_iter):
            u_k = phi @ a  # shape: [M]
            v_k = phi @ b

            # NS residuals for u equation
            A_u = u_k.unsqueeze(1) * grad_phi[:, :, 0] + v_k.unsqueeze(1) * grad_phi[:, :, 1] - self.nu * lap_phi

            # NS residuals for v equation
            A_v = u_k.unsqueeze(1) * grad_phi[:, :, 0] + v_k.unsqueeze(1) * grad_phi[:, :, 1] - self.nu * lap_phi

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

            A_np = A.cpu().detach().numpy()
            rhs_np = rhs.cpu().detach().numpy()

            w, residuals, rank, s = lstsq(A_np, rhs_np)
            a, b, c = w[:N], w[N:2 * N], w[2 * N:]
            a = torch.tensor(a, dtype=torch.float64, device=device)
            b = torch.tensor(b, dtype=torch.float64, device=device)
            c = torch.tensor(c, dtype=torch.float64, device=device)
            residual = (A_np @ w - rhs_np)
            print(f"[Picard {it}] LS MSE: {np.mean(residual**2):.2e}")
        
        return a, b, c, residual

class PoissonBox3D(PoissonBase):
    def __init__(self, interior_condition, boundary_condition, exact_solution, domain_size=1.0, bdry_penalty=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(interior_condition, boundary_condition, exact_solution, domain_size, bdry_penalty, device)
        self.name = "poisson3d"

    def generate_points(self, num_int, num_boundary, point_distribution='uniform'):
        if point_distribution == 'uniform':
            n = int(round(num_int ** (1 / 3)))
            x = torch.linspace(0, 1, n) * self.domain_size
            y = torch.linspace(0, 1, n) * self.domain_size
            z = torch.linspace(0, 1, n) * self.domain_size
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            self.interior_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1).to(self.device)
        elif point_distribution == 'random':
            self.interior_points = (torch.rand(num_int, 3) * self.domain_size).to(self.device)
        else:
            raise ValueError("Invalid point distribution method. Choose 'uniform' or 'random'.")

        n = int(round((num_boundary / 6) ** (1 / 2)))
        x = torch.linspace(0, 1, n).to(self.device) * self.domain_size
        y = torch.linspace(0, 1, n).to(self.device) * self.domain_size
        z = torch.linspace(0, 1, n).to(self.device) * self.domain_size
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

        self.boundary_points = torch.cat([
            torch.stack([xx[0], yy[0], zz[0]], dim=-1),    # x = 0
            torch.stack([xx[-1], yy[-1], zz[-1]], dim=-1),  # x = 1
            torch.stack([xx[:, 0], yy[:, 0], zz[:, 0]], dim=-1),  # y = 0
            torch.stack([xx[:, -1], yy[:, -1], zz[:, -1]], dim=-1),  # y = 1
            torch.stack([xx[:, :, 0], yy[:, :, 0], zz[:, :, 0]], dim=-1),  # z = 0
            torch.stack([xx[:, :, -1], yy[:, :, -1], zz[:, :, -1]], dim=-1),  # z = 1
        ]).reshape(-1, 3).to(self.device)

    def plot(self, function):
        n_plot = 20
        x = torch.linspace(0, 1, n_plot).to(self.device)
        y = torch.linspace(0, 1, n_plot).to(self.device)
        z = torch.linspace(0, 1, n_plot).to(self.device)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1).to(self.device)

        exact = self.exact_solution(grid_points).reshape(n_plot, n_plot, n_plot)
        pred = function(grid_points).reshape(n_plot, n_plot, n_plot)
        error = torch.abs(pred - exact)
        Error = error.cpu().detach().numpy()
        Exact = exact.cpu().detach().numpy()
        l2_error = np.sqrt(np.nanmean(Error**2))
        l2_error_rel = l2_error / np.sqrt(np.nanmean(Exact**2))
        print("the l2 relative error in problem " + str(self.name)+": ", l2_error_rel)
        print("the maximum pointwise error in problem " + str(self.name)+": ", np.nanmax(np.abs(Error)))

        # 对 z 维度做平均，只展示 x-y 平面
        exact_avg = exact.mean(dim=2).cpu().detach().numpy()
        pred_avg = pred.mean(dim=2).cpu().detach().numpy()
        error_avg = error.mean(dim=2).cpu().detach().numpy()
        xx2d = xx[:, :, 0].cpu().detach().numpy()
        yy2d = yy[:, :, 0].cpu().detach().numpy()

        for data, title, label in zip([error_avg, exact_avg, pred_avg],
                                      ["Absolute Error", "Exact Solution", "Model Prediction"],
                                      ["Error", "Exact", "Prediction"]):
            plt.figure(figsize=(6, 5))
            cf1 = plt.contourf(xx2d, yy2d, data, cmap="viridis", levels=100)
            plt.title(title)
            cbar = plt.colorbar(cf1, label=label)
            cbar.formatter = ticker.ScalarFormatter(useMathText=True)
            cbar.formatter.set_powerlimits((-2, 2))  # 设置何时启用科学计数法
            cbar.update_ticks()

            # 只移动 colorbar 的 “1e−10”
            offset_text = cbar.ax.yaxis.get_offset_text()
            offset_text.set_y(2.5)  # 数值越大越往上移

            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()



class Wave1D:
    def __init__(self, initial_displacement_condition, initial_speed_condition , exact_solution, x_range=[0, 1], t_range=[0, 2], wave_speed=1.0, bdry_penalty=1.0, init_penalty=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.initial_displacement_condition = initial_displacement_condition
        self.initial_speed_condition = initial_speed_condition
        self.exact_solution = exact_solution
        self.x_range = x_range
        self.t_range = t_range
        self.wave_speed = wave_speed
        self.name = "wave1d"
        self.bdry_penalty = bdry_penalty
        self.init_penalty = init_penalty
        
    def generate_points(self, num_int, num_boundary, point_distribution='uniform'):
        """Generate points in space-time domain"""
        # Generate interior points in space-time domain
        if point_distribution == 'uniform':
            nx = int(np.sqrt(num_int))
            nt = nx
            x = torch.linspace(self.x_range[0], self.x_range[1], nx)
            t = torch.linspace(self.t_range[0], self.t_range[1], nt)
            xx, tt = torch.meshgrid(x, t, indexing='ij')
            self.interior_points = torch.stack([xx.flatten(), tt.flatten()], dim=-1).to(self.device)
        elif point_distribution == 'random':
            x = torch.rand(num_int) * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
            t = torch.rand(num_int) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
            self.interior_points = torch.stack([x, t], dim=-1).to(self.device)

        # Generate boundary points (spatial boundaries and initial conditions)
        nb = num_boundary // 4  # Split points between boundaries and initial condition
        t_boundary = torch.linspace(self.t_range[0], self.t_range[1], nb)
        
        # Spatial boundaries (x = a and x = b for all t)
        left_boundary = torch.stack([torch.full_like(t_boundary, self.x_range[0]), t_boundary], dim=-1)
        right_boundary = torch.stack([torch.full_like(t_boundary, self.x_range[1]), t_boundary], dim=-1)
        
        # Initial condition (t = 0 for all x)
        x_initial = torch.linspace(self.x_range[0], self.x_range[1], nb)
        x_final = torch.linspace(self.x_range[0], self.x_range[1], nb)
        initial_points = torch.stack([x_initial, torch.zeros_like(x_initial)], dim=-1)
        final_points = torch.stack([x_final, torch.full_like(x_final, self.t_range[1])], dim=-1)
        
        self.boundary_points = torch.cat([left_boundary, right_boundary], dim=0).to(self.device)
        self.initial_points = initial_points.to(self.device)
        # self.initial_points = torch.cat([initial_points, final_points], dim=0).to(self.device)
        
    def plot(self, function):
        """Plot the solution in space-time domain as 2D color plots"""
        nx, nt = 100, 100
        x = torch.linspace(self.x_range[0], self.x_range[1], nx).to(self.device)
        t = torch.linspace(self.t_range[0], self.t_range[1], nt).to(self.device)
        xx, tt = torch.meshgrid(x, t, indexing='ij')
        grid_points = torch.stack([xx.flatten(), tt.flatten()], dim=-1).to(self.device)

        # Compute solutions
        exact_u = self.exact_solution(grid_points).reshape(nx, nt)
        pred_u = function(grid_points).reshape(nx, nt)
        error = torch.abs(exact_u - pred_u)

        # Convert to numpy for plotting
        xx = xx.cpu().detach().numpy()
        tt = tt.cpu().detach().numpy()
        exact_u = exact_u.cpu().detach().numpy()
        pred_u = pred_u.cpu().detach().numpy()
        error = error.cpu().detach().numpy()
        
        error_l2 = np.sqrt(np.nanmean(error**2))
        error_l2_rel =error_l2/np.sqrt(np.nanmean(exact_u**2))
        print("the maximum pointwise error in problem " + str(self.name)+": ", np.max(error))
        print("the l2 relative error in problem " + str(self.name)+": ", error_l2_rel)
        
        # Create 2D color plots
        def create_2d_plot(data, title, cmap="viridis"):
            plt.figure(figsize=(6, 5))
            cf1 = plt.contourf(xx, tt, data, cmap=cmap, levels=100)
            plt.title(title)
            cbar = plt.colorbar(cf1)
            cbar.formatter = ticker.ScalarFormatter(useMathText=True)
            cbar.formatter.set_powerlimits((-2, 2))  # 设置何时启用科学计数法
            cbar.update_ticks()

            # 只移动 colorbar 的 “1e−10”
            offset_text = cbar.ax.yaxis.get_offset_text()
            offset_text.set_x(1.0)   # 数值越大越往右移
            offset_text.set_y(2.5)  # 数值越大越往上移

            plt.xlabel("x")
            plt.ylabel("t")
            plt.show()

        create_2d_plot(error, "Absolute Error")
        create_2d_plot(exact_u, "Exact Solution")
        create_2d_plot(pred_u, "Model Prediction")

    def data(self):
        """Return the generated interior and boundary points"""
        return {
            'interior_points': self.interior_points,
            'boundary_points': self.boundary_points,
            'initial_points': self.initial_points
        }

    def loss_int(self, u, data):
        
        """Compute the PDE residual for the wave equation"""
        # Compute first derivatives
        grads = torch.autograd.grad(u, data, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]

        # Compute second derivatives
        u_xx = torch.autograd.grad(u_x, data, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_tt = torch.autograd.grad(u_t, data, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 1:2]

        # Wave equation residual: u_tt = c^2 * u_xx
        residual = u_tt - (self.wave_speed ** 2) * u_xx 
        
        # Interior loss (periodic loss)
        loss = torch.mean(residual ** 2)
        self.loss_int_value = loss
        return loss
    
    def loss_bdry(self, u, data):
        """Compute the periodic boundary loss for the wave equation"""
        # Get boundary points where x = 0 and x = 1
        x = data[:, 0]
        mask_left = (x == 0)
        mask_right = (x == 1)
        
        # Get u values at x = 0 and x = 1
        u_left = u[mask_left]
        u_right = u[mask_right]
        
        # Periodic boundary condition: u(0,t) = u(1,t)
        loss = torch.mean((u_left - u_right) ** 2)
        loss = torch.mean((u - self.exact_solution(data)) ** 2)
        self.loss_bdry_value = loss
        return loss
    
    def loss_init(self, u, data):
        """Compute the initial condition loss for the wave equation"""
        loss_disp = torch.mean((u - self.exact_solution(data)) ** 2)
        grads = torch.autograd.grad(u, data, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = grads[:, 1:2]
        loss_speed = torch.mean((u_t - self.initial_speed_condition(data)) ** 2)
        self.loss_init_value = loss_disp + loss_speed
        return loss_disp + loss_speed
    
    def loss_pde(self):
        return self.loss_int_value + self.bdry_penalty * self.loss_bdry_value + self.init_penalty * self.loss_init_value
    
    def leastsquareproblem(self, basis, body, bdry, init):
        u_body = basis(body)
        Nb, Mb = u_body.shape[0], u_body.shape[1]
        A_body = torch.zeros(Nb, Mb, device=self.device)
        for i in range(Mb):
            grads = torch.autograd.grad(u_body[:,i], body, grad_outputs=torch.ones_like(u_body[:,i]), create_graph=True)[0]
            u_x = grads[:, 0:1]
            u_t = grads[:, 1:2]
            u_xx = torch.autograd.grad(u_x, body, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
            u_tt = torch.autograd.grad(u_t, body, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 1:2]
            A_body[:,i] = (u_tt - (self.wave_speed ** 2) * u_xx).detach().squeeze(1) 
        f_body = torch.zeros(Nb, device=self.device)
        
        ### initial condition
        A_initial_disp = basis(init).detach() 
        f_initial_disp = self.exact_solution(init).detach()
        if f_initial_disp is not None:
            if f_initial_disp.dim() > 1 and f_initial_disp.shape[1] == 1:
                f_initial_disp = f_initial_disp.squeeze(1)
        
        u_init = basis(init)
        Ni, Mi = u_init.shape[0], u_init.shape[1]
        A_initial_speed = torch.zeros(Ni, Mi, device=self.device)
        for i in range(Mi):
            grads = torch.autograd.grad(u_init[:,i], init, grad_outputs=torch.ones_like(u_init[:,i]), create_graph=True)[0]
            u_t = grads[:, 1:2]
            A_initial_speed[:,i] = u_t.detach().squeeze(1)
        f_initial_speed = self.initial_speed_condition(init).detach()
        if f_initial_speed is not None:
            if f_initial_speed.dim() > 1 and f_initial_speed.shape[1] == 1:
                f_initial_speed = f_initial_speed.squeeze(1)
        
        ## periodic boundary condition
        x = bdry[:, 0]
        t = bdry[:, 1]
        mask_left = (x == self.x_range[0])
        mask_right = (x == self.x_range[1])
        t_left = t[mask_left]
        t_right = t[mask_right]
        assert torch.allclose(t_left, t_right), "Boundary times do not match!"
        u_left = basis(bdry[mask_left])
        u_right = basis(bdry[mask_right])
        A_bdry = (u_left - u_right).detach()  # shape [N_bdry_per_side, M]
        f_bdry = torch.zeros(A_bdry.shape[0], device=self.device)
        
        # ## explicit boundary condition
        # A_bdry = basis(bdry).detach()
        # f_bdry = self.exact_solution(bdry).detach()
        # if f_bdry is not None:
        #     if f_bdry.dim() > 1 and f_bdry.shape[1] == 1:
        #         f_bdry = f_bdry.squeeze(1)
        
        A = torch.cat((A_body, A_initial_disp, A_initial_speed, A_bdry), dim=0)
        rhs = torch.cat((f_body, f_initial_disp, f_initial_speed, f_bdry), dim=0)
        A_np = A.cpu().detach().numpy()
        rhs_np = rhs.cpu().detach().numpy()
        
        # rescaling
        c = 100.0
        for i in range(len(A_np)):
            ratio = c/(abs(A_np[i,:]).max() + 1)
            A_np[i,:] = A_np[i,:]*ratio
            rhs_np[i] = rhs_np[i]*ratio
            
        return (A_np, rhs_np)
        
