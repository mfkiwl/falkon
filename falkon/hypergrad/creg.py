import torch

from falkon import FalkonOptions
from falkon.hypergrad.common import full_rbf_kernel, get_scalar, cholesky
from falkon.hypergrad.complexity_reg import NystromKRRModelMixinN, HyperOptimModel
from falkon.hypergrad.leverage_scores import (
    creg_penfit, RegLossAndDeffv2, creg_plainfit,
    NoRegLossAndDeff
)
from falkon.kernels import GaussianKernel


class CompDeffPenFitTr(HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            flk_opt: FalkonOptions,
            num_trace_est: int = 20,
            flk_maxiter: int = 10,
            nystrace_ste: bool = False,
    ):
        self.stoch_model = StochasticDeffPenFitTr(sigma_init.clone(), penalty_init, centers_init.clone(), opt_centers, opt_sigma, opt_penalty, cuda, flk_opt, num_trace_est, flk_maxiter, nystrace_ste)
        self.det_model = DeffPenFitTr(sigma_init.clone(), penalty_init, centers_init.clone(), opt_centers, opt_sigma, opt_penalty, cuda)
        self.use_model = "stoch"

    def parameters(self):
        if self.use_model == "stoch":
            return self.stoch_model.parameters()
        else:
            return self.det_model.parameters()

    def named_parameters(self):
        if self.use_model == "stoch":
            return self.stoch_model.named_parameters()
        else:
            return self.det_model.named_parameters()

    def cuda(self):
        self.stoch_model.cuda()
        self.det_model.cuda()

    def train(self):
        self.stoch_model.train()
        self.det_model.train()

    def eval(self):
        self.stoch_model.eval()
        self.det_model.eval()

    def hp_loss(self, X, Y):
        if self.use_model == "stoch":
            self.det_model.centers = self.stoch_model.centers
            self.det_model.penalty = self.stoch_model.penalty
            self.det_model.sigma = self.stoch_model.sigma
        else:
            self.stoch_model.centers = self.det_model.centers
            self.stoch_model.penalty = self.det_model.penalty
            self.stoch_model.sigma = self.det_model.sigma

        ndeff, datafit, trace = self.det_model.hp_loss(X, Y)
        stoch_loss = self.stoch_model.hp_loss(X, Y)
        print(f"Deterministic: D-eff {ndeff:.2e} Data-Fit {datafit:.2e} Trace {trace:.2e}")

        if self.use_model == "stoch":
            return stoch_loss
        else:
            return [ndeff + datafit + trace]

    def predict(self, X):
        return self.det_model.predict(X)
        return self.stoch_model.predict(X)

    @property
    def centers(self):
        if self.use_model == "stoch":
            return self.stoch_model.centers
        return self.det_model.centers

    @property
    def sigma(self):
        if self.use_model == "stoch":
            return self.stoch_model.sigma
        return self.det_model.sigma

    @property
    def penalty(self):
        if self.use_model == "stoch":
            return self.stoch_model.penalty
        return self.det_model.penalty
    @property
    def loss_names(self):
        if self.use_model == "stoch":
            return ["stoch-creg-penfit"]
        return ["det-creg-penfit"]

    def __repr__(self):
        return f"CompDeffPenFitTr()"


class StochasticDeffPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            flk_opt: FalkonOptions,
            num_trace_est: int = 20,
            flk_maxiter: int = 10,
            nystrace_ste: bool = False,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.flk_opt = flk_opt
        self.num_trace_est = num_trace_est
        self.flk_maxiter = flk_maxiter
        self.nystrace_ste = nystrace_ste

    def hp_loss(self, X, Y):
        loss = creg_penfit(kernel_args=self.sigma, penalty=self.penalty, centers=self.centers,
                           X=X, Y=Y, num_estimators=self.num_trace_est, deterministic=False,
                           solve_options=self.flk_opt, solve_maxiter=self.flk_maxiter,
                           gaussian_random=False, use_stoch_trace=self.nystrace_ste)
        return [loss]

    def predict(self, X):
        if RegLossAndDeffv2.last_alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        kernel = GaussianKernel(sigma=self.sigma.detach(), opt=self.flk_opt)
        with torch.autograd.no_grad():
            return kernel.mmv(X, self.centers, RegLossAndDeffv2.last_alpha)

    @property
    def loss_names(self):
        return ["stoch-creg-penfit"]

    def __repr__(self):
        return f"StochasticDeffPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, t={self.num_trace_est}, " \
               f"flk_iter={self.flk_maxiter})"


class DeffPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.L, self.LB, self.c = None, None, None

    def hp_loss(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = (full_rbf_kernel(self.centers, self.centers, self.sigma) +
               torch.eye(m, device=X.device, dtype=X.dtype) * 1e-6)
        self.L = cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution / sqrt_var
        AAT = A @ A.T  # m*n @ n*m = m*m in O(n * m^2), equivalent to kmn @ knm.
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        self.LB = cholesky(B)  # LB @ LB.T = B
        AY = A @ Y  # m*1
        self.c = torch.triangular_solve(AY, self.LB, upper=False).solution / sqrt_var  # m*1

        C = torch.triangular_solve(A, self.LB, upper=False).solution  # m*n

        # Complexity (nystrom-deff)
        ndeff = C.square().sum()  # = torch.trace(C.T @ C)
        datafit = torch.square(Y).sum() - torch.square(self.c * sqrt_var).sum()
        trace = Kdiag - torch.trace(AAT) * variance

        return ndeff, datafit, trace

    def predict(self, X):
        if self.L is None or self.LB is None or self.c is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        with torch.autograd.no_grad():
            tmp1 = torch.triangular_solve(self.c, self.LB, upper=False, transpose=True).solution
            tmp2 = torch.triangular_solve(tmp1, self.L, upper=False, transpose=True).solution
            kms = full_rbf_kernel(self.centers, X, self.sigma)
            return kms.T @ tmp2

    @property
    def loss_names(self):
        return "nys-deff", "data-fit", "trace"

    def __repr__(self):
        return f"DeffPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"


class CompDeffNoPenFitTr(HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            flk_opt: FalkonOptions,
            num_trace_est: int = 20,
            flk_maxiter: int = 10,
            nystrace_ste: bool = False,
    ):
        self.stoch_model = StochasticDeffNoPenFitTr(sigma_init.clone(), penalty_init, centers_init.clone(), opt_centers, opt_sigma, opt_penalty, cuda, flk_opt, num_trace_est, flk_maxiter, nystrace_ste)
        self.det_model = DeffNoPenFitTr(sigma_init.clone(), penalty_init, centers_init.clone(), opt_centers, opt_sigma, opt_penalty, cuda)
        self.use_model = "stoch"

    def parameters(self):
        if self.use_model == "stoch":
            return self.stoch_model.parameters()
        else:
            return self.det_model.parameters()

    def named_parameters(self):
        if self.use_model == "stoch":
            return self.stoch_model.named_parameters()
        else:
            return self.det_model.named_parameters()

    def cuda(self):
        self.stoch_model.cuda()
        self.det_model.cuda()

    def train(self):
        self.stoch_model.train()
        self.det_model.train()

    def eval(self):
        self.stoch_model.eval()
        self.det_model.eval()

    def hp_loss(self, X, Y):
        if self.use_model == "stoch":
            self.det_model.centers = self.stoch_model.centers
            self.det_model.penalty = self.stoch_model.penalty
            self.det_model.sigma = self.stoch_model.sigma
        else:
            self.stoch_model.centers = self.det_model.centers
            self.stoch_model.penalty = self.det_model.penalty
            self.stoch_model.sigma = self.det_model.sigma

        ndeff, datafit, trace = self.det_model.hp_loss(X, Y)
        stoch_loss = self.stoch_model.hp_loss(X, Y)
        print(f"Deterministic: D-eff {ndeff:.2e} Data-Fit {datafit:.2e} Trace {trace:.2e}")

        if self.use_model == "stoch":
            return stoch_loss
        else:
            return [ndeff + datafit + trace]

    def predict(self, X):
        return self.det_model.predict(X)
        return self.stoch_model.predict(X)

    @property
    def centers(self):
        if self.use_model == "stoch":
            return self.stoch_model.centers
        return self.det_model.centers

    @property
    def sigma(self):
        if self.use_model == "stoch":
            return self.stoch_model.sigma
        return self.det_model.sigma

    @property
    def penalty(self):
        if self.use_model == "stoch":
            return self.stoch_model.penalty
        return self.det_model.penalty
    @property
    def loss_names(self):
        if self.use_model == "stoch":
            return ["stoch-creg-plainfit"]
        return ["det-creg-plainfit"]

    def __repr__(self):
        return f"CompDeffNoPenFitTr()"


class StochasticDeffNoPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            flk_opt: FalkonOptions,
            num_trace_est: int = 20,
            flk_maxiter: int = 10,
            nystrace_ste: bool = False,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.flk_opt = flk_opt
        self.num_trace_est = num_trace_est
        self.flk_maxiter = flk_maxiter
        self.nystrace_ste = nystrace_ste

    def hp_loss(self, X, Y):
        loss = creg_plainfit(kernel_args=self.sigma, penalty=self.penalty, centers=self.centers,
                             X=X, Y=Y, num_estimators=self.num_trace_est, deterministic=False,
                             solve_options=self.flk_opt, solve_maxiter=self.flk_maxiter,
                             gaussian_random=False, use_stoch_trace=self.nystrace_ste, warm_start=True)
        return [loss]

    def predict(self, X):
        if NoRegLossAndDeff.last_alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        kernel = GaussianKernel(self.sigma.detach(), opt=self.flk_opt)
        with torch.autograd.no_grad():
            return kernel.mmv(X, self.centers, NoRegLossAndDeff.last_alpha)

    @property
    def loss_names(self):
        return ["stoch-creg-plainfit"]

    def __repr__(self):
        return f"StochasticDeffNoPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, t={self.num_trace_est}, " \
               f"flk_iter={self.flk_maxiter})"


class DeffNoPenFitTr(NystromKRRModelMixinN, HyperOptimModel):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.alpha = None

    def hp_loss(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = (full_rbf_kernel(self.centers, self.centers, self.sigma) +
               torch.eye(m, device=X.device, dtype=X.dtype) * 1e-6)
        L = cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        LB = cholesky(B)  # LB @ LB.T = B
        AY = A @ Y
        c = torch.triangular_solve(AY, LB, upper=False).solution / sqrt_var

        tmp1 = torch.triangular_solve(c, LB, upper=False, transpose=True).solution
        self.alpha = torch.triangular_solve(tmp1, L, upper=False, transpose=True).solution
        d = A.T @ tmp1

        C = torch.triangular_solve(A, LB, upper=False).solution

        # Complexity (nystrom-deff)
        ndeff = C.square().sum()  # = torch.trace(C.T @ C)
        datafit = torch.square(Y).sum() - 2 * torch.square(
            c * sqrt_var).sum() + variance * torch.square(d).sum()
        trace = Kdiag - torch.trace(AAT) * variance
        # trace = trace / variance  # TODO: This is a temporary addition!

        return ndeff, datafit, trace

    def predict(self, X):
        if self.alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        with torch.autograd.no_grad():
            kms = full_rbf_kernel(self.centers, X, self.sigma)
            return kms.T @ self.alpha

    @property
    def loss_names(self):
        return "nys-deff", "data-fit", "trace"

    def __repr__(self):
        return f"DeffNoPenFitTr(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, " \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"