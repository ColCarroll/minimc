from autograd import grad


class AutogradPotential:
    def __init__(self, potential_func):
        self.potential = potential_func
        self.grad_potential = grad(potential_func)

    def __call__(self, coords):
        return self.potential(coords), self.grad_potential(coords)
