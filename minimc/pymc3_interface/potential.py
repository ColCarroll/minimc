import theano

import pymc3 as pm
from pymc3.model import Point
from pymc3.theanof import inputvars
from pymc3.blocking import ArrayOrdering, DictToArrayBijection


def get_args_for_theano_function(point=None, model=None):
    model = pm.modelcontext(model)
    if point is None:
        point = model.test_point
    return [point[k.name] for k in model.vars]


def get_theano_function_for_var(var, model=None, **kwargs):
    model = pm.modelcontext(model)
    kwargs["on_unused_input"] = kwargs.get("on_unused_input", "ignore")
    return theano.function(model.vars, var, **kwargs)


def allinmodel(vars, model):
    notin = [v for v in vars if v not in model.vars]
    if notin:
        raise ValueError("Some variables not in the model: " + str(notin))


class PyMC3Potential:
    def __init__(self, vars=None, model=None, point=None):
        self.model = pm.modelcontext(model)

        # Work out the full starting coordinates
        if point is None:
            point = self.model.test_point
        else:
            pm.util.update_start_vals(point, self.model.test_point, self.model)

        # Fit all the parameters by default
        if vars is None:
            vars = self.model.cont_vars
        self.vars = inputvars(vars)
        allinmodel(self.vars, self.model)

        # Work out the relevant bijection map
        point = Point(point, model=self.model)
        self.bijection = DictToArrayBijection(ArrayOrdering(self.vars), point)

        # Pre-compile the theano model and gradient
        nlp = -self.model.logpt
        grad = theano.grad(nlp, self.vars, disconnected_inputs="ignore")
        self.func = get_theano_function_for_var([nlp] + grad, model=self.model)

    def __call__(self, coords):
        res = self.func(
            *get_args_for_theano_function(
                self.bijection.rmap(coords), model=self.model
            )
        )
        d = dict(zip((v.name for v in self.vars), res[1:]))
        g = self.bijection.map(d)
        return res[0], g
