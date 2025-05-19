from .losses import dgswp, H_eps, H_module, get_dgswp_plan
from .gradient_flows import (DifferentiableGeneralizedWassersteinPlanGradientFlow,
                             PoincareDGSWPGradientFlow,
                             SlicedWassersteinGradientFlow,
                             MaxSlicedWassersteinGradientFlow,
                             RandomSearchSWGGGradientFlow,
                             SWGGGradientFlow,
                             AugmentedSlicedWassersteinGradientFlow)
from .datasets import data_gen, data_gen_torch
