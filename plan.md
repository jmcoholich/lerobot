I would like to create a modified version of the pi05 VLA which can be used to predict two scalar values instead.

Please create a copy of pi05_training.bash called pi05_iql.bash and modify it. Specifically, add a flag that, when passed, loads a different version of pi05 instead of PI05Pytorch(). Please create this new version, which will be mostly the same but will not include an action expert and instead project the final output to a scalar value via an MLP. Assume the training batch will come with a "value" (named that) and apply MSE loss in the forward function.
