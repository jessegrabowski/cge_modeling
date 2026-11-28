from typing import Literal

CompiledFunctions = Literal["root", "minimize", "path", "all"]
StepMethod = Literal["euler", "rk4"]
DEFAULT_STEP_METHOD: StepMethod = "rk4"
