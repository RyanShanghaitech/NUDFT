# NUDFT

The `dft()` and `idft()` functions performs the following formula:

$$
S(\vec{k}) = \sum_{\vec{x}}{I(\vec{x})e^{-2 \pi \vec{k} \vec{x}}}
$$

and

$$
I(\vec{x}) = \sum_{\vec{k}}{S(\vec{k})e^{2 \pi \vec{k} \vec{x}}}
$$