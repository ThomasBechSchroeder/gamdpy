# Tutorials for gamdpy

If you do not have access to a nvidia GPU, you can run the tutorials in [Google Colab](https://colab.research.google.com/), which provides GPU's free of charge: 
1) push the 'Open in Colab' button.
2) When the notebook is open in Colab, choose a runtime type with GPU before running ( under "Runtime" / "Change runtime type").
3) Insert and run a notebook cell with the following code:

```sh
!pip -q install gamdpy
from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1
```

* my_first_simulation.ipynb [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/ThomasBechSchroeder/gamdpy/blob/master/tutorials/my_first_simulation.ipynb)
* post_analysis.ipynb [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/ThomasBechSchroeder/gamdpy/blob/master/tutorials/post_analysis.ipynb)
 
