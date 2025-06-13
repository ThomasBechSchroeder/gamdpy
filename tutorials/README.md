# Tutorials for gamdpy

If you do not have access to a nvidia GPU, you can run the tutorials in [Google Colab](https://colab.research.google.com/), which provides GPU's free of charge: 
1) push the 'Open in Colab' button.
2) When the notebook is open in Colab, choose a runtime type with GPU before running ( under "Runtime" / "Change runtime type").
3) Insert and run a notebook cell with the following code:

```sh
!pip -q install git+https://github.com/ThomasBechSchroeder/gamdpy.git
```

* [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/ThomasBechSchroeder/gamdpy/blob/master/tutorials/my_first_simulation.ipynb): my_first_simulation.ipynb.

By default files on google colab are not persistent between sessions, so if you want to keep it, you should download the h5 file containing the results of the simulation (LJ_T0.70.h5).

* [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://colab.research.google.com/github/ThomasBechSchroeder/gamdpy/blob/master/tutorials/post_analysis.ipynb): post_analysis.ipynb. 

This notebook analyses the results from a gamdpy simulation stored in a h5 file, so you should upload the file LJ_T0.70.h5 produced by the 'my_first_simulation' tutorial above.
 
