# Resources and references

I’ve put together some resources and references using Python. These are the main libraries that supported my analyses, computations, and visualizations in the tasks/projects done during the internship.

    
    
**Core numerical and scientific computing:**

[NumPy](https://www.w3schools.com/python/numpy/numpy_intro.asp): Fundamental library for numerical computing, arrays, and linear algebra.

[SciPy](https://www.w3schools.com/python/scipy/scipy_intro.php): Scientific computing, including signal processing (find_peaks, hilbert) and interpolation (interp1d).

[os](https://docs.python.org/3.10/library/os.html): Provides a way of interacting with the operating system.

[math](https://docs.python.org/3/library/math.html): Provides basic mathematical functions and constants such as square root, trigonometric functions, and π.

**Data handling and manipulation:**

[pandas](https://pypi.org/project/pandas/): Data structures and analysis tools (DataFrame, Series).

[tslearn](https://pypi.org/project/tslearn/): Time-series machine learning, including preprocessing (TimeSeriesScalerMinMax) and shapelets learning (LearningShapelets).

**Statistics and metrics:**

[scipy.stats](https://docs.scipy.org/doc/scipy/tutorial/stats.html): Provides statistical functions such as skew and kurtosis.

[numpy.linalg](https://numpy.org/doc/2.1/reference/routines.linalg.html): Linear algebra operations, e.g. least squares (lstsq).

[scikit-learn](https://pypi.org/project/scikit-learn/) (sklearn): Tools for machine learning and model evaluation:

  TruncatedSVD for dimensionality reduction.

  StandardScaler for feature scaling.

  r2_score for model evaluation.

  metrics for additional evaluation metrics for regression and classification performance.
  
**Signal processing and electrophysiology**

[pyABF](https://swharden.com/pyabf/): Specialized library to read and analyze Axon Binary Files (ABF) from electrophysiology experiments.
[scipy.signal](https://www.askpython.com/python-modules/scipy-signal): Tools for signal processing, including filtering, Fourier transforms, and feature extraction.

**Machine learning and dimensionality reduction**

[UMAP](https://umap-learn.readthedocs.io/en/latest/): Dimensionality reduction and visualization technique.

[TensorFlow/Keras](https://pypi.org/project/tf-keras/): Deep learning framework (used here with the Adam optimizer).

[sklearn.cluster](https://scikit-learn.org/stable/api/sklearn.cluster.html): Algorithms for unsupervised clustering (e.g., K-Means, DBSCAN).

[sklearn.ensemble](https://scikit-learn.org/stable/api/sklearn.ensemble.html): Ensemble learning methods such as Random Forests and Gradient Boosting.

[sklearn.neighbors](https://scikit-learn.org/stable/modules/neighbors.html): Nearest-neighbor algorithms for classification, regression, and density estimation.

**Graphs and network analysis**

[networkx](https://networkx.org/documentation/stable/tutorial.html): Creation, analysis, and visualization of complex networks/graphs.

**Plotting and visualization**

[matplotlib](https://pypi.org/project/matplotlib/): Standard Python plotting library.

[matplotlib.cm](https://matplotlib.org/stable/api/cm_api.html): Colormap handling for visualizations.

[seaborn](https://pypi.org/project/seaborn/): High-level statistical data visualization built on matplotlib.

[plotly.express](https://plotly.com/python/plotly-express/): Interactive plotting library.

[plotly.io](https://plotly.com/python-api-reference/generated/plotly.io.html): Lower-level interface for rendering plotly figures.

**Jupyter utilities**

[%matplotlib widget](https://matplotlib.org/stable/api/widgets_api.html): Enables interactive plots within Jupyter notebooks.
