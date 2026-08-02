# Internship_work2_DeepONet_PINN
This is a resgister of all the work that I have done during my internship in BCAM applying DeepONet and PINN

**Repository structure:**  
The repository is designed as a log of my progress throughout the internship. Rather than providing introductory explanations of basic concepts, it focuses directly on the specific topics, analyses, and results that I worked on. This makes it a useful reference both for myself and for anyone interested in following the steps of the developed tasks.

- **[User manual](./User_manual.md)** User manual, it contains basic informatiom about the libraries used, etc.
  
- **[Tasks/projects done](./task)**: Main documentation and notebooks describing the work carried out during the internship.

  - **[Context](./task/Context.md)**: Initial context, objectives and motivation of the project.
  - **[Resources and References](./task/Libraries_used.md)**: Python packages, libraries and external references used throughout the project.

  - **Comparisons between the original and the adapted implementation**
    - **[Parameters](./task/comparison/Parameters.ipynb)**: Compares the original and modified `default_params.yml` configuration, explaining the changes introduced for neuronal burst classification.
    - **[DeepONet](./task/comparison/DON.ipynb)**: Compares the original DeepONet implementation with the adapted version used in this project, highlighting the modifications to the training pipeline and dataset handling.
    - **[Utility](./task/comparison/Utility.ipynb)**: Compares the original and modified dataset loading and preprocessing utilities, including the new normalization procedures and burst dataset management.
    - **[Architectures](./task/comparison/Architectures.ipynb)**: Compares the original neural network architectures with the adapted versions, describing the architectural changes introduced for burst modelling.
    - **[Init](./task/comparison/Init.ipynb)**: Compares the original and modified `__init__.py` file, showing how the project modules were reorganized and extended.

    
- **[Python related content](./codes)**: Codes, datasets and resources used throughout the project.
  - **[Burst recordings](./task/bursting)**: Experimental `.abf` files containing neuronal activity (future implementation).
  - **[Task codes](./codes)**: Python scripts developed for the different tasks carried out during the project.
  - **[Data simulation](./codes/Data_simulation.ipynb)**: Scripts for generating simulated neuronal burst datasets from the four-dimensional ODE model.
  - **[Google Colab notebooks](./codes/Execute_in_Colab.ipynb)**: Ready-to-run code for dataset visualization and model training in Google Colab.

