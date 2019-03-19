=================================
Forward Simulation Analysis (FSA)
=================================

After **Data Smoothing** (example_) and initial **DC Modeling** (example__) , we can perform **Forward Simulation Analysis (FSA)** to calculate accurate diffusion coefficients from diffusion profile.
The process requires many mannual inputs as smoothing goes on, so please do not close any window during the process.

.. code-block:: python

    import matplotlib.pyplot as plt
    from pydiffusion.io import read_csv, save_csv
    from pydiffusion.plot import profileplot, DCplot, SFplot
    from pydiffusion.fsa import FSA

Read data from previous data smoothing & DC modeling
----------------------------------------------------

Here we read previous results of Ni-Mo 1100C 1000 hours datasets, and plot them out.

(The ``Xspl`` info required to input manually if initial DC modeling is read from file)

.. code-block:: python

    NiMo_sm, diffsys_init = read_csv('data/NiMo_DC_init.csv')
    NiMo_exp, _ = read_csv('data/NiMo_exp.csv')
    Xp = [[.05, .2],
        [.5, .515],
        [.985]]
    diffsys_init.Xspl = Xp
    time = 3600*1000

    fig = plt.figure(figsize=(16, 6))
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    profileplot(NiMo_exp, ax1, ls='none', marker='o', fillstyle='none')
    profileplot(NiMo_sm, ax1, c='g', lw=2)
    SFplot(NiMo_sm, time, ax=ax2, Xlim=[0, 1])
    DCplot(diffsys_init, ax2, c='r', lw=2)
    plt.show()

.. image:: https://github.com/zhangqi-chen/pyDiffusion/blob/master/docs/examples/FSA_files/FSA_1.png

Left figure: smoothed profile vs. experimental profile Right figure: initial DC modeling vs. Sauer-Fraise calculation

Forward simulation analysis (FSA)
---------------------------------

**Forward simulation analysis (FSA)** is performed in the following steps:

1. Use DC modeling to simulate the diffusion process.
2. Compare the simulated profile vs. experimental profile, see if the difference between 2 profiles is smaller enough.
3. If not, compare the simulated profile vs. smoothed profile, adjust the DC modeling using **Point Mode** or **Phase Mode**.
4. Use the new DC modeling to simulate the diffusion process again...

**Point Mode**: Can be applied only when reference points in **Spline** functions are provided, i.e. `DiffSystem.Xspl is not None`. Each diffusion coefficient value at reference point is adjusted at first, then **Spline** function is generated depending on adjusted diffusion coefficients. In the logD vs. X figure, ***the shape of DC curve might change***.

**Phase Mode**: Can be applied on both **Spline** and **UnivariateSpline** modeling. The diffusion coefficients inside each phase are adjusted by the phase width comparison or manually input. In the logD vs. X figure, ***the shape of DC curve doesn't change***.

In the `FSA` function, default error is defined by the difference between smoothed profile and experimental profile. The error of each simulation round is defined by the difference between simulated profile and experimental profile.

The stop criteria of error is the cap for manually adjustment. One can compare the simulation result and experimental profile after each diffusion simulation to make decision of next adjustment.

* If the error is larger than this cap, the function will adjust the system by **Point Mode** or **Phase Mode** automatically;
* if the error is smaller than this cap, the function will ask to choose between further manually adjustment or exit. Of course, automatically adjustment can also be used when the criteria is reached.

In this example, automatic **Phase Mode** will be used.

.. code-block:: python

    NiMo_sim, diffsys_fsa = FSA(NiMo_exp, NiMo_sm, diffsys_init, time, Xlim=[0, 1], n=[250, 300])

Outputs:

.. code-block::

    Meshing Num=279, Minimum grid=0.525563 um

    Default error = 0.000995
    Input the stop criteria of error: [0.001991]
    .0016

    Use Phase Mode? [n]
    (The shape of diffusivity curve does not change)
    y
    653.080/1000.000 hrs simulated
    Simulation Complete
    Simulation 1, error = 0.002112(0.001600)
    Simulation Complete
    Simulation 2, error = 0.002396(0.001600)
    Simulation Complete
    Simulation 3, error = 0.001930(0.001600)
    Simulation Complete
    Simulation 4, error = 0.001585(0.001600)

    Satisfied with FSA? [n]

    Use Point Mode (y) or Phase Mode (n)? [y]n

    Phase Mode
    Manually input for each phase? [n]
    925.701/1000.000 hrs simulated
    Simulation Complete
    Simulation 5, error = 0.001418(0.001600)

    Satisfied with FSA? [n]y

FSA results
-----------

.. code-block:: python

    fig = plt.figure(figsize=(16, 6))
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    profileplot(NiMo_exp, ax1, ls='none', marker='o', fillstyle='none')
    profileplot(NiMo_sm, ax1, c='g', lw=2)
    profileplot(NiMo_sim, ax1, c='r', lw=2)
    SFplot(NiMo_sm, time, ax=ax2, Xlim=[0, 1])
    DCplot(diffsys_fsa, ax2, c='r', lw=2)
    plt.show()

.. image:: https://github.com/zhangqi-chen/pyDiffusion/blob/master/docs/examples/FSA_files/FSA_2.png

Save FSA results
----------------

Usually FSA results are saved by combining DC data with simulated profile data.

.. code-block:: python

    save_csv('NiMo.csv', NiMo_sim, diffsys_fsa)

Congratulations! Now you can perform forward simulation analysis based on raw diffusion data!

.. _example: https://github.com/zhangqi-chen/pyDiffusion/blob/master/docs/examples/DataSmooth.rst
.. __: https://github.com/zhangqi-chen/pyDiffusion/blob/master/docs/examples/DCModeling.rst