QPATH: Quantitative methods for pathology.

This package provides a series of libraries and tools for quantitating various features from histopathology images.
There are a number of image processing and analysis methods specifically designed to work on high content images, as
whole-slide scans of pathology slides. 

Developing such methods is almost impossible without making a number of simplifying assumptions. To start with, 
we assume the magnification of the microscope was set to 20x or 40x, so rather finer details are visible at highest
resolution. Also, the main pathologies of interest are the colon and breast cancers. Naturally, a number of
methods and default parameters make explicit or implicit assumptions of specific tissue characteristics. As much
as possible, these assumptions are either marked for progammers to adapt or are isolated in configuration files.

The project relies on other excellent and more generic image processing packages:

- Scikit-Image: http://www.scikit-image.org
- Mahotas: http://luispedro.org/software/mahotas/

and could not be possible without

- Numpy and SciPy: http://www.scipy.org
