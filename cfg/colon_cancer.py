__author__ = 'Vlad Popovici'

HE_OPTS = {'gauss1': np.sqrt(2.0),
    'gauss2': 1.0/np.sqrt(2.0),
    'strel1':  morph.disk(3),
    'bregm': 3.5,
    # options for nuclei extraction at 40x magnification:
    '40x_nuclei_min_area': 30}


X40 = {'gauss1': np.sqrt(2.0),
    'gauss2': 1.0/np.sqrt(2.0),
    'nuclei_regions_strel1' : (19,19),
    'detect_nuclei_blob_min_sg' : 2.5,
    'detect_nuclei_blob_max_sg' : 7.5,
    'detect_nuclei_blob_thr' : 0.01}

X20 = {'gauss1': np.sqrt(2.0),
    'gauss2': 1.0/np.sqrt(2.0),
    'nuclei_regions_strel1' : (9,9) }
