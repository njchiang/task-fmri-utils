#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
encoding.py: basic encoding process
inputs: subject data, model, parameters
outputs: either matrix or writes to NIFTI
"""

#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# this script represents just throwing pymvpa at the problem. doesn't work great, and I suspect it's
# because we're using an encoding model.
"""
Wrapper for PyMVPA analysis. Implemented as a program to be able to run multiple models quickly.

For troubleshooting and parameter testing an ipynb exists too.
"""
# initialize stuff
PROJECTTITLE = ''

import sys
import os
if sys.platform == 'darwin':
    sys.path.append(os.path.join("/Users", "njchiang", "GitHub", "task-fmri-utils"))
    sys.path.append(os.path.join("/Volumes", "JEFF", "UCLA", PROJECTTITLE, "code"))
else:
    sys.path.append(os.path.join("D:\\", "GitHub", "task-fmri-utils"))
    sys.path.append(os.path.join("D:\\", "fmri", PROJECTTITLE, "code"))
from fmri_core import projectanalysis_pymvpa as pa
from project_code import projectutils as pu
# need a trial_type attribute
PATHS, SUBLIST, CONTRASTS = pu.initpaths()


def runsub(sub, con, roi='grayMatter', addmotion=None, sg_params=[49,2],
           writeopts=None, bsargs=bsargs):
    ds, events = pu.preprocess_data(PATHS, SUBLIST, sub,
                                    filter_params=sg_params, roi=roi, z=True)
    design_kwargs = {'hrf_model': 'canonical', 'drift_model': 'blank'}
    if addmotion:
        rds, des = pu.preprocess_encoding(ds=ds, events=events, c=con,
                                          mp=addmotion, design_kwargs=design_kwargs)
    else:
        rds, des = pu.preprocess_encoding(ds=ds, events=events, c=con,
                                          mp=None, design_kwargs=design_kwargs)
    wts, oalphas, res, ceil = pa.encoding(PATHS, ds=rds, des=des, c=con,
                                          chunklen=12, nchunks=4, alphas=None,
                                          writeopts=writeopts, bsargs=bsargs)


def main(argv):
    import getopt
    # every possible variable
    con = None
    debug = False
    write = False
    roi = 'grayMatter'
    dstype = 'raw'
    r = 4  # searchlight radius
    sg_params = [49, 2]
    addmotion = True
    try:
        # figure out this line
        opts, args = getopt.getopt(argv, "dwh:m:c:b:r:f",
                                   ["mask=", "contrast=", "debug", "radius=", "datatype=", "filters=",
                                    "project=", "write"])
    except getopt.GetoptError:
        print 'searchlight.py -m <maskfile> -c <contrast> -f <filter_params -r <radius> \
            -b <dstype> -w -d'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'searchlight.py -m <maskfile> -c <contrast> -f <filter_params> -r <radius> \
            -b <dstype> -w -d'
            sys.exit()
        elif opt in ("-m", "--mask"):
            roi = arg
        elif opt in ("-c", "--contrast"):
            con = arg
        elif opt in ("-d", "--debug"):
            print "debug mode"
            debug = True
        elif opt in ("-w", "--write"):
            write = True
        elif opt in ("-r", "--radius"):
            r = arg
        elif opt in ("-b", "--dstype"):
            dstype = arg
        elif opt in ("-f", "--filter"):
            sg_params = arg.split(',')
        elif opt in ("-n", "--nomotion"):
            addmotion = False


    if not con:
        print "not a valid contrast... exiting"
        sys.exit(1)


    print "Mask: " + str(roi)
    print "Full Model: " + str(con)
    print "Searchlight Radius: " + str(r)
    print "Write results: " + str(write)

    if isinstance(con):
        con = [con]
    if addmotion:
        mc_params = pu.loadmotionparams(PATHS, SUBLIST)
    for s in SUBLIST.keys():
        if write:
            writeopts = {'outdir': os.path.join('multivariate', 'encoding'),
                         'sub': s, 'roi': roi, 'con': '+'.join(con)}
        else:
            writeopts = None
        bsargs = {'part_attr': 'chunks', 'mode': 'test', 'single_alpha': True, 'normalpha': False,
                  'nboots': 1, 'corrmin': .2, 'singcutoff': 1e-10, 'joined': None, 'plot': False, 'use_corr': True}
        runsub(paths=PATHS, sub=s, con=con, r=r, addmotion=mc_params[s], dstype=dstype, writeopts=writeopts,
               sg_params=sg_params, roi=roi, bsargs=bsargs)

if __name__ == "__main__":
    main(sys.argv[1:])