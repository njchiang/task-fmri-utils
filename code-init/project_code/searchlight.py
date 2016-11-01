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
import sys
import os


def runsub(sub, thisContrast, r, dstype='raw', roi='grayMatter', filter_params=[49,2], write=False):



def main(argv):
    import getopt
    # every possible variable
    con = None
    debug = False
    write = False
    roi = 'grayMatter'
    dstype = 'raw'
    projecttitle = ''
    r = 4  # searchlight radius
    sg_params = [49, 2]

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
        elif opt in ("-p", "--project"):
            projecttitle = arg

    if not con:
        print "not a valid contrast... exiting"
        sys.exit(1)


    if sys.platform == 'darwin':
        # plat = 'mac'
        sys.path.append(os.path.join("/Users", "njchiang", "GitHub", "task-fmri-utils"))
        sys.path.append(os.path.join("/Volumes", "JEFF", projecttitle))
    else:
        sys.path.append(os.path.join("D:\\", "GitHub", "task-fmri-utils"))
        sys.path.append(os.path.join("D:\\", "fmri", projecttitle))
    from code import projectutils as pu
    from analysis import projectanalysis as pa

    # need a trial_type attribute
    paths, sublist, contrasts = pu.initpaths()
    print "Mask: " + str(roi)
    print "Full Model: " + str(con)
    print "Searchlight Radius: " + str(r)
    print "Write results: " + str(write)

    for s in sublist.keys():
        runsub(sub=s, thisContrast=con, r=r, dstype=dstype, write=write,
               filter_params=sg_params, roi=roi)

if __name__ == "__main__":
    main(sys.argv[1:])