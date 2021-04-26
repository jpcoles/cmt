#
# Read membrane thickness data and plot contours.
# Optionally subtract various mean values (see help).
#
# Copyright(c) 2021 Jonathan Coles <jonathan.coles@tum.de>
#
# This program is distributed under the GNU GPL v2 license.
# See LICENSE for details.
#

try:
    # See if the CMashing library is installed. This will automatically
    # register the colormaps.
    import cmasher as cmr
except:
    pass

import sys,os
import numpy as np
import pylab as pl
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from matplotlib.colors import Normalize, BoundaryNorm

#
# Escape codes for colored output.
#
class C:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ERROR     = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

def main(args):

    vmin = args.vmin
    vmax = args.vmax

    #
    # Load the mean if it's from a file.
    #
    mean = 0
    if args.mean not in ['local', 'global']:
        try:
            mean = float(args.mean)
        except:
            if not os.path.exists(args.mean):
                warn(f'Mean file {fname} does not exist. Skipping.')
            else:
                mean = np.mean(loaddat(args.mean, args)['D'], axis=0)

    #
    # Process the files and subtract off the mean, or compute the global mean.
    # Also remember the extremes of the data.
    #
    gbl_mean = 0
    Ms = []
    vmin = +np.inf
    vmax = -np.inf
    for i,fname in enumerate(args.file):
        if not os.path.exists(fname):
            warn(f'{fname} does not exist. Skipping.')
            continue
        M = loaddat(fname, args)
        M['AVG'] = np.mean(M['D'], axis=0) 

        if args.mean == 'local':
            M['AVG'] -= np.mean(M['AVG'])
        elif args.mean == 'global':
            gbl_mean += np.mean(M['AVG'])
        elif args.mean != 0:
            M['AVG'] -= mean

        if vmin is not None: vmin = min(vmin, np.amin(M['AVG']))
        if vmax is not None: vmax = max(vmax, np.amax(M['AVG']))
        Ms.append( [fname, M] )


    #
    # Subtract off the global mean if requested.
    #
    if args.mean == 'global':
        gbl_mean /= len(Ms)
        vmin = +np.inf
        vmax = -np.inf
        for [fname,M] in Ms:
            M['AVG'] -= gbl_mean
            if vmin is not None: vmin = min(vmin, np.amin(M['AVG']))
            if vmax is not None: vmax = max(vmax, np.amax(M['AVG']))

    vmin = args.vmin if args.vmin is not None else vmin
    vmax = args.vmax if args.vmax is not None else vmax

    #
    # Plot the global average as the first plot.
    #
    if args.plot_global_average:
        nplots = len(Ms)
        if nplots:
            X    = Ms[0][1]['X']
            Y    = Ms[0][1]['Y']
            Davg = Ms[0][1]['AVG']
            for [fname,M] in Ms[1:]:
                Davg += M['AVG']
            Davg /= len(Ms)
            Ms.insert(0, ['Average', dict(AVG=Davg, X=X, Y=Y)])

    nplots = len(Ms)
    if nplots == 0: return

    #
    # Figure out how many rows and columns we need to make
    #
    if args.rows and not args.cols:
        nrows = args.rows
        ncols = int(np.ceil(nplots / nrows))
    elif args.cols and not args.rows:
        ncols = args.cols
        nrows = int(np.ceil(nplots / ncols))
    else:
        ncols = int(np.ceil(np.sqrt(nplots)))
        nrows = int(np.ceil(nplots / ncols))

    assert (nrows * ncols >= nplots)

    figopts  = dict(figsize=np.array([4.0 * ncols,3 * nrows]) * 172.5/72.27,dpi=72.27)
    saveopts = dict(dpi=72.27, bbox_inches='tight')

    fig,ax = pl.subplots(nrows=nrows, ncols=ncols, squeeze=False, **figopts)

    cmap=cm.get_cmap(args.colormap)

    contour_kwargs = dict(cmap=cmap, extend='both')

    #
    # Setup the contouring arguments.
    #
    if vmax > vmin:

        normalizer=Normalize(vmin,vmax)
        im=cm.ScalarMappable(norm=normalizer, cmap=cmap)
        #normalizer = BoundaryNorm(contour_levels, cmap.N)

        if args.contour_levels is not None:
            contour_levels = np.linspace(vmin,vmax,args.contour_levels+1)
            contour_kwargs['levels'] = contour_levels
            contour_kwargs['norm'] = BoundaryNorm(contour_levels, cmap.N)
    else:
        contour_kwargs['levels'] = 0


    #
    # Iterate over the data and plot.
    #
    for i,[fname,M] in enumerate(Ms):

        r,c = np.divmod(i,ncols)

        _ax = ax[r,c]

        if args.filled:
            cntr=_ax.contourf( M['Y'],M['X'], M['AVG'], origin=None, **contour_kwargs )
        else:
            cntr=_ax.contour(  M['Y'],M['X'], M['AVG'], origin=None, **contour_kwargs )

        divider = make_axes_locatable(_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        _ax.set_xlabel('y [A]')
        _ax.set_ylabel('x [A]')
        _ax.set_title(fname)

        cbar_kwargs = dict(cax=cax)
        if args.vmin and args.vmax:
            cbar_kwargs['extend'] = 'both'
        elif args.vmin:
            cbar_kwargs['extend'] = 'min'
        elif args.vmax:
            cbar_kwargs['extend'] = 'max'

        if args.smooth_colorbar:
            cbar = fig.colorbar(im, **cbar_kwargs)
        else:
            cbar = fig.colorbar(cntr, **cbar_kwargs)

        cbar.set_label('Thickness [A]')

    title = []

    if args.date:
        from datetime import datetime
        now = datetime.now()
        date_time = now.strftime("Created on: %m/%d/%Y, %H:%M:%S")
        title.append(date_time)

    if args.title:
        title.append(args.title)

    if title:
        fig.suptitle('\n\n'.join(title))

    #
    # Display or save the results.
    #
    if args.ofile is None:
        pl.show()
    else:
        if args.verbose >= 1:
            print(C.OKGREEN + 'Saving plot to %s' % args.ofile + C.ENDC)
        pl.savefig(args.ofile, **saveopts)

def loaddat(fname, args):

    if args.verbose >= 2:
        print('Loading %s' % fname)

    try:
        X,Y,D = pl.loadtxt(fname, skiprows=1, usecols=[1,2,3], unpack=True)
    except Exception as e:
        error(str(e))
        sys.exit(1)
    X = np.unique(X)
    Y = np.unique(Y)

    nx,ny = len(X), len(Y)
    gridsize = nx * ny

    nframes = D.shape[0] // gridsize
    assert D.shape[0] == nframes * gridsize

    M = np.reshape(D, [nframes,nx,ny])

    return dict(D=M, X=X,Y=Y)

def warn(s):
    print(C.WARNING + 'WARNING: ' + s + C.ENDC)

def error(s):
    print(C.ERROR + 'ERROR: ' + s + C.ENDC)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Plot membrane thickness grids as contours.', 
        epilog='Written by Jonathan Coles <jonathan.coles@tum.de>')

    parser.add_argument(
        'file',     help='DAT file(s) to plot.', metavar='dat-file', nargs='+')
    parser.add_argument(
        '-o',       help='Filename for figure. File extension determines file type.', metavar='output-file', type=str, dest='ofile', default=None)
    parser.add_argument(
        '--verbose', '-v', help="Set verbose level. Repeated usage increases verbosity.", action='count', default=0)
    parser.add_argument(
        '--levels', help='Set the number of contour levels.', type=int, dest='contour_levels', default=None)
    parser.add_argument(
        '--title',  help='Title for figure.', type=str, default='')
    parser.add_argument(
        '--mean',   help='A real number, "local", "global", or the mean over another dat file. Remove the given mean from each figure.', default=0)
    parser.add_argument(
        '--vmin',   help='Set minimum contour value.', type=float, default=None)
    parser.add_argument(
        '--vmax',   help='Set maximum contour value.', type=float, default=None)
    parser.add_argument(
        '--rows',   help='Set maximum number of plot rows.', type=int, default=None)
    parser.add_argument(
        '--cols',   help='Set maximum number of plot columns.', type=int, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--no-date',   help='Do no add the creation date to the title.', action='store_false', dest='date')
    group.add_argument(
        '--date',      help='Add the creation date to the title.', action='store_true', dest='date', default=True)
    parser.add_argument(
        '--plot-global-average',    help='Plot the average contour over all inputs.', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--no-filled',  help='Plot contours as lines', action='store_false', dest='filled')
    group.add_argument(
        '--filled',     help='Plot filled contours.', action='store_true', dest='filled', default=True)
    parser.add_argument(
        '--smooth-colorbar',        help='Use a smoothly varying colorbar.', action='store_true')
    parser.add_argument(
        '--colormap',               help='Change the colormap to one supported by Matplotlib or CMashing. Use "list" to see all available colormaps.', type=str, default='viridis')

    args = parser.parse_args()

    if args.verbose >= 3:
        print('Arguments:')
        print(args)

    if args.colormap == 'list':
        print('Available colormaps:')
        for cm in pl.colormaps():
            print('    ' + cm)
        sys.exit(0)


    main(args)
