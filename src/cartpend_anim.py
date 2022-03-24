import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from mplsvgpath import load_path_patches, \
    get_stroke_color, get_fill_color, get_stroke_width
from matplotlib.transforms import Affine2D
from matplotlib.patches import PathPatch
from matplotlib import animation

class CartPendAnim:

    def __init__(self, svgpath, scale=1, flipy=False):
        fig = plt.figure(figsize=(19.20, 10.80))
        plt.axis('equal')
        ax = plt.gca()
        objs = load_path_patches(svgpath)
        patches = {}

        self.t0 = Affine2D()
        self.t0.scale(scale, -scale if flipy else scale)

        for k in objs:
            path,style = objs[k]
            fillcolor = get_fill_color(style)
            strokecolor = get_stroke_color(style)
            linewidth = get_stroke_width(style)

            if fillcolor is not None:
                patch = PathPatch(path, fill=True, fc=fillcolor, ec=strokecolor, lw=linewidth)
            else:
                patch = PathPatch(path, fill=False, ec=strokecolor, lw=linewidth)

            patches[k] = patch
            patch.set_transform(self.t0 + ax.transData)
            ax.add_patch(patch)
        
        self.fig = fig
        self.ax = ax
        self.patches = patches

    def move(self, q):
        x,theta = q
        for k in self.patches:
            p = self.patches[k]
            t = Affine2D()
            t.clear()
            if k == 'pendulum':
                t.rotate(-theta)
                t.translate(x, 0)
                p.set_transform(self.t0 + t + self.ax.transData)
            else:
                t.translate(x, 0)
                p.set_transform(self.t0 + t + self.ax.transData)

    def run(self, simdata, filepath=None, fps=60, animtime=None, speedup=None):
        qsp = make_interp_spline(simdata['t'], simdata['q'], k=1)
        t1 = simdata['t'][0]
        t2 = simdata['t'][-1]

        x = simdata['q'][:,0]
        x1 = np.min(x)
        x2 = np.max(x)

        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(x1 - 1, x2 + 1)
        self.ax.set_axisbelow(True)

        plt.grid(True)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99)

        if speedup is not None:
            assert speedup > 0
            nframes = int((t2 - t1) * fps / speedup)
        elif animtime is not None:
            assert animtime > 0
            nframes = int(animtime * fps)
            speedup = (t2 - t1) / animtime
        else:
            nframes = int((t2 - t1) * fps)
            speedup = 1

        def animinit():
            return self.patches.values()

        def animupdate(iframe):
            ti = iframe * speedup / fps
            q = qsp(ti)
            self.move(q)
            return self.patches.values()

        anim = animation.FuncAnimation(self.fig, animupdate, init_func=animinit, frames=nframes, blit=True)
        if filepath is not None:
            if filepath.endswith('.gif'):
                Writer = animation.writers['imagemagick']
                writer = Writer(fps=fps)
                anim.save(filepath, writer, dpi=40)
            else:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='Maksim Surov'), bitrate=800*60)
                anim.save(filepath, writer)
        else:
            plt.show()


def test():
    anim = CartPendAnim('fig/cartpend.svg', scale=0.01, flipy=True)
    t = np.linspace(0, 10, 1000)
    simdata = {
        't': t,
        'q': np.array([0*t, t]).T
    }
    anim.run(simdata)


if __name__ == '__main__':
    test()

