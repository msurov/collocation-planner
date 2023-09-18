import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from mplsvgpath import load_pathes, \
    get_stroke_color, get_fill_color, get_stroke_width
from matplotlib.transforms import Affine2D
from matplotlib.patches import PathPatch
from matplotlib import animation
from matplotlib.colors import hsv_to_rgb


def get_links_positions(q):
    q = np.array(q)
    x = np.zeros(q.shape)
    x[...,0] = q[...,0]
    x[...,1:] = np.sin(q[...,1:])
    x = np.cumsum(x, axis=-1)

    y = np.zeros(q.shape)
    y[...,1:] = np.cos(q[...,1:])
    y = np.cumsum(y, axis=-1)

    return x,y


    # nlinks = len(q) - 1
    # positions = np.zeros((nlinks, 2))
    # positions[0,0] = q[0]
    # positions[1:,0] = np.sin(q[1:-1])
    # positions[1:,1] = np.cos(q[1:-1])
    # positions = np.cumsum(positions, axis=0)
    # return positions


def gen_rgb(i):
    h = ((i * 47) % 251) / 251
    s = 1
    v = 1
    return hsv_to_rgb((h,s,v))

class CartPendAnim:

    def __init__(self, svgpath, nlinks=1, scale=1e-2, flipy=True):
        fig = plt.figure(figsize=(19.20, 10.80))
        plt.axis('equal')
        self.fig = fig
        self.ax = plt.gca()

        self.t0 = Affine2D()
        self.t0.scale(scale, -scale if flipy else scale)

        pathes = load_pathes(svgpath)
        self.create_cart_body(pathes)
        self.create_links(pathes, nlinks)
        self.nlinks = nlinks

    def create_cart_body(self, pathes):
        self.body = []

        for pathid in pathes:
            path,style = pathes[pathid]
            fillcolor = get_fill_color(style)
            strokecolor = get_stroke_color(style)
            linewidth = get_stroke_width(style)

            if pathid == 'link':
                continue

            p = PathPatch(path, fill = fillcolor is not None, 
                fc = fillcolor, ec = strokecolor, lw = linewidth)
            p.set_transform(self.t0 + self.ax.transData)
            self.body.append(p)
            self.ax.add_patch(p)

    def create_links(self, pathes, nlinks):
        self.links = []

        path,style = pathes['link']
        fillcolor = get_fill_color(style)
        strokecolor = get_stroke_color(style)
        linewidth = get_stroke_width(style)

        args = {
            'fill': fillcolor is not None, 
            'fc': fillcolor,
            'ec': strokecolor,
            'lw': linewidth
        }
        for i in range(nlinks):
            args['fc'][0:3] = gen_rgb(i)
            p = PathPatch(path, **args)
            p.set_transform(self.t0 + self.ax.transData)
            self.links.append(p)
            self.ax.add_patch(p)

    def move(self, q):
        x,y = get_links_positions(q)
        thetas = q[1:]
        t = Affine2D()
        t.translate(x[0], y[0])

        for p in self.body:
            p.set_transform(self.t0 + t + self.ax.transData)

        for i,link in enumerate(self.links):
            t = Affine2D()
            t.rotate(-thetas[i])
            t.translate(x[i], y[i])
            link.set_transform(self.t0 + t + self.ax.transData)

    def run(self, simdata, filepath=None, fps=60, animtime=None, speedup=None):
        qsp = make_interp_spline(simdata['t'], simdata['q'], k=1)
        t1 = simdata['t'][0]
        t2 = simdata['t'][-1]
        q = simdata['q']

        x,y = get_links_positions(q)
        x1 = np.min(x)
        x2 = np.max(x)
        y1 = np.min(y)
        y2 = np.max(y)
        plt.plot(x1, y1, 'o', alpha=0)
        plt.plot(x2, y2, 'o', alpha=0)
        self.ax.set_axisbelow(True)

        plt.grid(True)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99)

        if speedup is not None:
            assert speedup > 0
            nframes = int((t2 - t1) * fps / speedup) + 1
        elif animtime is not None:
            assert animtime > 0
            nframes = int(animtime * fps) + 1
            speedup = (t2 - t1) / animtime
        else:
            nframes = int((t2 - t1) * fps) + 1
            speedup = 1

        def animinit():
            return self.body + self.links

        def animupdate(iframe):
            ti = iframe * speedup / fps
            q = qsp(ti)
            self.move(q)
            return self.body + self.links

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
    anim = CartPendAnim('fig/cartpend.svg', nlinks=3)
    t = np.linspace(0, 10, 1000)
    simdata = {
        't': t,
        'q': np.array([t, 2*t+1, 3*t+2, -4*t+2]).T
    }
    anim.run(simdata)


if __name__ == '__main__':
    test()

