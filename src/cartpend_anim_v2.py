import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation


class CartPendAnim:
    def __init__(self):
        fig = plt.figure('Quadrotor', figsize=(19.20,10.80))
        plt.axis('equal')
        ax = plt.gca()
        self.ax = ax
        self.fig = fig
        self.l = 0.5
        self.carw = 0.2
        self.carh = 0.1
        self.line, = plt.plot([0,0], [0,self.l])
        self.carbody = Rectangle([0,0], self.carw, self.carh)
        ax.add_patch(self.carbody)
        self.tip = Circle([0,0], radius=0.02)
        ax.add_patch(self.tip)

    def update(self, q):
        x,theta = q
        x1 = x
        y1 = 0
        x2 = x1 + self.l * np.sin(theta)
        y2 = y1 + self.l * np.cos(theta)
        self.carbody.set_xy([x1 - self.carw/2, y1 - self.carh/2])
        self.line.set_data([x1, x2], [y1, y2])
        self.tip.set_center([x2, y2])
    
    @property
    def patches(self):
        return self.carbody, self.line, self.tip

    def run(self, simdata, filepath=None, fps=60, animtime=None, speedup=None):
        qsp = make_interp_spline(simdata['t'], simdata['q'], k=1)
        t1 = simdata['t'][0]
        t2 = simdata['t'][-1]

        x = simdata['q'][:,0]
        x1 = np.min(x)
        x2 = np.max(x)

        self.ax.set_ylim(-self.l, self.l)
        self.ax.set_xlim(x1, x2)

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
            return self.patches

        def animupdate(iframe):
            ti = t1 + iframe * speedup / fps
            q = qsp(ti)
            self.update(q)
            return self.patches
        
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


if __name__ == '__main__':
    anim = CartPendAnim()
    t = np.linspace(0, 1, 100)
    q = np.array([np.sin(t), -t]).T
    simdata = {
        't': t,
        'q': q
    }
    anim.run(simdata)
