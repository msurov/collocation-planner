import svgpathtools
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def get_fill_color(style):
    if 'fill' not in style: return None
    r,g,b = svgpathtools.hex2rgb(style['fill'])
    a = float(style['fill-opacity']) if 'fill-opacity' in style else 1.
    return (r/255.,g/255.,b/255.,a)

def get_stroke_color(style):
    if 'stroke' in style:
        stroke = style['stroke']
        if stroke == 'none':
            return None
        r,g,b = svgpathtools.hex2rgb(stroke)
    else:
        return None
    if 'stroke-opacity' in style:
        a = float(style['stroke-opacity'])
    else:
        a = 1
    color = (r/255, g/255, b/255, a)
    return color

def get_stroke_width(style):
    if 'stroke-width' in style:
        return float(style['stroke-width'])
    return 1

def complex2pair(z):
    return (z.real, z.imag)

def get_segment_vertcmd(obj):
    if isinstance(obj, svgpathtools.path.Line):
        v = [complex2pair(obj.start), complex2pair(obj.end)]
        c = [Path.MOVETO, Path.LINETO]
    elif isinstance(obj, svgpathtools.path.Arc):
        ccgen = obj.as_cubic_curves()
        v = []
        c = []
        for cc in ccgen:
            v += [complex2pair(cc.start), complex2pair(cc.control1), complex2pair(cc.control2), complex2pair(cc.end)]
            c += [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    elif isinstance(obj, svgpathtools.path.CubicBezier):
        v = [complex2pair(obj.start), complex2pair(obj.control1), complex2pair(obj.control2), complex2pair(obj.end)]
        c = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        assert False, f'Unimplemented object type {type(obj)}'

    return v,c

def get_pathobj(path):
    v = []
    c = []

    for i in range(len(path)):
        vi,ci = get_segment_vertcmd(path[i])

        if i > 0 and np.allclose(path[i-1].end, path[i].start):
            v += vi[1:]
            c += ci[1:]
        else:
            v += vi
            c += ci

    return Path(v, c)

def get_style_props(attrs):
    if 'style' in attrs:
        entries = attrs['style'].split(';')
        return dict([e.split(':') for e in entries])
    return None

def get_fill_color(style):
    if 'fill' not in style: return None
    r,g,b = svgpathtools.hex2rgb(style['fill'])
    a = float(style['fill-opacity']) if 'fill-opacity' in style else 1.
    return (r/255.,g/255.,b/255.,a)

def load_pathes(svgpath):
    paths,attributes = svgpathtools.svg2paths(svgpath)
    objs = {}

    for p,a in zip(paths,attributes):
        if len(p) == 0:
            continue
        path = get_pathobj(p)
        style = get_style_props(a)
        objs[a['id']] = (path, style)

    return objs

def clone_pathpatch(pp):
    res = PathPatch(pp.get_path())
    res.update_from(pp)
    return res

def clone_patches(patches):
    return [clone_pathpatch(pp) for pp in patches]
