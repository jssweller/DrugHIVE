import numpy as np
from matplotlib import colors
from copy import copy

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def combine_figs(figlist):
    '''Combines list of plotly figures into one.'''
    fig0 = copy(figlist[0])
    for fig in figlist[1:]:
        for trace in fig.data:
            fig0.add_trace(trace)
    fig0.update_layout(fig0.layout)
    return fig0


def make_plotly_grid(figlist, cols=4, figsize=(300, 300), zoom=0.5, titles=None, specs=None):
    '''Makes grid of plotly figures as subplots.'''
    cols = min(cols, len(figlist))
    rows = len(figlist) // cols + (len(figlist) % cols > 0)

    if specs is None:
        spec = {'type': 'surface'}
        specs = [[spec]*cols]*rows

    figa = make_subplots(rows=rows, cols=cols,
                                                horizontal_spacing = 0.01,
                                                vertical_spacing = 0.01,
                                                specs=specs,
                                                subplot_titles=titles,
                                                )
    for i in range(rows):
        for j in range(cols):
            p = int(i*cols+j)
            if p > len(figlist)-1:
                continue
            if i<rows:
                if not isinstance(figlist[p], list):
                    figlist[p] = [figlist[p]]
                for fg in figlist[p]:
                    if p!=0:
                        fg.update_traces(showlegend=False)
                    for k, trace in enumerate(fg.data):
                        figa.add_trace(trace, row=i+1,col=j+1)
                        figa.layout['scene%d'%(i*cols + j + 1)].update(fg.layout.scene)

    figa.update_scenes(
        camera = {
                'center': { 'x': 0, 'y': 0, 'z': 0 }, 
                'eye': { 'x': 0.5/zoom, 'y': 0.5/zoom, 'z': 0.5/zoom }, 
                    })
    
    figa.update_layout(width=cols*figsize[0], height=rows*figsize[1])
    return figa
    
    
def make_legend_names_unique(fig):
    legnames={}
    for i in range(len(fig.data)):
        d = fig.data[i]
        if d.legendgroup not in legnames.keys():
            legnames[d.legendgroup] = []
        if d.name not in legnames[d.legendgroup]:
            legnames[d.legendgroup].append(d.name)
            fig.data[i].showlegend=True
        else:
            fig.data[i].showlegend=False
    return fig


def sort_legend(fig):
    legnames = {}
    for i in range(len(fig.data)):
        d = fig.data[i]
        if d.legendgroup not in legnames.keys():
            legnames[d.legendgroup] = []
        if d.name not in legnames[d.legendgroup] and (d.legendgroup is not None):
            legnames[d.legendgroup].append((i,d.name))
            fig.data[i].showlegend=True
        else:
            fig.data[i].showlegend=False
    
    figdat2 = []
    tracelist = []
    for group, vals in legnames.items():
        tracelist += sorted(vals, key=lambda x: int(x[1].split('=')[1].split(' ')[0].strip()))
    print(tracelist)
    
    idxs = [i for i,_ in tracelist]
    idxs += [i for i in range(len(fig.data)) if i not in idxs]
    
    for i in idxs:
        figdat2.append(fig.data[i])
    
    fig.data = figdat2
    return fig


def save_html(fig, file, bgcolor='#ffffff'):
    new_fig = with_css_style(fig, bgcolor=bgcolor)
    with open(file, 'w') as fp:
        fp.write(new_fig)
        
        
def with_css_style(fig, bgcolor='#ffffff'):
    plot_div = plotly.offline.plot(fig,output_type = 'div')
    template = """
    <head>
    <body style="background-color:{bgcolor:s};">
    </head>
    <body>
    {plot_div:s}
    </body>""".format(plot_div = plot_div, bgcolor = bgcolor)
    return template


def adjust_rgb_brightness(color, brightness):
    if isinstance(color, np.ndarray):
        color[:3] += brightness
    elif isinstance(color, str):
        if 'rgb' in color:
            color = color.replace('rgb(','')[:-1].split(',')
            color = np.asarray(color, dtype=int) + brightness
        elif color in colors.cnames.keys():
            color = colors.to_rgba_array(color).flatten()[:3]
            color = color * 255 + brightness
            color = color.round().astype(int)
    else:
        print('other', color)
        return color
    color = np.maximum(np.minimum(color.round(), 255), 0)
    color = 'rgb' + str(tuple(color))
    return color


def get_mesh_lines(verts, faces):
    Xe = []
    Ye = []
    Ze = []
    for T in verts[faces]:
        Xe.extend([T[k%3][0] for k in range(4)]+[ None])
        Ye.extend([T[k%3][1] for k in range(4)]+[ None])
        Ze.extend([T[k%3][2] for k in range(4)]+[ None])
    return Xe, Ye, Ze