import os, sys
import warnings
import numpy as np
import pandas as pd
import torch
from copy import copy, deepcopy
from matplotlib import pyplot as plt
from collections import Counter
from typing import Union, Iterable

import plotly
from plotly import graph_objects as go
from skimage.measure import marching_cubes

from .molecules import AtomTyperDefault, BaseAtomTyper
from .visutils import adjust_rgb_brightness, make_legend_names_unique, get_mesh_lines


class DensityGrid(object):
    '''
    Density gridding class.
    '''
    def __init__(self,
                 grid_size,
                 n_batch=1,
                 resolution=1,
                 values=None,
                 channels=['default'],
                 channels_dict=None,
                 center=(0, 0, 0),
                 atom_coords=None,
                 atom_types=None,
                 device='cpu',
                 **kwargs):
        
        self.device = device
        self.values = values
        self.grid_size = grid_size
        self.channels = channels
        self.resolution = resolution
        self.center = center
        self.atom_coords = atom_coords
        self.atom_types = atom_types
        self.init_atoms()
        self.channels_dict = channels_dict
        self.n_batch = n_batch


    def __getitem__(self, idx):
        return self.values[idx]

    @property
    def ATOM_COLORS(self):
        ATOM_COLORS = {
                'atomic_num=5 (B)': 'rgb(255,181,181)',  # pale red
                'atomic_num=6 (C)': 'rgb(150,150,150)',  # light grey
                'atomic_num=7 (N)': 'rgb(48, 80, 248)',  # blue
                'atomic_num=8 (O)': 'rgb(240,0,0)',  # red
                'atomic_num=9 (F)': 'rgb(144, 224, 80)',  # soft green
                'atomic_num=11 (Na)': 'rgb(171, 92, 242)',  # soft violet
                'atomic_num=12 (Mg)': 'rgb(138, 255, 0)',  # green
                'atomic_num=15 (P)': 'rgb(255,165,0)',  # orange
                'atomic_num=16 (S)': 'rgb(255,200,50)',  # yellow
                'atomic_num=17 (Cl)': 'rgb(31, 240, 31)',  # vivid lime green
                'atomic_num=19 (K)': 'rgb(143, 64, 212)',  # moderate violet
                'atomic_num=20 (Ca)': 'rgb( 61, 255, 0)',  # lime green
                'atomic_num=26 (Fe)': 'rgb(224, 102, 51)',  # bright orange
                'atomic_num=30 (Zn)': 'rgb(125, 128, 176)',  # desaturated dark blue
                'atomic_num=35 (Br)': 'rgb(165,42,42)',  # brown,
                'atomic_num=53 (I)': 'rgb(75,0,130)',  # violet
            }

        cnum = {int(key.split(' ')[0].split('=')[1]):val for key,val in ATOM_COLORS.items()}
        cnum_str = {'e'+str(key): val for key, val in cnum.items()}
        csymbol = {key.split(' ')[1][1:-1]:val for key,val in ATOM_COLORS.items()}

        ATOM_COLORS.update(cnum)
        ATOM_COLORS.update(cnum_str)
        ATOM_COLORS.update(csymbol)
        DEFAULT_SYMBOLS = ['cumulative','default','X',0]
        ATOM_COLORS.update({key:'rgb(82, 80, 87)' for key in DEFAULT_SYMBOLS})

        ATOM_COLORS['HBA'] = 'rgb(118, 50, 186)' # purple
        ATOM_COLORS['HBD'] = 'rgb(158, 148, 43)' # yellow
        ATOM_COLORS['Aromatic'] = 'rgb(50, 173, 161)' # teal
        ATOM_COLORS['FChargePos'] = 'rgb(240,0,0)' # red
        ATOM_COLORS['FChargeNeut'] = 'rgb(82, 80, 87)' # grey
        ATOM_COLORS['FChargeNeg'] = 'rgb(48, 80, 248)' # blue

        for key in list(ATOM_COLORS.keys()):
            if isinstance(key, str):
                ATOM_COLORS[key.lower()] = ATOM_COLORS[key]
                ATOM_COLORS[key.upper()] = ATOM_COLORS[key]

        return ATOM_COLORS

    @property
    def shape(self):
        return self.values.shape

    @property
    def channels(self):
        return self._channels
    
    @channels.setter
    def channels(self, values):
        values = [x if isinstance(x,list) else [x] for x in values]
        self._channels = values

    @property
    def n_batch(self):
        return self._n_batch
    
    @n_batch.setter
    def n_batch(self, value):
        if hasattr(self,'n_batch'):
            if value > self.n_batch:
                self.expand_grid(newdim=value, axis=0)
        self._n_batch = value       
            
    @property
    def values(self):
        return self._values[:self.n_batch]
    
    @values.setter
    def values(self, values):
        self._values = values

    @property
    def num_channels(self):
        return len(self.channels)

    @property
    def types(self):
        return self.channels

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        self._resolution = value

    @property
    def elements(self):
        return self.channels_dict.get('elements', None)

    @property
    def atom_properties(self):
        return self.channels_dict.get('properties', None)

    @property
    def channels_dict(self):
        return self._channels_dict

    @channels_dict.setter
    def channels_dict(self, values=None):
        self._channels_dict = values

    @property
    def center(self):
        '''Coordinates of grid center (angstroms). i.e. offset of grid coordinates.'''
        return self._center

    @center.setter
    def center(self, center):
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center)
        self._center = center
        
    @property
    def center_grid(self):
        return torch.tensor([int(self.grid_size/self.resolution/2)-0.5]*3).to(self.device)

    @property
    def grid_size_voxels(self):
        return self.values.shape[-1]

    @property
    def grid_offset(self):
        '''Offset of grid center in voxel lengths.
        '''
        return (self.center / self.resolution)

    @property
    def atom_coords(self):
        return self._atom_coords

    @atom_coords.setter
    def atom_coords(self, values):
        values = np.asarray(values)
        if values.ndim == 2:
            values.reshape(-1, *values.shape)
        self._atom_coords = values

    @property
    def atom_coords_grid(self):
        '''Grid coordinates of atoms (voxel length)'''
        return self.atom_coords / self.resolution + self.center_grid - self.grid_offset


    @property
    def atom_types(self):
        return self._atom_types

    @atom_types.setter
    def atom_types(self, values):
        self._atom_types = values

    @property
    def atom_types_nums(self):
        return self.atomtyper.vec2nums(self.atom_types)    
    
    @property
    def atom_types_symbols(self):
        return self.atomtyper.vec2symbols(self.atom_types)

    @property
    def atom_channels(self):
        channels = self.atom_types.clone().flatten()
        channels[:] = -1
        for i, t in enumerate(self.atom_types.flatten()):
            ch = self.get_type_channel(type_num=t)
            if ch is not None:
                channels[i] = self.get_type_channel(type_num=t)
        return channels.reshape(self.atom_types.shape).int()
    
    def get_grid_vals(self, grid, grid_coords, grid_types=None):
        '''Gets voxel values from grid for given grid coordinates and types.'''
        c = grid_coords.round().long()
        if grid_types is not None:
            t = grid_types.long()
            gvals = grid[t, c[:,0], c[:,1], c[:,2]]
        else:
            gvals = grid[:, c[:,0], c[:,1], c[:,2]]
        return gvals
    
    def get_atom_grid_vals(self, grid_coords=None, grid_types=None, use_types=True):
        '''Gets voxel values from grid for each atom.'''
        grid_vals_all = []
        
        if grid_coords is None:
            grid_coords = self.atom_coords_grid
        if grid_types is None and use_types:
            grid_types = self.atom_channels_flat
            
        for i in range(len(self.values)):
            grid_vals_all.append(self.get_grid_vals(grid=self.values[i],
                                                    grid_coords=grid_coords[i],
                                                    grid_types=grid_types[i] if grid_types is not None else None,
                                                ))
        return grid_vals_all   
    

    def is_in_bounds(self, coords, pad, voxels=True):
        ''''Checks if all atom coordinates are in bounds of grid.'''
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)

        assert isinstance(coords, torch.Tensor), f'input coords must be of type torch.Tensor or np.ndarray, received {type(coords)}'

        if not voxels:
            coords = self.angstroms2voxels(coords)
            pad /= self.resolution
        lims = (0 + pad, self.values.shape[-1] - pad - 1)
        b = torch.logical_and((coords >= lims[0]).all(dim=-1), (coords < lims[1]).all(dim=-1))
        b = torch.logical_or(b, torch.isnan(coords).any(dim=-1))
        return b


    def get_type_channel(self, typ):
        '''Returns channel index for given type name or index.'''
        if not isinstance(typ,str):
            typ = self.atomtyper.num2type(typ)
        for i, ch in enumerate(self.channels):
            if isinstance(ch, Iterable):
                if typ in ch:
                    return i
            elif typ == ch:
                return i
        return None

    
    def init_grid(self, gridnum=None):
        if (gridnum is None) or (self.values is None):
            dim = int(self.grid_size/self.resolution)
            self.values = torch.zeros((self.n_batch, self.num_channels, dim, dim, dim))
        else:
            self.values[gridnum] = 0
            
    
    def expand_grid(self, newdim, axis=0):
        '''Adds dimension to grid with empty values.'''
        shapeadd = list(self.values.shape)
        shapeadd[axis] = newdim-self.values.shape[axis]
        self.values = torch.cat([self.values, torch.zeros(shapeadd)], axis=axis)
    
    
    def init_coordinate_encoder(self, encoding='softcube'):
        '''Initializes coordinate encoder.'''
        if encoding == 'softcube' or encoding == 'soft':
            self.coordinate_encoder = CoordinateEncoderCube()
    
    def copy(self):
        '''Returns copy of self.'''
        newself = deepcopy(self)
        newself.values = self.values.clone()
        return newself
    
    def sum(self, axis=0):
        '''Returns sum of grid values along axis.'''
        return self.values.sum(axis=axis)

    def get_mgrid(self):
        X, Y, Z = np.mgrid[0:self.shape[-3], 0:self.shape[-2], 0:self.shape[-1]]
        return X, Y, Z

    def voxels2angstroms(self, grid_coords):
        return (grid_coords + self.grid_offset - self.center_grid) * self.resolution 

    def angstroms2voxels(self, coords):
        return coords / self.resolution + self.center_grid - self.grid_offset
                
    def plot3d(self, 
               gridnum=0, 
               cumulative=False, 
               channels=None, 
               normalize=False, 
               opacity=1, 
               plot_surface=True, 
               plot_lines=False, 
               linewidth=1, 
               colors=None, 
               linecolors=None, 
               isomin=None, 
               isomax=None, 
               channel_plot_names=None, 
               surface_fill=1, 
               showlegend=False, 
               showaxes=False,
               figsize=(1000,1000),
               zoom=1,
               brightness=0,
               **kwargs):
        '''Plots grid values as 3D surface.'''
        
        ATOM_COLORS = self.ATOM_COLORS

        X, Y, Z = self.get_mgrid()
        go_volumes = []
        
        if channel_plot_names is None:
            channel_plot_names = self.channels
        
        if channels is None or channels == 'all':
            plot_channels = np.arange(len(self.channels))
        elif channels in ['atom','atoms','atomic','atomic_num','atomic_nums', 'elements']:
            plot_channels = np.asarray(self.channels_elements_only).astype(int)
        elif channels in ['features','properties']:
            plot_channels = np.asarray(self.channels_features_only).astype(int)
        elif isinstance(channels, list) or isinstance(channels,np.ndarray):
            if isinstance(channels[0], str):
                plot_channels = []
                for i, ch in enumerate(self.channels):
                    if any([name in ch for name in channels]):
                        plot_channels.append(i)
                plot_channels = np.array(plot_channels).astype(int)
            elif isinstance(channels[0], int):
                plot_channels = np.asarray(channels).astype(int)
        elif isinstance(channels, int):
            plot_channels = np.array([channels])
        else:
            print('invalid input for argument \'channels\'... plotting all channels.')
            plot_channels = np.arange(len(self.channels))
        
        if len(plot_channels) == 0 and ['cumulative'] in self.channels:
            plot_channels = np.array([self.channels.index(['cumulative'])]).astype(int)        

        if cumulative:
            grids = self.values[gridnum, plot_channels]
            grids = [torch.amax(grids, dim=0)]
        else:
            grids = self.values[gridnum, plot_channels]
        
        plot_channel_names = [self.channels[k][0] for k in plot_channels]
        colors0 = list(copy(plotly.colors.qualitative.Pastel))[:-1]
        if colors is None:
            colors = plotly.colors.qualitative.Dark24
            colors = [ATOM_COLORS[name] if name in ATOM_COLORS.keys() else colors0.pop(-1) for name in plot_channel_names]
            # print('colors', colors)
        if isinstance(colors, dict):
            colors = [colors[self.channels[k][0]] for k in plot_channels]
        else:
            if isinstance(colors,str):
                colors = [colors]
                colors = colors + plotly.colors.qualitative.Dark2

        
        gridmaxs = np.asarray([grid.max().detach().numpy() for grid in grids])
        gridmins = np.asarray([grid.min().detach().numpy() for grid in grids])
        gridamps = (gridmaxs-gridmins).round(1)

        if isomin is None:
            isomin = 0
        if isomax is None:
            isomax = (1-gridmaxs.max()*0.4)
        for pi, i in enumerate(np.argsort(gridamps)):
            level=(1-isomax)

            if gridmaxs[i] < level:
                if len(go_volumes) > 0 or pi < len(grids)-1:
                    continue
            dat = grids[i].clone().detach().numpy()    
            if normalize:
                dat = dat-gridmins.min()
                
            dat = dat
            opac=opacity
            name = str(channel_plot_names[plot_channels[i]])
            if cumulative:
                name = 'cumulative'
                color = colors[0]
            else:
                color = colors[i]
            
            if brightness != 0:
                color = adjust_rgb_brightness(color, brightness)

            try:
                verts, faces, normals, values = marching_cubes(dat, 
                                                            level=level, 
                                                            spacing=(1.0, 1.0, 1.0), 
                                                            gradient_direction='descent', 
                                                            step_size=1,
                                                            allow_degenerate=False, 
                                                            method='lewiner', 
                                                            mask=None)
            except ValueError:
                continue
            
            
            if plot_surface and surface_fill==1:
                
                new_volume = go.Mesh3d(
                x=verts[:,0].flatten(),
                y=verts[:,1].flatten(),
                z=verts[:,2].flatten(),
                color=color,
                i = faces[:,0].flatten(),
                j = faces[:,1].flatten(),
                k = faces[:,2].flatten(),
                opacity=opac,
                name=name,
                showlegend=showlegend,
                **kwargs
                )
                go_volumes.append(new_volume)
                
            if plot_lines or surface_fill<1:
                if surface_fill<1 and surface_fill>0 and linewidth==1:
                    linewidth = surface_fill * 20
                if linecolors is not None:
                    color = linecolors[i]
                Xe,Ye,Ze = get_mesh_lines(verts,faces)
                new_volume = go.Scatter3d(
                                   x=Xe,
                                   y=Ye,
                                   z=Ze,
                                   mode='lines',
                                   name=name,
                                   line=dict(color= color, width=linewidth),
                                   opacity=opac,
                                   showlegend=showlegend,
                                   **kwargs
                )
                go_volumes.append(new_volume)
                

            go_volumes.append(new_volume)

        fig = go.Figure(data=go_volumes)
        
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False,)


                    
        fig.update_scenes(
            camera = {
                    'center': { 'x': 0, 'y': 0, 'z': 0 }, 
                    'eye': { 'x': .7/zoom, 'y': 0, 'z': 0.4/zoom },  # for easy
                        })

        fig.update_layout(scene_aspectmode='manual',
                          scene_aspectratio=dict(x=1, y=1, z=1))
        
        fig.update_layout(width=figsize[0], height=figsize[1])
        fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})
        fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))

        fig.update_layout(scene=dict(
            xaxis=dict(nticks=4, range=[0, len(grids[-1])],),
            yaxis=dict(nticks=4, range=[0, len(grids[-1])],),
            zaxis=dict(nticks=4, range=[0, len(grids[-1])],),
                )
        )
                
        if not showaxes:
            fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        
        if showlegend:
            fig = make_legend_names_unique(fig)
        
        return fig


    def plot3d_points(self, 
                       gridnum=0, 
                       grid_coords=True, 
                       cumulative=False, 
                       show_full_grid=True, 
                       colors=None, 
                       label='', 
                       size=20,
                       channels=None,
                       opacity=1,
                       showlegend=False,
                       showaxes=False,
                       figsize=(1000,1000),
                       zoom=1,
                       brightness=0,
                       **kwargs):
        '''Plots atomic coordinates and types within grid as 3D scatterplot.'''
        
        ATOM_COLORS = self.ATOM_COLORS        
        
        if channels is None:
            channels = np.arange(len(self.channels)).tolist()
        
        size0 = copy(size)
        if self.atom_coords is None or self.atom_types is None:
            return None
        
        colors0 = list(copy(plotly.colors.qualitative.Pastel))[:-1]
        if colors is None:
            colors = plotly.colors.qualitative.Dark24
            colors = [ATOM_COLORS[t[0]] if t[0] in ATOM_COLORS.keys() else colors0.pop(-1) for t in self.channels]
            if cumulative:
                colors.insert(0,'black')
        if isinstance(colors, dict):
            colors = [colors[self.channels[_][0]] for _ in channels]
        else:
            if isinstance(colors,str):
                colors = [colors]
                colors = colors + plotly.colors.qualitative.Dark2
            elif isinstance(colors,list):
                colors = colors

       
        dat = []
        if grid_coords:
            coords = self.atom_coords_grid[gridnum]
        else:
            coords = self.atom_coords[gridnum]
            
        for i, (tlist, coord) in enumerate(zip(self.atom_channels[gridnum], coords.reshape(-1,3))):
            if len(tlist) == 0 or any(np.isnan(coord)):
                continue

            for t in tlist:
                if int(t) not in channels:
                    continue
                ci = channels.index(t)
                atype = self.channels[t][0]
                size = copy(size0)
                if cumulative:
                    color = colors[0]
                else:
                    if atype not in ATOM_COLORS.keys():
                        ATOM_COLORS[atype] = colors[ci]
                    color = ATOM_COLORS[atype]

                if brightness != 0:
                    color = adjust_rgb_brightness(color, brightness)

                if atype == 'H' or atype == 1:
                    size = size/2

                if label is not None:
                    atype = '{} {}'.format(atype,label)
                dat.append([atype] + coord.tolist() + [size, color])
                
        if len(dat) == 0:
            return
        df = pd.DataFrame(
            dat, columns=['type', 'x', 'y', 'z', 'size', 'color']).sort_values('size')
        
        
        fig = go.Figure()
        dat = df
        fig.add_trace(go.Scatter3d(mode='markers',
                                    x=dat['x'],
                                    y=dat['y'],
                                    z=dat['z'],
                                    marker=dict(
                                        color=dat['color'],
                                        size=dat['size'],
                                        sizemode='diameter',
                                        opacity=opacity
                                    ),
                                    showlegend=showlegend,
                                   **kwargs
                                  )
                     )
        
        fig.update_layout(scene_xaxis_showticklabels=False,
                          scene_yaxis_showticklabels=False,
                          scene_zaxis_showticklabels=False)

        if show_full_grid:
            x, y, z = self.get_mgrid()
            fig.update_layout(
                scene=dict(
                    xaxis=dict(nticks=4, range=[x.min(), x.max()],),
                    yaxis=dict(nticks=4, range=[y.min(), y.max()],),
                    zaxis=dict(nticks=4, range=[z.min(), z.max()],),
                ))

        fig.update_scenes(
            camera = {
                    'center': { 'x': 0, 'y': 0, 'z': 0 }, 
                    'eye': { 'x': .7/zoom, 'y': 0, 'z': 0.4/zoom }, 
                        })
        
        fig.update_layout(scene_aspectmode='manual',
                          scene_aspectratio=dict(x=1, y=1, z=1))
                          
        fig.update_layout(width=figsize[0], height=figsize[1])
        fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})
        fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
        
        if not showaxes:
            fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        
        if showlegend:
            fig = make_legend_names_unique(fig)

        return fig
                        
            
class GridEncoder(DensityGrid):
    def __init__(self,
                 grid_size : float,
                 resolution : float = 1,
                 values : torch.Tensor = None,
                 channels : list = ['default'],
                 center : tuple = (0, 0, 0),
                 n_batch : int = 1,
                 device : str = 'cpu',
                 atomtyper : BaseAtomTyper = None
                 ):
        """_summary_

        Args:
            grid_size (float): _description_
            resolution (float, optional): _description_. Defaults to 1.
            values (torch.Tensor, optional): _description_. Defaults to None.
            channels (list, optional): _description_. Defaults to ['default'].
            center (tuple, optional): _description_. Defaults to (0, 0, 0).
            n_batch (int, optional): _description_. Defaults to 1.
            device (str, optional): _description_. Defaults to 'cpu'.
            atomtyper (BaseAtomTyper, optional): _description_. Defaults to None.
        """                 

        if atomtyper is None:
            atomtyper =  AtomTyperDefault()
        self.atomtyper = atomtyper

        self.n_batch = n_batch
        self.grid_size = grid_size
        self.values = values
        self.channels = channels
        self.resolution = resolution
        self.center = center
        self.coords_enc=None
        self.init_grid()
        self.device = device
        
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        self._device = value

    @property
    def center(self):
        '''Center of atomic coordinates.'''
        return self._center
    
    @center.setter
    def center(self, values):
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=float)
        self._center = values  
    
    @property
    def atom_coords(self):
        '''Atomic coordinates (angstroms).'''
        return self._atom_coords
    
    @atom_coords.setter
    def atom_coords(self, values):
        '''Tensor of atom coordinates (batches x atoms x 3).'''
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=float)

        if values.shape[-1] != 3:
            shape = list(values.shape) + [3]
            shape[-2] = -1
            values.reshape(*shape)
        if values.ndim == 2:
            values = values.reshape(-1, *values.shape)
        self._atom_coords = values
        
    @property
    def atom_types(self):
        '''Atomic types array.'''
        return self._atom_types
    
    @atom_types.setter
    def atom_types(self, values):
        '''Tensor of 1hot vectors of atom types (batches x atoms x 1024).'''
        if isinstance(values, list):
            values = self.atomtyper.nums2vec(values)
        elif isinstance(values, np.ndarray):
            values = torch.from_numpy(values).int()

        if values.ndim == 1:
            values = values.reshape(1,-1,1)
        if values.ndim == 2:
            values = values.reshape(1,*values.shape)

        if values.int().max() > 1:
            # convert to 1-hot vector
            values = self.atomtyper.nums2vec(values)
            
        self._atom_types = values
        
        
    @property
    def channels(self):
        '''Channel labels.'''
        return self._channels
    
    @channels.setter
    def channels(self, values):
        '''List of channel type indices that define grid channels.'''
        chs = []
        for cgroup in values:
            if not isinstance(cgroup, list):
                cgroup = [cgroup]
            
            cnew = []
            for i,c in enumerate(cgroup):
                c = [c]
                cnew += c
            chs.append(cnew)
                    
        self._channels = chs
        
    
    @property
    def atom_channels(self):
        '''Channel indices of each atom.'''
        vals = []
        for i in range(self.n_batch):
            vals.append([self.channel_1hot2list(x) for x in self.atom_channels_1hot[i]])
        return vals

    @property
    def atom_channels_flat(self):
        '''Channel indices of each atom, Flattened '''
        vals = []
        for i in range(self.n_batch):
            vals.append(torch.concat([self.channel_1hot2list(x) for x in self.atom_channels_1hot[i]]))
        return vals
    
    @property
    def atom_channels_1hot(self):
        '''1hot vector of channels for each atom.'''
        vals = torch.zeros((*self.atom_types.shape[:2],len(self.channels)), dtype=bool, device=self.device)
        for i, tnums in enumerate(self.channel_nums):
            try:
                vals[:,:,i] = self.atom_types[:,:,tnums].any(axis=-1)
            except:
                # print('i',i, 'tnums',tnums, 'vals',vals.shape, 'atypes',self.atom_types.s)
                pass
        return vals
    
    @property
    def bonds(self):
        '''Tensor of atom index pairs that each bond connects (batch x bonds x 2).'''
        return self._bonds
    
    @bonds.setter
    def bonds(self, values):
        self._bonds = values
    
    @property
    def bond_types(self):
        '''Tensor of 1hot vectors of bond types (batches x bonds x 1024).'''
        return self._bond_types
    
    @bond_types.setter
    def bond_types(self, values):
        if isinstance(values, list):
            values = torch.tensor(values, dtype=int)
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)
        if values.int().max() > 1:
            if values.ndim == 1:
                values = values.reshape(1,-1)
            values = self.atomtyper.nums2vec(values)
        
        if values.ndim == 2:
            values = values.reshape(1,*values.shape)
        self._bond_types = values
    
    @property
    def bond_coords(self):
        '''Tensor of bond coordinates (batches x bonds x 3).'''
        return self._bond_coords
    
    @bond_coords.setter
    def bond_coords(self, values):
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=float)
        if values.ndim == 2:
            values = values.reshape(-1, *values.shape)
        self._bond_coords = values

    @property
    def bond_coords_grid(self):
        '''Converts stored bond voxel coordinates to grid coordinates.'''
        return self.bond_coords / self.resolution + self.center_grid - self.grid_offset
    
    @property
    def bond_channels(self):
        vals = []
        for i in range(self.n_batch):
            vals.append([self.channel_1hot2list(x) for x in self.bond_channels_1hot[i]])
        return vals
    
    
    @property
    def bond_channels_1hot(self):
        vals = torch.zeros((*self.bond_types.shape[:2],len(self.channels)), dtype=bool)
        for i, tnums in enumerate(self.channel_nums):
            vals[:,:,i] = self.bond_types[:,:,tnums].any(axis=-1)
        return vals
    
    @property
    def channel_nums(self):
        channels = self.channels
        channel_nums = []
        for i in range(len(self.channels)):
            channel_nums.append([])
            for type_name in channels[i]:
                if type_name in self.atomtyper.group_names:
                    for t in self.atomtyper.groups[type_name]:
                        channel_nums[i].append(self.atomtyper.type2num(t))
                else:
                    channel_nums[i].append(self.atomtyper.type2num(type_name))
        return channel_nums

    @property
    def channel_colors(self):
        '''Default colors to use for plotting channels.'''
        return [self.ATOM_COLORS.get(ch[0], self.ATOM_COLORS['default']) for ch in self.channels]
    
        
    @property
    def channels_elements_only(self):
        return torch.arange(len(self.channels))[[self.is_element_channel(x) for x in self.channels]]
    
    @property
    def channels_features_only(self):
        return torch.arange(len(self.channels))[[self.is_feature_channel(x) for x in self.channels]]
    
    @property
    def channels_bonds_only(self):
        return torch.arange(len(self.channels))[[self.is_bond_channel(x) for x in self.channels]]

    def init_default_atom(self):
        '''Initialize atom_coords and atom_types for single atom located at center of grid.'''
        self.atom_coords = torch.zeros((1,1,3))
        self.atom_types = self.atomtyper.types2vec(['default']).unsqueeze(0)

    
    def channel_1hot2list(self, vec):
        '''Converts a 1hot channel vector to a list of channel indices.'''
        return torch.arange(len(self.channels), dtype=int)[vec]
    
    def is_element_channel(self, channel):
        if not isinstance(channel, Iterable):
            channel = [channel]
        return any([self.atomtyper.is_element(c) for c in channel])
    
    
    def is_feature_channel(self, channel):
        if not isinstance(channel, Iterable):
            channel = [channel]
        return any([self.atomtyper.is_atom_feature(c) for c in channel])
    
    
    def is_bond_channel(self, channel):
        if not isinstance(channel, Iterable):
            channel = [channel]
        return any([self.atomtyper.is_bond(c) for c in channel])
    
    def get_channels_with_types(self, nums):
        '''Returns indices for channels with given type indices.'''
        chans = []
        for n in nums:
            ch = self.get_type_channel(n)
            if ch is not None:
                chans.append(ch)
        chans = set(chans)
        if None in chans:
            chans.remove(None)
        return sorted(list(chans))
    

    def get_channels_in_range(self, tmin, tmax):
            '''Returns all channels that are in a specified range of type indices.'''
            chans = []
            for i, ch in enumerate([np.array([self.atomtyper.type2num(x) for x in clist]) for clist in self.channels]):
                if np.logical_and(ch >= tmin, ch <= tmax).any():
                    chans.append(i)
            return chans
        

    def get_1hot(self, atom_types):
        '''
        converts numeric types to 1hot vector in test example (not for general use)
        '''
        assert len(self.channels) > 0, 'Must set self.channels before calling init_types_1hot.'
        atom_1types = np.zeros((atom_types.shape[0], atom_types.shape[1] ,len(self.channels)), dtype=bool)
        for g in range(len(atom_types)):
            for i, typ in enumerate(atom_types[g]):
                if typ == 0:
                    continue
                typ = self.get_type_channel(typ)
                if typ is not None:
                    atom_1types[g, i, typ] = 1
        
        return atom_1types
    
    
    def element_nums2vec(self, element_nums):
        # change this to accept an (n x n_types) vector
        ''' Takes a vector of atomic numbers (n x 1) and converts it to a 1hot encoded types vector (n x ntypes)
        '''
        if element_nums.ndim == 1:
            tvec = torch.zeros((element_nums.shape[0], self.atomtyper.ntypes), dtype=bool)
            for i, t in enumerate(element_nums):
                tvec[i] = self.atomtyper.types2vec([int(t)])
        elif element_nums.ndim == 2:
            tvec = torch.zeros((element_nums.shape[0], element_nums.shape[1], self.atomtyper.ntypes))
            for b in range(len(element_nums)):
                for i, t in enumerate(element_nums[b]):
                    if t == 0:
                        continue
                    tvec[b, i] = self.atomtyper.types2vec([int(t)])
        else:
            raise Exception(f'Invalid input shape. element_nums.shape = {element_nums.shape}')
            
        return tvec
        


    def bond_nums2vec(self, bond_nums : Union[torch.Tensor, np.ndarray]):
        """        
        Takes a vector of atomic numbers (n x 1) and converts it to a 1hot encoded types vector (n x ntypes)
           

        Args:
            bond_nums (torch.Tensor): bond numbers

        Returns:
            torch.Tensor: 1hot encoded bond type vector
        """ 
        if bond_nums.ndim == 1:
            tvec = torch.zeros((bond_nums.shape[0], self.atomtyper.ntypes), dtype=bool)
            for i, t in enumerate(bond_nums):
                tvec[i] = self.atomtyper.nums2vec([int(t)])
        elif bond_nums.ndim == 2:
            tvec = torch.zeros((bond_nums.shape[0], bond_nums.shape[1], self.atomtyper.ntypes))
            for b in range(len(bond_nums)):
                for i, t in enumerate(bond_nums[b]):
                    if t == 0:
                        continue
                    tvec[b, i] = self.atomtyper.nums2vec([int(t)])
        else:
            raise Exception(f'Invalid input shape. bond_nums.shape = {bond_nums.shape}')
            
        return tvec
    
    
    def update_bonds(self, bonds, atom_coords=None):
        '''Updates bond information based on a set of bonds and optional atom coordinates.'''
        if bonds.ndim != 3:
            bonds = bonds.reshape(1,-1,3)
        self.bonds = bonds
        self.bond_coords = torch.zeros(bonds.shape)
        self.bond_coords[:] = torch.nan
        
                
        if atom_coords is None:
            atom_coords = self.atom_coords
        
        if isinstance(atom_coords, np.ndarray):
            atom_coords = torch.from_numpy(atom_coords)
            
        elemtypes = self.atom_channels_1hot[:,:,self.channels_elements_only].any(axis=-1)
        
        for mi in range(self.n_batch):
            for i in range(len(bonds[mi])):
                a0 = int(bonds[mi,i,0])
                a1 = int(bonds[mi,i,1])
                b_order = bonds[mi,i,2]
                if b_order != 0 and elemtypes[mi,a0] and elemtypes[mi,a1]:
                    self.bond_coords[mi,i] = (atom_coords[mi,a0] + atom_coords[mi,a1])/2
                        
    
    def encode_bonds(self, bonds=None, bond_types=None, atom_coords=None, encoding='softcube'):
        '''Encodes a set of bonds to the grid.'''
        if atom_coords is None:
            atom_coords = self.atom_coords
        
        if bond_types is None:
            bond_types = self.bond_types
        
        if bonds is None:
            bonds = self.bonds
        
        self.update_bonds(bonds, atom_coords=atom_coords)
        for i in range(self.n_batch):
            self.values[i] = self.encode_grid_soft(grid_coords=self.bond_coords_grid[i],
                                  grid=self.values[i],
                                  atom_types= self.bond_channels_1hot[i],
                                  encoding=encoding)
            
    def to_device(self, device):
        device_update_list = ['values', 'center', 'resolution','grid_size', 
                              'atom_coords', 'atom_types', 'n_batch', 'coords_enc']
        
        for name in device_update_list:
            if hasattr(self, name) and getattr(self,name) is not None:
                if not isinstance(getattr(self, name), torch.Tensor):
                    setattr(self, name, torch.tensor(getattr(self,name), device=device))
                else:
                    setattr(self, name, getattr(self,name).to(device))
        self.device = device


    def encode_coords2grid(self, coords=None, types=None, encoding='softcube'):
        '''Encodes a set of atom coordinates and types to the grid.'''
        if coords is not None:
            self.atom_coords = coords
        if types is not None:
            self.atom_types = types
        
        if not hasattr(self,'values'):
            self.init_grid()
            
        if len(self.atom_coords) > len(self.values):
            self.n_batch = len(self.atom_coords)
        
        if 'soft' in encoding:
            for i in range(len(self.atom_coords)):
                self.values[i] = 0
                self.encode_grid_soft(grid_coords=self.atom_coords_grid[i],
                                        grid=self.values[i],
                                        atom_types=self.atom_channels_1hot[i],
                                        encoding=encoding)
                
        elif encoding == 'hard':
            for i in range(len(self.atom_coords)):
                self.values[i] = 0
                self.encode_grid_hard(grid_coords=self.atom_coords_grid[i], 
                                        grid=self.values[i], 
                                        atom_types=self.atom_channels_1hot[i])
        else:
            raise Exception(f'Invalid encoding: {encoding}')

    def valid_coord_type_idxs(self, grid_coords, channels_1hot, in_bounds_margin=1, out_bounds_warn=False):        
        '''Returns a set of valid indices for input coordinates/types'''
        idxs_valid = self.valid_coord_idxs(grid_coords, in_bounds_margin, out_bounds_warn)
        idxs_valid = torch.logical_and(idxs_valid, self.valid_type_idxs(channels_1hot))
        return idxs_valid

    def valid_coord_idxs(self, grid_coords, in_bounds_margin=1, out_bounds_warn=False):
        '''Returns a set of valid indices for input coordinates'''
        idxs_valid = self.is_in_bounds(grid_coords, pad=in_bounds_margin)
        idxs_valid = torch.logical_and(idxs_valid, ~torch.isnan(grid_coords).any(dim=-1))
        if not idxs_valid.all() and out_bounds_warn:
            warnings.warn('Warning: Not all atoms fit in grid! Out of bounds atoms will be excluded...')
        return idxs_valid

    def valid_type_idxs(self, channels_1hot):
        '''Returns a set of valid indices for input channels'''
        idxs_valid = ~(channels_1hot.sum(dim=-1) == 0)
        return idxs_valid
    

    def get_encode_weights(self, grid_coords, grid, encoding='softcube', atom_types=None):
        '''Generates the encoded voxel values for an atom at a particular grid location.'''
        eps = 1e-8

        if not hasattr(self, 'coordinate_encoder'):
            self.init_coordinate_encoder(encoding)

        if grid.ndim == 3:
            grid = grid.reshape(-1,*grid.shape)
        
        if atom_types is None:
            idxs_valid = self.valid_coord_idxs(grid_coords, in_bounds_margin=1, out_bounds_warn=False)
        else:
            idxs_valid = self.valid_coord_type_idxs(grid_coords, atom_types, in_bounds_margin=1, out_bounds_warn=False)
        grid_coords = grid_coords[idxs_valid]
        
        if atom_types is not None:
            atom_types = atom_types[idxs_valid] 
            grid_coords, atom_types = self.flatten_coords_types(grid_coords, atom_types) # rearrange coord/types lists to have one entry per atom type

        c = grid_coords
        cround = c.round().long()
        dr = c - cround + eps
        weights = self.coordinate_encoder.encode_func(dr).reshape(len(c),3,3,3)
        return weights, cround
    
    def encode_grid_soft(self, grid_coords, grid, encoding='soft', atom_types=None):
        '''Encodes a set of atom coordinates into grid values using a hard encoding, where each atom is represented by multiple voxel values.'''
        eps = 1e-8
        dims = 3
        
        if not hasattr(self, 'coordinate_encoder'):
            self.init_coordinate_encoder(encoding)

        if atom_types is None:
            atom_types = torch.ones((len(grid_coords), 1), dtype=int)
        if grid.ndim == 3:
            grid = grid.reshape(-1,*grid.shape)

        idxs_valid = self.valid_coord_type_idxs(grid_coords, atom_types, in_bounds_margin=1, out_bounds_warn=False)
        if len(idxs_valid) == 0:
            print('All atoms invalid. Check GridEncoder channels / atom types. Also check that atom coordinates are inside of grid.')
        grid_coords, atom_types = grid_coords[idxs_valid] , atom_types[idxs_valid] 
        grid_coords, atom_types = self.flatten_coords_types(grid_coords, atom_types) # rearrange coord/types lists to have one entry per atom type


        c = grid_coords
        cround = c.round().long()
        dr = c - cround + eps
        weights = self.coordinate_encoder.encode_func(dr).reshape(len(c),3,3,3)
        
        gshape = torch.tensor(grid.shape)
        numels = torch.stack([torch.prod(gshape[i+1:]) for i in range(grid.ndim)]).to(self.device) # number of elements in each subtensor
        
        vcoords = cround.unsqueeze(1) + self.coordinate_encoder.coords0       
        
        vcoords = torch.concat((atom_types.repeat_interleave(27, dim=1).unsqueeze(-1), vcoords), dim=-1) # concatenate type index with coordinate index
        idxs_flat = torch.mul(numels, vcoords).sum(dim=-1).flatten() # get flattened indices
        grid.view(grid.numel())[idxs_flat] += weights.flatten() # assign weights to grid
        return grid

    def flatten_coords_types(self, coords, types):
        '''Returns a list of coordinates and types with a separate entry for each atom type. Atoms with multiple types will have duplicate entries in coordinates.'''
        # print(types.nonzero())
        device = types.device
        if not isinstance(types, list):
            num_types = types.sum(dim=-1)
            if num_types.max() == 1:
                types = types.nonzero()[:,1].reshape(-1,1)
                return coords, types
            else:
                types = [x.nonzero() for x in types] # convert 1hot vec to indices


        num_types = torch.tensor([len(x) for x in types], device=device)
        
        newcoords = torch.zeros((num_types.sum(), 3), device=device)
        newtypes = torch.zeros((num_types.sum(), 1), dtype=int, device=device)
        n = 0
        for i in range(len(num_types)):
            for j in range(num_types[i]):
                newcoords[n] = coords[i]
                newtypes[n] = types[i][j]
                n += 1
        return newcoords, newtypes

    
    def encode_grid_hard(self, grid_coords, grid, atom_types=None):
        '''Encodes a set of atom coordinates into grid values using a hard encoding, where each atom is represented by a single voxel value.'''
        if atom_types is None:
            atom_types = torch.zeros(len(grid_coords), dtype=int)
        if grid.ndim == 3:
            grid = grid.reshape(-1,*grid.shape)
        for c, t in zip(grid_coords, atom_types):
            if any(torch.isnan(c)) or sum(t)==0 or not self.is_in_bounds(c, pad=0):
                continue
            c = c.round().int()
            try:
                grid[t, c[0],c[1],c[2]] += 1
            except Exception as e:
                print('grid', grid.values.shape)
                print('t',t, 'c', c)
                raise e
        return grid

    
    def decode_grid_soft(self, grid, grid_coords, atom_types, tight_box=True, correct_overlap=False):
        '''Decodes grid values into atomic coordinates for a particular grid location.'''
        dims = 3
        if self.coords_enc is None:
            mesh = np.meshgrid(*[[-1,0,1]]*dims, indexing='ij')
            self.coords_enc = np.stack(mesh, axis=-1).reshape(-1,dims)
            self.coords_enc = torch.tensor(self.coords_enc, dtype=float)
            if self.device is not None:
                self.coords_enc = self.coords_enc.to(self.device)
                self.coordinate_encoder.coords0.to(self.device)

        grid_fit = grid.clone()

        # get overlapping coordinates and counts
        if correct_overlap:
            cenc = self.coords_enc.clone()
            cenc = cenc[(cenc>=0).all(dim=1)]

            coords0 = grid_coords[atom_types.any(axis=-1)].reshape(-1,3)
            types0 = atom_types[atom_types.any(axis=-1)].reshape(-1,atom_types.shape[-1])

            vx_coords = torch.tile(coords0.floor(), (len(cenc),1)) + cenc.repeat_interleave(len(coords0), dim=0)
            vx_types = torch.tile(types0, (len(cenc),1))

            vxs = torch.cat([vx_types, vx_coords], dim=1).int()
            d = Counter([tuple(x.tolist()) for x in vxs])

            # correct overlapping grid values
            for c, v in d.items():
                if v > 1:
                    print(c, v)
                    grid_fit[c] /= v

        # decode coordinates
        newcoords = grid_coords.clone()
        newcoords[:] = torch.nan
        for i, (c,t) in enumerate(zip(grid_coords, atom_types)):
            tlist = self.channel_1hot2list(t)
            if any(torch.isnan(c)) or len(tlist)==0:
                continue
            t = np.random.choice(tlist) # randomly choose the channel to decode
            cround = c.round().int()
            dr = c-cround
            gvals = grid_fit[t,
                     int(cround[0]-1):int(cround[0]+2), 
                     int(cround[1]-1):int(cround[1]+2), 
                     int(cround[2]-1):int(cround[2]+2)]

            gvals = gvals.flatten()
            
            if tight_box: 
                # only pass voxels bordering atom to decode_func
                dists = (self.coords_enc - dr).abs()
                idxs = (dists < 1).all(axis=1) # grid voxels bordering atom
                gvals = gvals[idxs]
                coords = self.coords_enc[idxs]
            newcoords[i] = self.coordinate_encoder.decode_func(gvals, coords) + cround
        return newcoords
    
    
    def decode_grid2coords(self, 
                           grid=None, 
                           atom_coords=None, 
                           atom_types=None, 
                           encoding='soft', 
                           tight_box=True, 
                           correct_overlap=False):
        '''Decodes grid values into a set of atomic coordinates from a set of grid coordinates.'''
        if grid is not None:
            self.values = grid
        if atom_coords is not None:
            self.atom_coords = atom_coords
        if atom_types is not None:
            self.atom_types = atom_types
        
        if len(self.values) > self.n_batch:
            self.n_batch = len(self.atom_coords)
        
        grid = self.values
        grid_coords = self.atom_coords_grid
        atom_types = self.atom_channels_1hot
        
        pred_coords = grid_coords.clone()
        pred_coords[:] = torch.nan
        if 'soft' in encoding:
            for i in range(len(grid_coords)):
                pred_coords[i] = self.decode_grid_soft(grid=grid[i],
                                                      grid_coords=grid_coords[i],
                                                      atom_types=atom_types[i],
                                                      tight_box=tight_box,
                                                      correct_overlap=correct_overlap)
        return pred_coords
    
    
    
    def add_encoding_noise(self, noise_p=0.9, max_val=0.3):
        '''
        noise_p: Probability of adding noise to any given atom
        max_val: Maximum value of noise in any voxel. Value drawn from uniform distribution [0,max_val)
        
        '''
            
        noise_voxels_p = 10**(-np.arange(1,5, dtype=float))
        noise_voxels_p /= noise_voxels_p.sum()

        for i in range(len(self.values)):            
            grid_coords = self.atom_coords_grid[i]
            atom_types = self.atom_channels[i]
            for c,t in zip(grid_coords, atom_types):
                if any(torch.isnan(c)) or sum(t) == 0:
                    continue
                if np.random.random() < noise_p:
                    n_noise_voxels = np.random.choice(np.arange(1, len(noise_voxels_p)+1), p=noise_voxels_p)
#                     t = np.random.choice(np.arange(len(t))[t])  # add noise to only some types
                    self.add_atom_encoding_noise(i,c,t, n_noise_voxels, max_val=max_val)
                    
                        
    def add_atom_encoding_noise(self, gridnum, atom_coord, atom_type, n_noise_voxels, max_val=0.3):
        '''
        n_noise_voxels: # of surrounding voxels to add noise to
        max_val: Maximum value of noise in any voxel. Value drawn from uniform distribution [0,max_val)
        
        '''
        if not hasattr(self, 'ebox_border'):
            self.ebox_border = self.get_ebox_border()
            
        c, tlist = atom_coord, atom_type
        t = np.random.choice(tlist)
        noise_idxs = np.arange(len(self.ebox_border))
        cfloor = c.floor().int()
        np.random.shuffle(noise_idxs)
        noise_coords = self.ebox_border[noise_idxs] + cfloor
        ni = 0
        for cn in noise_coords:
            if ni >= n_noise_voxels:
                break
            if self.values[gridnum,t,cn[0],cn[1],cn[2]] == 0:  # check this is not a voxel belonging to a neighboring atom
                value = np.random.random()*max_val
                self.values[gridnum,t,cn[0],cn[1],cn[2]] += value
                ni += 1

                
    def get_ebox(self, pad_width=0):
        '''returns voxel coordinates of encoding box relative to the lower left encoding (floor(coordinates))'''
        x = torch.arange(0, int(2+2*pad_width))
        grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
        ebox = torch.stack([grid_x, grid_y, grid_z], dim=-1) - pad_width
        return ebox

    
    def get_ebox_border(self):
        '''returns voxel coordinates of the voxels surrounding the encoding box relative to the lower left encoding (floor(coordinates))'''
        ebox = self.get_ebox(pad_width=1)
        ebox_inner = ebox[1:-1,1:-1,1:-1].reshape(-1,3)
        ebox_outer = torch.cat([ebox[:,:,::len(ebox)-1].reshape(-1,3),
                                ebox[:,::len(ebox)-1,:].reshape(-1,3),
                                ebox[::len(ebox)-1,:,:].reshape(-1,3),
                               ], dim=0)
        ebox_outer = torch.unique(ebox_outer, dim=0).reshape(-1,3)
        return ebox_outer



class CoordinateEncoderCube(object):
    '''encodes atom coordinates to grid based on volume of cube within each voxel.'''


    def __init__(self) -> None:
        self.coords0 = torch.stack(torch.meshgrid(*[torch.arange(3)-1]*3, indexing='ij'), axis=-1).reshape(-1,3).unsqueeze(0)
        self.device = 'cpu'

    def to_device(self, device):
        self.device = device
        self.coords0 = self.coords0.to(device)

    def encode_func(self, dr, verbose=False):
        '''encodes coordinates into a cube of voxel values.'''
        r = 0.5
        if not isinstance(dr, torch.Tensor):
            dr = torch.tensor(dr, dtype=float, device=self.device)

        if dr.numel() > 3:
            dr = dr.reshape(-1,3)
        
        dr = dr.unsqueeze(1)
        N = len(dr)
        w = torch.zeros((N,27), device=self.device).float()

        cdiff = self.coords0 - dr
        inz = (cdiff.abs() <= 2*r).all(dim=-1) # include only voxels with overlap
        wnz = 1 - torch.abs(cdiff[inz]) # get sides lengths of cube volume within each voxel
        wnz = torch.prod(wnz, dim=-1) # multiply side lengths to get volume within each coordinate voxel
        w[inz] = wnz.float()
        return w.reshape(N,-1,1)

    def decode_func(self, weights, coords=None):
        '''decodes a cube of voxel values into a set of coordinates.'''
        weights = weights.reshape(-1,1)
        if coords is None:
            if len(weights) == 27:
                coords = self.coords0
            if len(weights) == 8:
                # assume upper right corner
                coords = self.coords0.reshape(3,3,3)[1:,1:,1:].reshape(-1,3)
                
        wcoords = coords.reshape(-1,3) * weights
        return wcoords.sum(axis=0)


            