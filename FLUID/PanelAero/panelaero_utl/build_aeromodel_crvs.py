import numpy as np


class AeroGridFromCurves:
    """
    Build an aerodynamic grid from leading edge and trailing edge curves.
    Compatible with the aerogrid format from build_aeromodel.py
    """
    
    def __init__(self, le_curve_func, te_curve_func, n_span, n_chord):

        self.le_curve_func = le_curve_func
        self.te_curve_func = te_curve_func
        self.n_span = n_span
        self.n_chord = n_chord
        self.aerogrid = None
    
    def build_aerogrid(self, eid_start=1000, cp=0):

        # Create corner points grid
        caero_grid, caero_panels = self._create_grid_and_panels(eid_start, cp)
        
        # Build aerogrid properties (same as original code)
        ID = []
        l = []  # length of panel
        A = []  # area of one panel
        N = []  # unit normal vector
        offset_l = []  # 25% point l
        offset_k = []  # 50% point k
        offset_j = []  # 75% downwash control point j
        offset_P1 = []  # Vortex point at 25% chord, 0% span
        offset_P3 = []  # Vortex point at 25% chord, 100% span
        r = []  # vector P1 to P3, span of panel

        for i_panel in range(len(caero_panels['ID'])):
            # Find corner point indices
            index_1 = np.where(caero_panels['cornerpoints'][i_panel][0] == caero_grid['ID'])[0][0]
            index_2 = np.where(caero_panels['cornerpoints'][i_panel][1] == caero_grid['ID'])[0][0]
            index_3 = np.where(caero_panels['cornerpoints'][i_panel][2] == caero_grid['ID'])[0][0]
            index_4 = np.where(caero_panels['cornerpoints'][i_panel][3] == caero_grid['ID'])[0][0]

            # Calculate panel edges
            l_1 = caero_grid['offset'][index_2] - caero_grid['offset'][index_1]
            l_2 = caero_grid['offset'][index_3] - caero_grid['offset'][index_4]
            b_1 = caero_grid['offset'][index_4] - caero_grid['offset'][index_1]
            b_2 = caero_grid['offset'][index_3] - caero_grid['offset'][index_2]
            l_m = (l_1 + l_2) / 2.0
            b_m = (b_1 + b_2) / 2.0

            ID.append(caero_panels['ID'][i_panel])
            l.append(np.linalg.norm(l_m))
            A.append(np.linalg.norm(np.cross(l_m, b_m)))
            N.append(np.cross(l_1, b_1) / np.linalg.norm(np.cross(l_1, b_1)))
            offset_l.append(caero_grid['offset'][index_1] + 0.25 * l_m + 0.50 * b_1)
            offset_k.append(caero_grid['offset'][index_1] + 0.50 * l_m + 0.50 * b_1)
            offset_j.append(caero_grid['offset'][index_1] + 0.75 * l_m + 0.50 * b_1)
            offset_P1.append(caero_grid['offset'][index_1] + 0.25 * l_1)
            offset_P3.append(caero_grid['offset'][index_4] + 0.25 * l_2)
            r.append((caero_grid['offset'][index_4] + 0.25 * l_2) - 
                    (caero_grid['offset'][index_1] + 0.25 * l_1))

        n = len(ID)
        set_l = np.arange(n * 6).reshape((n, 6))
        set_k = np.arange(n * 6).reshape((n, 6))
        set_j = np.arange(n * 6).reshape((n, 6))
        
        aerogrid = {
            'ID': np.array(ID),
            'l': np.array(l),
            'A': np.array(A),
            'N': np.array(N),
            'offset_l': np.array(offset_l),
            'offset_k': np.array(offset_k),
            'offset_j': np.array(offset_j),
            'offset_P1': np.array(offset_P1),
            'offset_P3': np.array(offset_P3),
            'r': np.array(r),
            'set_l': set_l,
            'set_k': set_k,
            'set_j': set_j,
            'CD': caero_panels['CD'],
            'CP': caero_panels['CP'],
            'n': n,
            'coord_desc': 'bodyfixed',
            'cornerpoint_panels': caero_panels['cornerpoints'],
            'cornerpoint_grids': np.hstack((caero_grid['ID'][:, None], caero_grid['offset']))
        }
        
        self.aerogrid = aerogrid
        return aerogrid
    
    def _create_grid_and_panels(self, eid_start, cp):
        """
        Create grid points and panel connectivity from the curves.
        """
        # Parametric discretization
        u_span = np.linspace(0.0, 1.0, self.n_span + 1)
        u_chord = np.linspace(0.0, 1.0, self.n_chord + 1)
        
        # Generate grid points
        grids = {'ID': [], 'offset': []}
        grids_map = np.zeros((self.n_chord + 1, self.n_span + 1), dtype='int')
        
        grid_ID = eid_start * 100  # Offset for grid IDs
        
        for i_span in range(self.n_span + 1):
            # Get LE and TE points at this spanwise location
            le_point = np.array(self.le_curve_func(u_span[i_span]))
            te_point = np.array(self.te_curve_func(u_span[i_span]))
            
            for i_chord in range(self.n_chord + 1):
                # Linear interpolation between LE and TE
                offset = le_point + u_chord[i_chord] * (te_point - le_point)
                
                grids['ID'].append(grid_ID)
                grids['offset'].append(offset)
                grids_map[i_chord, i_span] = grid_ID
                grid_ID += 1
        
        # Build panels from corner points
        panels = {"ID": [], 'CP': [], 'CD': [], "cornerpoints": []}
        panel_ID = eid_start
        
        for i_span in range(self.n_span):
            for i_chord in range(self.n_chord):
                panels['ID'].append(panel_ID)
                panels['CP'].append(cp)
                panels['CD'].append(cp)
                # Panel corners: 1-2-3-4 in counter-clockwise order
                panels['cornerpoints'].append([
                    grids_map[i_chord, i_span],      # corner 1
                    grids_map[i_chord + 1, i_span],  # corner 2
                    grids_map[i_chord + 1, i_span + 1],  # corner 3
                    grids_map[i_chord, i_span + 1]   # corner 4
                ])
                panel_ID += 1
        
        # Convert to numpy arrays
        panels['ID'] = np.array(panels['ID'])
        panels['CP'] = np.array(panels['CP'])
        panels['CD'] = np.array(panels['CD'])
        panels['cornerpoints'] = np.array(panels['cornerpoints'])
        grids['ID'] = np.array(grids['ID'])
        grids['offset'] = np.array(grids['offset'])
        
        return grids, panels


