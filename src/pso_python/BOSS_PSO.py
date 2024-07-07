import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from General.StringUtils import preface_char
from General.MathUtils import vsini_kernel, interp_tensor, map_range

torch.set_grad_enabled(False)



class BOSS_PSO:
    def __init__(self, verbose=True) -> None:
        '''
        observed: (N, 4) array, columns:
            0: wavelength
            1: flux
            2: variance
            3: spectral resolution
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.wl = torch.from_numpy(np.logspace(np.log10(3500), np.log10(10450), 87508)).to(device=self.device)

        self.teff_gridpoints1 = torch.from_numpy(np.arange(6000, 10100, 100)).to(device=self.device)
        self.teff_gridpoints2 = torch.from_numpy(np.concatenate([np.arange(9800, 10000, 100), np.arange(10000, 26000, 250)])).to(device=self.device)
        self.teff_gridpoints3 = torch.from_numpy(np.arange(25500, 30250, 250)).to(device=self.device)

        self.m_h_gridpoints1 = torch.from_numpy(np.arange(-0.8, 0.9, 0.1)).to(device=self.device)
        self.m_h_gridpoints2 = torch.from_numpy(np.arange(-0.8, 0.9, 0.1)).to(device=self.device)
        self.m_h_gridpoints3 = torch.from_numpy(np.arange(-0.8, 0.9, 0.1)).to(device=self.device)

        self.logg_gridpoints1 = torch.from_numpy(np.arange(2.5, 5.1, 0.1)).to(device=self.device)
        self.logg_gridpoints2 = torch.from_numpy(np.arange(3.0, 5.1, 0.1)).to(device=self.device)
        self.logg_gridpoints3 = torch.from_numpy(np.arange(3.3, 5.1, 0.1)).to(device=self.device)

        self.model_grid1 = torch.zeros(len(self.teff_gridpoints1), len(self.m_h_gridpoints1), len(self.logg_gridpoints1), 87508, device=self.device)
        self.model_grid2 = torch.zeros(len(self.teff_gridpoints2), len(self.m_h_gridpoints2), len(self.logg_gridpoints2), 87508, device=self.device)
        self.model_grid3 = torch.zeros(len(self.teff_gridpoints3), len(self.m_h_gridpoints3), len(self.logg_gridpoints3), 87508, device=self.device)

        self.global_best_model = self.physical_to_internal(torch.tensor([10_000, 0.0, 4.0, 100.0, 0.0])).to(device=self.device)
        self.gobal_best_cost = 1.0e10

        for i in (tqdm(range(len(self.teff_gridpoints1) * len(self.m_h_gridpoints1) * len(self.logg_gridpoints1)), desc='Loading grid 1')
                   if verbose else range(len(self.teff_gridpoints1) * len(self.m_h_gridpoints1) * len(self.logg_gridpoints1))):
            t = i//(len(self.m_h_gridpoints1) * len(self.logg_gridpoints1))
            m = (i//len(self.logg_gridpoints1)) % len(self.m_h_gridpoints1)
            l = i % len(self.logg_gridpoints1)
            self.model_grid1[t,m,l] = torch.from_numpy(
                np.load('boss_grid_with_continuum/' + self.gridpoint_to_filename(self.teff_gridpoints1[t], self.m_h_gridpoints1[m], self.logg_gridpoints1[l])) * 1e5
            ).to(device=self.device)
        for i in (tqdm(range(len(self.teff_gridpoints2) * len(self.m_h_gridpoints2) * len(self.logg_gridpoints2)), desc='Loading grid 2')
                   if verbose else range(len(self.teff_gridpoints2) * len(self.m_h_gridpoints2) * len(self.logg_gridpoints2))):
            t = i//(len(self.m_h_gridpoints2) * len(self.logg_gridpoints2))
            m = (i//len(self.logg_gridpoints2)) % len(self.m_h_gridpoints2)
            l = i % len(self.logg_gridpoints2)
            self.model_grid2[t,m,l] = torch.from_numpy(
                np.load('boss_grid_with_continuum/' + self.gridpoint_to_filename(self.teff_gridpoints2[t], self.m_h_gridpoints2[m], self.logg_gridpoints2[l])) * 1e5
            ).to(device=self.device)
        for i in (tqdm(range(len(self.teff_gridpoints3) * len(self.m_h_gridpoints3) * len(self.logg_gridpoints3)), desc='Loading grid 3')
                   if verbose else range(len(self.teff_gridpoints3) * len(self.m_h_gridpoints3) * len(self.logg_gridpoints3))):
            t = i//(len(self.m_h_gridpoints3) * len(self.logg_gridpoints3))
            m = (i//len(self.logg_gridpoints3)) % len(self.m_h_gridpoints3)
            l = i % len(self.logg_gridpoints3)
            self.model_grid3[t,m,l] = torch.from_numpy(
                np.load('boss_grid_with_continuum/' + self.gridpoint_to_filename(self.teff_gridpoints3[t], self.m_h_gridpoints3[m], self.logg_gridpoints3[l])) * 1e5
            ).to(device=self.device)

    def gridpoint_to_filename(self, teff:float, m_h:float, logg:float) -> str:
        result = 'l' + ('m' if m_h < -1e-3 else 'p') + '00'
        result += (f'{m_h:.2f}').split('.')[1] + '_'
        result += preface_char(str(int(np.rint(teff))), 5) + '_'
        result += preface_char((f'{logg:.1f}').replace('.', ''), 3) + '0.npy'
        return result

    def physical_to_internal(self, params:torch.Tensor) -> torch.Tensor:
        result = params.clone()
        if len(result.shape) == 1:
            result[0] = torch.log10(result[0])
            result[3] = torch.log10(result[3])
        elif len(result.shape) == 2:
            result[:,0] = torch.log10(result[:,0])
            result[:,3] = torch.log10(result[:,3])
        else:
            raise ValueError("Number of dimensions of input should be 1 or 2!")
        return result
        
    def internal_to_physical(self, params:torch.Tensor) -> torch.Tensor:
        result = params.clone()
        if len(result.shape) == 1:
            result[0] = 10 ** result[0]
            result[3] = 10 ** result[3]
        elif len(result.shape) == 2:
            result[:,0] = 10 ** result[:,0]
            result[:,3] = 10 ** result[:,3]
        else:
            raise ValueError("Number of dimensions of input should be 1 or 2!")
        return result

    def within_bounds(self, params) -> bool:
        if params[0] > 25750 and params[2] < 3.3:
            return False
        if params[0] > 10000 and params[2] < 3.0:
            return False
        return (
            params[0] >= 6000 and params[0] <= 30000 and
            params[1] >= -0.8 and params[1] <= 0.8 and
            params[2] >= 2.5 and params[2] <= 5.0 and
            params[3] >= 1 and params[3] <= 500 and
            params[4] >= -300 and params[4] <= 300
        )

    def interpolate(self, teff:float, m_h:float, logg:float) -> torch.Tensor:
        # Decide which grid to use
        if teff <= 9900 or (teff <= 10_000 and logg < 3.0):
            model_grid = self.model_grid1
            teff_gridpoints = self.teff_gridpoints1
            m_h_gridpoints = self.m_h_gridpoints1
            logg_gridpoints = self.logg_gridpoints1
        elif teff <= 25500 or (teff <= 25750 and logg < 3.3):
            model_grid = self.model_grid2
            teff_gridpoints = self.teff_gridpoints2
            m_h_gridpoints = self.m_h_gridpoints2
            logg_gridpoints = self.logg_gridpoints2
        else:
            model_grid = self.model_grid3
            teff_gridpoints = self.teff_gridpoints3
            m_h_gridpoints = self.m_h_gridpoints3
            logg_gridpoints = self.logg_gridpoints3
        
        # Convert physical values of Teff, [M/H] and log g to internal values
        x_teff = teff * 1e-2 - (teff // 1e2)
        x_m_h = m_h
        x_logg = logg - 4

        # Find correct location in model grid
        # The window in the model grid we need to interpolate is between the start index and
        # the start index plus 4, unless this window is partially outside of the model grid,
        # in which case we need to do quadratic interpolation instead of cubic interpolation
        # in one or more of the dimensions
        teff_start = -1
        while teff_gridpoints[teff_start+2] < teff:
            teff_start += 1
        m_h_start = -1
        while m_h_gridpoints[m_h_start+2] < m_h:
            m_h_start += 1
        logg_start = -1
        while logg_gridpoints[logg_start+2] < logg:
            logg_start += 1

        teff_end = teff_start + 4
        m_h_end = m_h_start + 4
        logg_end = logg_start + 4

        # Test if we need to do quadratic interpolation
        if teff_start < 0:
            teff_start = 0
        elif teff_start > len(teff_gridpoints) - 4:
            teff_end -= 1

        if m_h_start < 0:
            m_h_start = 0
        elif m_h_start > len(m_h_gridpoints) - 4:
            m_h_end -= 1

        if logg_start < 0:
            logg_start = 0
        elif logg_start > len(logg_gridpoints) - 4:
            logg_end -= 1
        
        #region Teff interpolation
        if teff_end == teff_start + 3:
            # Convert Teff to internal values
            xp = teff_gridpoints[teff_start:teff_start+3].unsqueeze(1).unsqueeze(2).unsqueeze(3) * 1e-2 - (teff // 1e2)

            col0_denom = 1 / ( xp[0]**2 - xp[0]*xp[1] - xp[0]*xp[2] + xp[1]*xp[2])
            col1_denom = 1 / (-xp[1]**2 + xp[0]*xp[1] - xp[0]*xp[2] + xp[1]*xp[2])
            col2_denom = 1 / ( xp[2]**2 + xp[0]*xp[1] - xp[0]*xp[2] - xp[1]*xp[2])

            c2 = (model_grid[teff_start, m_h_start:m_h_end, logg_start:logg_end] * col0_denom
                  - model_grid[teff_start+1, m_h_start:m_h_end, logg_start:logg_end] * col1_denom
                  + model_grid[teff_start+2, m_h_start:m_h_end, logg_start:logg_end] * col2_denom)
            c1 = (-model_grid[teff_start, m_h_start:m_h_end, logg_start:logg_end] * (xp[1]+xp[2]) * col0_denom
                  + model_grid[teff_start+1, m_h_start:m_h_end, logg_start:logg_end] * (xp[0]+xp[2]) * col1_denom
                  - model_grid[teff_start+2, m_h_start:m_h_end, logg_start:logg_end] * (xp[0]+xp[1]) * col2_denom)
            c0 = (model_grid[teff_start, m_h_start:m_h_end, logg_start:logg_end] * xp[1] * xp[2] * col0_denom
                  - model_grid[teff_start+1, m_h_start:m_h_end, logg_start:logg_end] * xp[0] * xp[2] * col1_denom
                  + model_grid[teff_start+2, m_h_start:m_h_end, logg_start:logg_end] * xp[0] * xp[1] * col2_denom)
            
            result = c2 * x_teff**2 + c1 * x_teff + c0
        
        elif teff_end == teff_start + 4:
            xp = teff_gridpoints[teff_start:teff_start+4].unsqueeze(1).unsqueeze(2).unsqueeze(3) * 1e-2 - (teff // 1e2)

            col0_denom = 1 / ( xp[0]**3 - xp[0]**2*xp[1] - xp[0]**2*xp[2] - xp[0]**2*xp[3] + xp[0]*xp[1]*xp[2] + xp[0]*xp[1]*xp[3] + xp[0]*xp[2]*xp[3] - xp[1]*xp[2]*xp[3])
            col1_denom = 1 / (-xp[1]**3 + xp[1]**2*xp[0] + xp[1]**2*xp[2] + xp[1]**2*xp[3] - xp[0]*xp[1]*xp[2] - xp[0]*xp[1]*xp[3] + xp[0]*xp[2]*xp[3] - xp[1]*xp[2]*xp[3])
            col2_denom = 1 / ( xp[2]**3 - xp[2]**2*xp[0] - xp[2]**2*xp[1] - xp[2]**2*xp[3] + xp[0]*xp[1]*xp[2] - xp[0]*xp[1]*xp[3] + xp[0]*xp[2]*xp[3] + xp[1]*xp[2]*xp[3])
            col3_denom = 1 / (-xp[3]**3 + xp[3]**2*xp[0] + xp[3]**2*xp[1] + xp[3]**2*xp[2] + xp[0]*xp[1]*xp[2] - xp[0]*xp[1]*xp[3] - xp[0]*xp[2]*xp[3] - xp[1]*xp[2]*xp[3])

            c3 = (model_grid[teff_start, m_h_start:m_h_end, logg_start:logg_end] * col0_denom
                - model_grid[teff_start+1, m_h_start:m_h_end, logg_start:logg_end] * col1_denom
                + model_grid[teff_start+2, m_h_start:m_h_end, logg_start:logg_end] * col2_denom
                - model_grid[teff_start+3, m_h_start:m_h_end, logg_start:logg_end] * col3_denom)
            c2 = (model_grid[teff_start, m_h_start:m_h_end, logg_start:logg_end] * (-xp[1] - xp[2] - xp[3]) * col0_denom
                + model_grid[teff_start+1, m_h_start:m_h_end, logg_start:logg_end] * (xp[0] + xp[2] + xp[3]) * col1_denom
                + model_grid[teff_start+2, m_h_start:m_h_end, logg_start:logg_end] * (-xp[0]-xp[1]-xp[3]) * col2_denom
                + model_grid[teff_start+3, m_h_start:m_h_end, logg_start:logg_end] * (xp[0]+xp[1]+xp[2]) * col3_denom)
            c1 = (model_grid[teff_start, m_h_start:m_h_end, logg_start:logg_end] * (xp[1]*xp[2] + xp[1]*xp[3] + xp[2]*xp[3]) * col0_denom 
                + model_grid[teff_start+1, m_h_start:m_h_end, logg_start:logg_end] * (-xp[0]*xp[2] - xp[0]*xp[3] - xp[2]*xp[3]) * col1_denom 
                + model_grid[teff_start+2, m_h_start:m_h_end, logg_start:logg_end] * (xp[0]*xp[1] + xp[0]*xp[3] + xp[1]*xp[3]) * col2_denom 
                + model_grid[teff_start+3, m_h_start:m_h_end, logg_start:logg_end] * (-xp[0]*xp[1] - xp[0]*xp[2] - xp[1]*xp[2]) * col3_denom)
            c0 = (-model_grid[teff_start, m_h_start:m_h_end, logg_start:logg_end] * xp[1] * xp[2] * xp[3] * col0_denom
                + model_grid[teff_start+1, m_h_start:m_h_end, logg_start:logg_end] * xp[0] * xp[2] * xp[3] * col1_denom
                - model_grid[teff_start+2, m_h_start:m_h_end, logg_start:logg_end] * xp[0] * xp[1] * xp[3] * col2_denom
                + model_grid[teff_start+3, m_h_start:m_h_end, logg_start:logg_end] * xp[0] * xp[1] * xp[2] * col3_denom)
            
            result = c3 * x_teff**3 + c2 * x_teff**2 + c1 * x_teff + c0
        else:
            raise ValueError("AAAHHH SOMETHING WENT WRONG! PANIC!!!!")
        #endregion Teff interpolation
        
        #region [M/H] interpolation
        if m_h_end == m_h_start + 3:
            xp = m_h_gridpoints[m_h_start:m_h_start+3].unsqueeze(1).unsqueeze(2)

            col0_denom = 1 / ( xp[0]**2 - xp[0]*xp[1] - xp[0]*xp[2] + xp[1]*xp[2])
            col1_denom = 1 / (-xp[1]**2 + xp[0]*xp[1] - xp[0]*xp[2] + xp[1]*xp[2])
            col2_denom = 1 / ( xp[2]**2 + xp[0]*xp[1] - xp[0]*xp[2] - xp[1]*xp[2])

            c2 =  result[0] * col0_denom                 - result[1] * col1_denom                 + result[2] * col2_denom
            c1 = -result[0] * (xp[1]+xp[2]) * col0_denom + result[1] * (xp[0]+xp[2]) * col1_denom - result[2] * (xp[0]+xp[1]) * col2_denom
            c0 =  result[0] * xp[1] * xp[2] * col0_denom - result[1] * xp[0] * xp[2] * col1_denom + result[2] * xp[0] * xp[1] * col2_denom
            
            result = c2 * x_m_h**2 + c1 * x_m_h + c0
        
        elif m_h_end == m_h_start + 4:
            xp = m_h_gridpoints[m_h_start:m_h_start+4].unsqueeze(1).unsqueeze(2)

            col0_denom = 1 / ( xp[0]**3 - xp[0]**2*xp[1] - xp[0]**2*xp[2] - xp[0]**2*xp[3] + xp[0]*xp[1]*xp[2] + xp[0]*xp[1]*xp[3] + xp[0]*xp[2]*xp[3] - xp[1]*xp[2]*xp[3])
            col1_denom = 1 / (-xp[1]**3 + xp[1]**2*xp[0] + xp[1]**2*xp[2] + xp[1]**2*xp[3] - xp[0]*xp[1]*xp[2] - xp[0]*xp[1]*xp[3] + xp[0]*xp[2]*xp[3] - xp[1]*xp[2]*xp[3])
            col2_denom = 1 / ( xp[2]**3 - xp[2]**2*xp[0] - xp[2]**2*xp[1] - xp[2]**2*xp[3] + xp[0]*xp[1]*xp[2] - xp[0]*xp[1]*xp[3] + xp[0]*xp[2]*xp[3] + xp[1]*xp[2]*xp[3])
            col3_denom = 1 / (-xp[3]**3 + xp[3]**2*xp[0] + xp[3]**2*xp[1] + xp[3]**2*xp[2] + xp[0]*xp[1]*xp[2] - xp[0]*xp[1]*xp[3] - xp[0]*xp[2]*xp[3] - xp[1]*xp[2]*xp[3])

            c3 = (result[0] * col0_denom
                - result[1] * col1_denom
                + result[2] * col2_denom
                - result[3] * col3_denom)
            c2 = (result[0] * (-xp[1] - xp[2] - xp[3]) * col0_denom
                + result[1] * (xp[0] + xp[2] + xp[3]) * col1_denom
                + result[2] * (-xp[0]-xp[1]-xp[3]) * col2_denom
                + result[3] * (xp[0]+xp[1]+xp[2]) * col3_denom)
            c1 = (result[0] * (xp[1]*xp[2] + xp[1]*xp[3] + xp[2]*xp[3]) * col0_denom 
                + result[1] * (-xp[0]*xp[2] - xp[0]*xp[3] - xp[2]*xp[3]) * col1_denom 
                + result[2] * (xp[0]*xp[1] + xp[0]*xp[3] + xp[1]*xp[3]) * col2_denom 
                + result[3] * (-xp[0]*xp[1] - xp[0]*xp[2] - xp[1]*xp[2]) * col3_denom)
            c0 = (-result[0] * xp[1] * xp[2] * xp[3] * col0_denom
                + result[1] * xp[0] * xp[2] * xp[3] * col1_denom
                - result[2] * xp[0] * xp[1] * xp[3] * col2_denom
                + result[3] * xp[0] * xp[1] * xp[2] * col3_denom)
            
            result = c3 * x_m_h**3 + c2 * x_m_h**2 + c1 * x_m_h + c0
        else:
            raise ValueError("AAAHHH SOMETHING WENT WRONG! PANIC!!!!")
        #endregion [M/H] interpolation
        
        #region log g interpolation
        if logg_end == logg_start + 3:
            xp = logg_gridpoints[logg_start:logg_start+3].unsqueeze(1) - 4

            col0_denom = 1 / ( xp[0]**2 - xp[0]*xp[1] - xp[0]*xp[2] + xp[1]*xp[2])
            col1_denom = 1 / (-xp[1]**2 + xp[0]*xp[1] - xp[0]*xp[2] + xp[1]*xp[2])
            col2_denom = 1 / ( xp[2]**2 + xp[0]*xp[1] - xp[0]*xp[2] - xp[1]*xp[2])

            c2 =  result[0] * col0_denom                 - result[1] * col1_denom                 + result[2] * col2_denom
            c1 = -result[0] * (xp[1]+xp[2]) * col0_denom + result[1] * (xp[0]+xp[2]) * col1_denom - result[2] * (xp[0]+xp[1]) * col2_denom
            c0 =  result[0] * xp[1] * xp[2] * col0_denom - result[1] * xp[0] * xp[2] * col1_denom + result[2] * xp[0] * xp[1] * col2_denom
            
            result = c2 * x_logg**2 + c1 * x_logg + c0
        
        elif logg_end == logg_start + 4:
            xp = logg_gridpoints[logg_start:logg_start+4].unsqueeze(1) - 4

            col0_denom = 1 / ( xp[0]**3 - xp[0]**2*xp[1] - xp[0]**2*xp[2] - xp[0]**2*xp[3] + xp[0]*xp[1]*xp[2] + xp[0]*xp[1]*xp[3] + xp[0]*xp[2]*xp[3] - xp[1]*xp[2]*xp[3])
            col1_denom = 1 / (-xp[1]**3 + xp[1]**2*xp[0] + xp[1]**2*xp[2] + xp[1]**2*xp[3] - xp[0]*xp[1]*xp[2] - xp[0]*xp[1]*xp[3] + xp[0]*xp[2]*xp[3] - xp[1]*xp[2]*xp[3])
            col2_denom = 1 / ( xp[2]**3 - xp[2]**2*xp[0] - xp[2]**2*xp[1] - xp[2]**2*xp[3] + xp[0]*xp[1]*xp[2] - xp[0]*xp[1]*xp[3] + xp[0]*xp[2]*xp[3] + xp[1]*xp[2]*xp[3])
            col3_denom = 1 / (-xp[3]**3 + xp[3]**2*xp[0] + xp[3]**2*xp[1] + xp[3]**2*xp[2] + xp[0]*xp[1]*xp[2] - xp[0]*xp[1]*xp[3] - xp[0]*xp[2]*xp[3] - xp[1]*xp[2]*xp[3])

            c3 = (result[0] * col0_denom
                - result[1] * col1_denom
                + result[2] * col2_denom
                - result[3] * col3_denom)
            c2 = (result[0] * (-xp[1] - xp[2] - xp[3]) * col0_denom
                + result[1] * (xp[0] + xp[2] + xp[3]) * col1_denom
                + result[2] * (-xp[0]-xp[1]-xp[3]) * col2_denom
                + result[3] * (xp[0]+xp[1]+xp[2]) * col3_denom)
            c1 = (result[0] * (xp[1]*xp[2] + xp[1]*xp[3] + xp[2]*xp[3]) * col0_denom 
                + result[1] * (-xp[0]*xp[2] - xp[0]*xp[3] - xp[2]*xp[3]) * col1_denom 
                + result[2] * (xp[0]*xp[1] + xp[0]*xp[3] + xp[1]*xp[3]) * col2_denom 
                + result[3] * (-xp[0]*xp[1] - xp[0]*xp[2] - xp[1]*xp[2]) * col3_denom)
            c0 = (-result[0] * xp[1] * xp[2] * xp[3] * col0_denom
                + result[1] * xp[0] * xp[2] * xp[3] * col1_denom
                - result[2] * xp[0] * xp[1] * xp[3] * col2_denom
                + result[3] * xp[0] * xp[1] * xp[2] * col3_denom)
            
            result = c3 * x_logg**3 + c2 * x_logg**2 + c1 * x_logg + c0
        else:
            raise ValueError("AAAHHH SOMETHING WENT WRONG! PANIC!!!!")
        #endregion log g interpolation
        
        return result

    def set_observed(self, observed:np.ndarray) -> None:
        self.observed = torch.from_numpy(observed).to(device=self.device)

    def create_gaussian_kernels(self, size:int) -> torch.Tensor:
        sigmas = 61402.18438872159 / interp_tensor(self.wl, self.observed[:,0], self.observed[:,3])
        x = torch.arange(-size // 2 + 1., size // 2 + 1., device=self.device)
        x = x.view(1, -1)  # Shape: (1, kernel_size)
        sigmas = sigmas.view(-1, 1)  # Shape: (batch_size, 1)
        kernels = torch.exp(-0.5 * (x / sigmas)**2)
        kernels = kernels / kernels.sum(dim=1, keepdim=True)
        return kernels

    def produce_model(self, params, check_if_within_bounds=True) -> torch.Tensor:
        if check_if_within_bounds and not self.within_bounds(params):
            raise ValueError("Parameters not within bounds!")
        
        # Interpolate model
        model = self.interpolate(params[0], params[1], params[2])
        # Perform vsini convolution
        model = model.unsqueeze(0).unsqueeze(0)
        kernel = torch.from_numpy(vsini_kernel(params[3])).unsqueeze(0).unsqueeze(0).to(device=self.device)
        model = F.conv1d(model, kernel, padding='same').squeeze()

        #region Perform resolution convolution
        # =========================================================================================
        # Determine the kernel size by taking the smallest resolution, which translates to
        # the maximum broadening. For this pixel we still want the kernel to cover 3-sigma.
        kernel_size = int(3 * 61402.18438872159 / torch.min(self.observed[:,3]).item())
        # Make sure the kernel size is odd
        if kernel_size // 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        # Pad Y for convolution
        model_padded = F.pad(model, (padding, padding), mode='constant')
        for i in range(padding):
            model_padded[i] = model[0]
            model_padded[len(model_padded)-i-1] = model[0]
        
        # Generate Gaussian kernels
        kernels = self.create_gaussian_kernels(kernel_size)

        # Convolution
        model_padded = model_padded.unfold(0, kernel_size, 1)  # Shape: (batch_size, kernel_size)
        model = (model_padded * kernels).sum(dim=1)
        #endregion Perform resolution convolution =================================================

        # Radial velocity and resampling
        # Shift observed wavelength to find wavelength in rest frame of star
        wl_shifted = self.observed[:,0] * (1.0 - params[4] / 299_792.458)
        return interp_tensor(wl_shifted, self.wl, model)
    
    def fit_continuum(self, model) -> torch.Tensor:
        # Cutting spectrum into partially overlapping chunks
        n_chunks = 5
        overlap = 0.2 # 0 = no overlap, 0.5 = max overlap
        chunks_startstop = torch.zeros(size=(n_chunks, 2), dtype=torch.int32)
        for i in range(n_chunks):
            start = torch.clamp(map_range(i - overlap, 0, n_chunks, self.observed[0,0], self.observed[-1,0]), self.observed[0,0], self.observed[-1,0])
            stop = torch.clamp(map_range(i + 1 + overlap, 0, n_chunks, self.observed[0,0], self.observed[-1,0]), self.observed[0,0], self.observed[-1,0])
            chunks_startstop[i, 0] = torch.le(self.observed[:,0], start).sum().item()
            chunks_startstop[i, 1] = min(torch.le(self.observed[:,0], stop).sum().item(), len(self.observed) - 1)
        
        # Fitting a polynomial to each chunk
        porder = 8
        #pfits = torch.zeros((n_chunks, porder+1), device=self.device)
        pvals = torch.zeros((n_chunks, len(self.observed)), device=self.device)
        y = self.observed[:,1] / model
        for c in range(n_chunks):
            wl_remap = ((self.observed[:,0] - self.observed[chunks_startstop[c][0],0])
                        / (self.observed[chunks_startstop[c][1],0] - self.observed[chunks_startstop[c][0],0]) * 2 - 1)

            # Polynomial fit
            powers = torch.arange(porder+1)
            wl_pow = torch.pow(wl_remap.unsqueeze(dim=0), powers.unsqueeze(dim=1))
            b = (y[chunks_startstop[c][0]:chunks_startstop[c][1]] * wl_pow[:,chunks_startstop[c][0]:chunks_startstop[c][1]]).sum(dim=1)
            
            powers = torch.arange(2*porder+1)
            A = torch.pow(wl_remap[chunks_startstop[c][0]:chunks_startstop[c][1]].unsqueeze(dim=0), powers.unsqueeze(dim=1)).sum(dim=1).unfold(0, porder+1, 1)
            
            coef = torch.matmul(torch.inverse(A), b)

            # Evaluate polynomial
            pvals[c] = (wl_pow * coef.unsqueeze(dim=1)).sum(dim=0)
        
        # Blend polynomials
        for c in range(1, n_chunks):
            mask = torch.sin(torch.clamp((self.observed[:,0] - self.observed[chunks_startstop[c-1][1],0])
                                         / (self.observed[chunks_startstop[c][0],0] - self.observed[chunks_startstop[c-1][1],0])
                                         * 1.4 - 0.2, 0, 1) * np.pi/2)**2
            pvals[c-1] *= mask
            pvals[c] *= 1-mask
        
        return pvals.sum(dim=0)

    def calculate_cost(self, params) -> float:
        model = self.produce_model(params)
        model *= self.fit_continuum(model)
        return torch.mean(((model - self.observed[:,1]) / self.observed[:,2]) ** 2).item()
        

