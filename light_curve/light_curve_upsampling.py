import os,json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Distance
import extinction
from extinctions import reddening
import warnings
from light_curve.GP_fitting import *
from config import *
from utils import *
import multiprocessing as mp
from typing import Tuple, Optional, List, Union
from astropy.stats import sigma_clip
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings("ignore", category=DeprecationWarning) 



def ext(ra,dec):
    red = (reddening.Reddening(ra, dec)).query_local_map(dustmap='sfd')*0.86
    AV = 3.1*float(str(red)[1:-1])
    
    wave = np.array([4829.50, 6463.75, 4900.12, 6241.27, 7563.76, 8690.10, 9644.63]) 

    AC_keys = ['ZTF_g', 'ZTF_r', 'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y']
    AC = extinction.fitzpatrick99(wave, AV, 3.1)
    
    ext_val = {}
    for A, B in zip(AC_keys, AC):
        ext_val[A] = B
    
    return ext_val


class LightCurveUpsamplingPipeline:
    '''
    This class is used to simulate new light curves for a given ZTF object using 2D GP from avocado.
    The host data remains the same.
    The new light curve is upsampled to the same cadence as the original light curve.
    The transient photometric data are re-calculated.
    '''
    def __init__(self, ztf_object = None, designated_class = None,  
                 mag_path=None,
                 output_path=None,
                 ztf_obj_label_path=None,
                 img_host_data = None,
                 img_z = None,
                 min_detection = 2,
                 gp_fitting = True,
                 load_gp = True
                 ):
        '''
        initialize the class
        '''
        if ztf_object is None and designated_class is None:
            raise ValueError("Either ztf_object or designated_class must be provided")
        
        self.__mag_path = os.path.join(DEFAULT_DATA_PATH, 'mag_sets_v4') \
            if mag_path is None else mag_path
   
        self.__output_path = PHOTO_OUTPUT_PATH \
            if output_path is None else output_path
        
        # self.img_sample_list = load_sample_imgs()

        if ztf_object is None and designated_class is not None: 
            self.__label_dict = self.__load_class_samples(ztf_obj_label_path)
            self.ztf_object = self.__shuffle_class_samples(designated_class)
        else:
            self.ztf_object = ztf_object
            
        print('light curve upsampling ztf object: ', self.ztf_object)
        
        self.executed = True
        self.__output_dir = os.path.join(self.__output_path, self.ztf_object)
        self.__mag_dir = os.path.join(self.__mag_path, self.ztf_object + '.json')

        if not os.path.exists(self.__mag_dir):
            print(f"Invalid magnitude directory: {self.__mag_dir}")
            self.executed = False
        try: 
            self.mag_data = self.load_mag_data
            if self.mag_data is None:
                print(f"No magnitude data for {self.ztf_object}")
                self.executed = False
        except:
            print(f"Error loading magnitude data for {self.ztf_object}")
            self.executed = False

        
        self.matched_host_data = img_host_data
        self.img_z = img_z
        self.lc_z = self.lc_redshift
        self.predictions = None
        self.pred_times = None
        self.prediction_uncertainties = None
        self.lc_features = None
        self.lc_data = None
        self.photo_sample_list = load_sample_lc()
        self.min_detection = min_detection
        
        if self.executed:
            self.ra, self.dec = self.mag_data['objectData']['ramean'], self.mag_data['objectData']['decmean']
            self.object_exts = ext(self.ra, self.dec)
            if gp_fitting:
                if load_gp:
                    if self.photo_sample_list is not None:
                        if self.ztf_object in self.photo_sample_list:
                            self.load_light_curve()
                            valid = self.load_gp_fitting()
                            if not valid:
                                print(f"No good GP fitting results found for {self.ztf_object}")
                                self.executed = False
                                self.valid_lc = False
                                return 
                        
                        else:
                            print(f"No light curve data for {self.ztf_object}")
                            self.executed = False
                            self.valid_lc = False
                            return 
                    else:
                        print(f'need to provide photo_sample_list path in the utils.py')
                        self.executed = False
                        self.valid_lc = False
                        return 
                else: 
                    self.load_light_curve()
                    self.lc_features = self.get_light_curve_statistics(self.lc_data, peak_define = 'mag', min_detection = self.min_detection)
                    self.valid_lc = self.lc_features['g_phase'] or self.lc_features['r_phase']
                    if self.valid_lc:
                        print(f"Light curve for {self.ztf_object} is valid")
                        self.fit_2d_gp()
                    else:
                        print(f"No more than {self.min_detection} detections (before peak) for {self.ztf_object} in g-band or r-band")
                        self.executed = False
                        return 
            else:
               self.load_light_curve()
               self.lc_features = self.get_light_curve_statistics(self.lc_data, peak_define = 'mag', min_detection = min_detection)
               self.valid_lc = self.lc_features['g_phase'] or self.lc_features['r_phase']
               if self.valid_lc:
                    print(f"Light curve for {self.ztf_object} is valid")
               else:
                    print(f"No more than {min_detection} detections (before peak) for {self.ztf_object} in g-band or r-band")
                    self.executed = False
                    return 

        else:
            print(f"No magnitude data for {self.ztf_object}")
            self.valid_lc = False
            self.executed = False
            return 

    def __load_class_samples(self, ztf_obj_label_path):
        '''
        load the class samples
        label = ['SLSN-I', 'TDE']
        '''

        if not os.path.exists(ztf_obj_label_path):
            raise FileNotFoundError(f"Label file not found: {ztf_obj_label_path}")
            
        info_df = pd.read_csv(ztf_obj_label_path)
        if info_df.empty:
            raise ValueError("Empty label file")
            
        if not all(col in info_df.columns for col in ['type', 'ZTFID']):
            raise ValueError("Label file missing required columns 'type' and/or 'ZTFID'")

        # only pick up the SLSN-I and TDE samples

        obj_dict = {'SLSN-I':info_df[info_df['type'] == 'SLSN-I']['ZTFID'].values.tolist(), 
               'TDE':info_df[info_df['type'] == 'TDE']['ZTFID'].values.tolist()}
        
        good_obj_dict = {'SLSN-I':[x for x in obj_dict['SLSN-I'] if x in self.photo_sample_list],  
                         'TDE':[x for x in obj_dict['TDE'] if x in self.photo_sample_list]}
        
        return good_obj_dict 
    
    def __shuffle_class_samples(self, label):
        '''
        shuffle the class samples
        '''

        if label not in self.__label_dict:
            raise ValueError(f"Invalid label: {label}. Must be one of {list(self.__label_dict.keys())}")
            
        if len(self.__label_dict[label]) == 0:
            raise ValueError(f"No samples available for label: {label}")

        np.random.seed(None)
        return np.random.choice(self.__label_dict[label], size = 1, replace = False)[0] 

   

    @property
    def lc_redshift(self, input_z = None):
        '''
        get the redshift of the object. If the spectroscopic redshift is not available, use the photometric redshift.
        '''

        # try: 
        if input_z is None:
            obj_info = load_redshift_database()
            if self.ztf_object not in obj_info:
                return None
            archive_z = obj_info[self.ztf_object]  
            if archive_z is not None:
                return archive_z
            else:
                if 'TNS' in self.mag_data and 'z' in self.mag_data['TNS'] and self.mag_data['TNS']['z'] is not None:
                    return float(self.mag_data['TNS']['z'])
                else:
                    return None       
        else:
            return input_z
        # except:
        #     print(f"Error light curve loading redshift for {self.ztf_object}")
        #     return None

    @property
    def load_mag_data(self):
        """
        load the magnitude data
        """
        try: 
            with open(self.__mag_dir, 'r') as f:
                mag_data = json.load(f)
            return mag_data
        except:
            print(f"Error loading magnitude data for {self.ztf_object}")
            return None


    def load_light_curve(self):
        '''
        load the light curve
        '''
        if self.mag_data is None:
            self.mag_data = self.load_mag_data
            if self.mag_data is None:
                return None
        if 'objectData' in self.mag_data and 'discMjd' in self.mag_data['objectData']:
            self.disc_mjd = self.mag_data['objectData']['discMjd']
        elif 'objectData' in self.mag_data and 'jdmin' in self.mag_data['objectData']:
            self.disc_mjd = self.mag_data['objectData']['jdmin'] - 2400000.5
        else:
            print(f"Error loading light curve for {self.ztf_object}")
            print(self.mag_data['objectData'].keys())
            return None
            
        max_mjd = self.disc_mjd + 300

        # Extract g and r band data
        g_band_data = []
        r_band_data = []
        max_gap = 200
        neighbor_mjd = self.disc_mjd
        for candidate in self.mag_data['candidates']:
            if 'candid' in candidate.keys() and candidate['mjd'] < max_mjd:
                if candidate['mjd'] - neighbor_mjd <= max_gap: 
                    if candidate['fid'] == 1:  # g-band
                        g_band_data.append((candidate['mjd'], candidate['magpsf'], candidate['sigmapsf']))
                    elif candidate['fid'] == 2:  # r-band
                        r_band_data.append((candidate['mjd'], candidate['magpsf'], candidate['sigmapsf']))
                    neighbor_mjd = candidate['mjd']

        # Sort the data by date
        g_band_data.sort(key=lambda x: x[0])
        r_band_data.sort(key=lambda x: x[0])

        # Separate dates and magnitudes
        g_dates, g_mags, g_mags_err = zip(*g_band_data) if g_band_data else ((), (), ())
        r_dates, r_mags, r_mags_err = zip(*r_band_data) if r_band_data else ((), (), ())

        # remove the extinction correction for the light curve
        g_mags = g_mags - self.object_exts['ZTF_g']
        r_mags = r_mags - self.object_exts['ZTF_r']

        first_date = min(g_dates + r_dates) if g_dates or r_dates else None
        g_days = np.array(g_dates) - first_date
        r_days = np.array(r_dates) - first_date

        # convert to pandas dataframe
        lc_data = pd.DataFrame({
            'time': np.concatenate([g_days, r_days]),
            'mag': np.concatenate([g_mags, r_mags]),
            'mag_err': np.concatenate([g_mags_err, r_mags_err]),
            'band': np.concatenate([np.repeat('ztfg', len(g_days)), np.repeat('ztfr', len(r_days))])
        })

        lc_data = self.uniform_light_curve(lc_data)
     

        self.lc_data = lc_data
        self.detrend_and_clip()


    def apply_img_z_to_lc(self, upsampled_lc, ztf_limit = 20.5):
        '''
        here we reconstruct the light curve to the img_z.
        the brightness change is estimated using formula:
        m_img = m_lc + 5 * log10(d_img**2 / d_lc**2)
        since needle does not use mag_error, here we do not consider the mag_error change.
        '''


        if self.img_z is not None and self.lc_z is not None:
            upsampled_lc['time'] = upsampled_lc['time'] * (1 + self.img_z) / (1 + self.lc_z)
            img_d = Distance(z = self.img_z, unit = 'Mpc', cosmology = None)
            lc_d = Distance(z = self.lc_z, unit = 'Mpc', cosmology = None)
            upsampled_lc['mag'] = upsampled_lc['mag'] + 2.5 * np.log10((img_d**2) / (lc_d**2))
        else:
            print('no redshift for this light curve or img_z is not provided, return the original light curve.')

        upsampled_lc['mag_err'] = 2.4 * 10**(-5) * 10 ** (0.2 * upsampled_lc['mag']) # ZTF mag error
        upsampled_lc['mag'] = upsampled_lc['mag'] + np.random.uniform(-1, 1) * upsampled_lc['mag_err']


        upsampled_lc = upsampled_lc[upsampled_lc['mag'] <= ztf_limit] # ZTF limit cut

        if len(upsampled_lc) == 0:
            print(f"No detections for {self.ztf_object} after applying img_z to the light curve and ZTF limit correction.")
            return None
        
        return upsampled_lc
    

    def detrend_and_clip(self, spline_smooth=1e-2, sigma=3, universal_gap_limit = 30, band_gap_limit = 80):
        '''
        detrend and clip the light curve, here we use the spline fitting and sigma clipping to remove the outliers.
        spline_smooth: the smoothing factor for the spline fitting, default is 1e-2.
        sigma: the sigma value for the sigma clipping, default is 3.
        gap_limit: the gap limit for the light curve, default is 80.
        for detection gaps between different bands, we use the universal_gap_limit.
        for detection gaps within the same band, we use the band_gap_limit.
        '''
        print('--------------------------------detrending and clipping the light curve--------------------------------')
        self.lc_data = self.lc_data.dropna()

        time_diff = np.diff(self.lc_data["time"])
        time_diff = np.append(time_diff, 0)  # Aligns length with band_mjd
        mask = time_diff <= universal_gap_limit
        self.lc_data = self.lc_data.iloc[mask]
    
        for band in ['ztfg', 'ztfr']:
        
            band_data = self.lc_data.loc[self.lc_data["band"] == band, "mag"]
            band_mjd = self.lc_data.loc[self.lc_data["band"] == band, "time"]
            if len(band_mjd) > 0:
                time_diff = np.diff(band_mjd)
                time_diff = np.append(time_diff, 0)  # Aligns length with band_mjd
                mask = time_diff <= band_gap_limit
                band_data = band_data.iloc[mask]
                band_mjd = band_mjd.iloc[mask]  # Keep time and mag in sync
               
            # Check if we have enough data points for spline fitting (minimum 4 for cubic spline)
            if len(band_data) < 4:
                print(f'Warning: Not enough data points ({len(band_data)}) for spline fitting in band {band}. Skipping detrending.')
                if len(band_data) <= 1:
                    print('remove this band data: ', band)
                    self.lc_data = self.lc_data[self.lc_data["band"] != band]
                continue
            
            # Sort data by time to ensure x is increasing
            sort_idx = np.argsort(band_mjd)
            band_mjd = band_mjd.iloc[sort_idx]
            band_data = band_data.iloc[sort_idx]
            
            spline = UnivariateSpline(band_mjd, band_data, s=spline_smooth * len(band_mjd))
            
            trend = spline(band_mjd)
            residuals = band_data - trend
            clipped = sigma_clip(residuals, sigma=sigma)
            mask = ~clipped.mask
        
            masked_band_data = band_data[mask]
            masked_band_mjd = band_mjd[mask]
         
            self.lc_data.loc[self.lc_data["band"] == band, "mag"] = masked_band_data
            self.lc_data.loc[self.lc_data["band"] == band, "time"] = masked_band_mjd


        # Drop any rows that became NaN after detrending/clipping
        self.lc_data = self.lc_data.dropna()

        self.lc_data = self.lc_data.reset_index(drop=True)


        # self.mean_g, self.mean_r, self.std_g, self.std_r = self.normalize_param
  
        # print('Outliers removed by detrending and clipping.')

    def uniform_light_curve(self, lc_data, window_size = 1):
        '''
        merge detection within the window size.
        '''
        print('--------------------------------uniforming the light curve--------------------------------')

        if not isinstance(lc_data, pd.DataFrame):
            raise TypeError("lc_data must be a pandas DataFrame")
            
        if window_size <= 0:
            raise ValueError("window_size must be positive")
            
        required_cols = ['time', 'mag', 'mag_err', 'band']
        if not all(col in lc_data.columns for col in required_cols):
            raise ValueError(f"lc_data missing required columns: {required_cols}")

        # Remove any duplicate rows based on all columns
        lc_data = lc_data.drop_duplicates(subset=['time', 'mag', 'mag_err', 'band'], keep='first')
      
        g_data = lc_data[lc_data['band'] == 'ztfg'].reset_index(drop=True)
        r_data = lc_data[lc_data['band'] == 'ztfr'].reset_index(drop=True)

        g_time_diff_idx = np.where(np.diff(g_data['time'].values) < window_size)[0]
        r_time_diff_idx = np.where(np.diff(r_data['time'].values) < window_size)[0]

        if g_time_diff_idx.size > 0 or r_time_diff_idx.size > 0:
            if g_time_diff_idx.size > 0:
                for i in range(len(g_time_diff_idx)): 
                    g_data.loc[g_time_diff_idx[i], 'time'] = np.mean([g_data['time'][g_time_diff_idx[i]], g_data['time'][g_time_diff_idx[i]+1]])
                    g_data.loc[g_time_diff_idx[i], 'mag'] = np.mean([g_data['mag'][g_time_diff_idx[i]], g_data['mag'][g_time_diff_idx[i]+1]])
                    g_data.loc[g_time_diff_idx[i], 'mag_err'] = np.mean([g_data['mag_err'][g_time_diff_idx[i]], g_data['mag_err'][g_time_diff_idx[i]+1]])
        
                g_data = g_data.drop(g_time_diff_idx + 1)
        
            if r_time_diff_idx.size > 0:
                for i in range(len(r_time_diff_idx)):
                    r_data.loc[r_time_diff_idx[i], 'time'] = np.mean([r_data['time'][r_time_diff_idx[i]], r_data['time'][r_time_diff_idx[i]+1]])
                    r_data.loc[r_time_diff_idx[i], 'mag'] = np.mean([r_data['mag'][r_time_diff_idx[i]], r_data['mag'][r_time_diff_idx[i]+1]])
                    r_data.loc[r_time_diff_idx[i], 'mag_err'] = np.mean([r_data['mag_err'][r_time_diff_idx[i]], r_data['mag_err'][r_time_diff_idx[i]+1]])
            
                r_data = r_data.drop(r_time_diff_idx + 1)

        lc_data = pd.concat([g_data, r_data]).reset_index(drop=True)

        # time_diff_idx = np.where(np.diff(lc_data['time'].values) < window_size)[0]

        # if time_diff_idx.size > 0:
        #     for i in range(len(time_diff_idx)):
        #         lc_data.loc[time_diff_idx[i], 'time'] = np.mean([lc_data['time'][time_diff_idx[i]], lc_data['time'][time_diff_idx[i]+1]])
        #         lc_data.loc[time_diff_idx[i], 'mag'] = np.mean([lc_data['mag'][time_diff_idx[i]], lc_data['mag'][time_diff_idx[i]+1]])
        #         lc_data.loc[time_diff_idx[i], 'mag_err'] = np.mean([lc_data['mag_err'][time_diff_idx[i]], lc_data['mag_err'][time_diff_idx[i]+1]])
        #     lc_data = lc_data.drop(time_diff_idx + 1).reset_index(drop=True)

        
        return lc_data
       

    def get_light_curve_statistics(self, lc_data, peak_define = 'mag', min_detection = 2, extend_phase = 5):
        '''
        get one light curve statistics - could be original or upsampled.
        '''
        if not isinstance(lc_data, pd.DataFrame):
            raise TypeError("lc_data must be a pandas DataFrame")
            
        if peak_define not in ['mag', 'snr']:
            raise ValueError("peak_define must be either 'mag' or 'snr'")
            
        required_cols = ['time', 'mag', 'mag_err', 'band']
        if not all(col in lc_data.columns for col in required_cols):
            raise ValueError(f"lc_data missing required columns: {required_cols}")

        lc_features = {}
        
        lc_features['g_num'] = len(lc_data[lc_data['band'] == 'ztfg'])
        lc_features['r_num'] = len(lc_data[lc_data['band'] == 'ztfr'])
        lc_features['g_mean'] = np.mean(lc_data[lc_data['band'] == 'ztfg']['mag'].values) if lc_features['g_num'] > 0 else None
        lc_features['r_mean'] = np.mean(lc_data[lc_data['band'] == 'ztfr']['mag'].values) if lc_features['r_num'] > 0 else None

        if peak_define == 'snr':
            g_mag_error_snr  = 1/lc_data[lc_data['band'] == 'ztfg']["mag_err"]
            r_mag_error_snr  = 1/lc_data[lc_data['band'] == 'ztfr']["mag_err"]
        
            g_mag_error_snr_max_idx = np.argmax(g_mag_error_snr) if lc_features['g_num'] > 0 else None
            r_mag_error_snr_max_idx = np.argmax(r_mag_error_snr) if lc_features['r_num'] > 0 else None
            g_mag_error_snr_min_idx = np.argmin(g_mag_error_snr) if lc_features['g_num'] > 0 else None
            r_mag_error_snr_min_idx = np.argmin(r_mag_error_snr) if lc_features['r_num'] > 0 else None
        
            lc_features['g_peak'] = lc_data[lc_data['band'] == 'ztfg']['mag'].values[g_mag_error_snr_max_idx] if g_mag_error_snr_max_idx is not None else None
            lc_features['r_peak'] = lc_data[lc_data['band'] == 'ztfr']['mag'].values[r_mag_error_snr_max_idx] if r_mag_error_snr_max_idx is not None else None
            g_peak_time = lc_data[lc_data['band'] == 'ztfg']['time'].values[g_mag_error_snr_max_idx] if g_mag_error_snr_max_idx is not None else np.nan
            r_peak_time = lc_data[lc_data['band'] == 'ztfr']['time'].values[r_mag_error_snr_max_idx] if r_mag_error_snr_max_idx is not None else np.nan
            lc_features['earliest_peak_time'] = np.nanmin([g_peak_time, r_peak_time]) + extend_phase if g_peak_time is not None or r_peak_time is not None else None
            lc_features['g_faint'] = lc_data[lc_data['band'] == 'ztfg']['mag'].values[g_mag_error_snr_min_idx] if g_mag_error_snr_min_idx is not None else None
            lc_features['r_faint'] = lc_data[lc_data['band'] == 'ztfr']['mag'].values[r_mag_error_snr_min_idx] if r_mag_error_snr_min_idx is not None else None
        elif peak_define == 'mag':
            lc_features['g_peak'] = np.min(lc_data[lc_data['band'] == 'ztfg']['mag'].values) if lc_features['g_num'] > 0 else None
            lc_features['r_peak'] = np.min(lc_data[lc_data['band'] == 'ztfr']['mag'].values) if lc_features['r_num'] > 0 else None
            g_peak_time = lc_data[lc_data['band'] == 'ztfg']['time'].values[np.argmin(lc_data[lc_data['band'] == 'ztfg']['mag'].values)] if lc_features['g_num'] > 0 else np.nan
            r_peak_time = lc_data[lc_data['band'] == 'ztfr']['time'].values[np.argmin(lc_data[lc_data['band'] == 'ztfr']['mag'].values)] if lc_features['r_num'] > 0 else np.nan
            lc_features['earliest_peak_time'] = np.nanmin([g_peak_time, r_peak_time]) + extend_phase if g_peak_time is not None or r_peak_time is not None else None
            lc_features['g_faint'] = np.max(lc_data[lc_data['band'] == 'ztfg']['mag'].values) if lc_features['g_num'] > 0 else None
            lc_features['r_faint'] = np.max(lc_data[lc_data['band'] == 'ztfr']['mag'].values) if lc_features['r_num'] > 0 else None


        lc_features['g_mag_err_min'] = np.min(lc_data[lc_data['band'] == 'ztfg']['mag_err']) if lc_features['g_num'] > 0 else None
        lc_features['g_mag_err_max'] = np.max(lc_data[lc_data['band'] == 'ztfg']['mag_err']) if lc_features['g_num'] > 0 else None
        lc_features['r_mag_err_min'] = np.min(lc_data[lc_data['band'] == 'ztfr']['mag_err']) if lc_features['r_num'] > 0 else None
        lc_features['r_mag_err_max'] = np.max(lc_data[lc_data['band'] == 'ztfr']['mag_err']) if lc_features['r_num'] > 0 else None
        
        # if there are no detection before peak in one/two band, set the phase to None, also do not consider this band in the upsampling process.
        g_time_peak_check = lc_data[lc_data['band'] == 'ztfg']['time'].values - lc_features['earliest_peak_time'] if lc_features['earliest_peak_time'] is not None else None
        r_time_peak_check = lc_data[lc_data['band'] == 'ztfr']['time'].values - lc_features['earliest_peak_time'] if lc_features['earliest_peak_time'] is not None else None
        
        lc_features['g_prepeak_num'] = len(g_time_peak_check[g_time_peak_check < 0]) if g_time_peak_check is not None else 0
        lc_features['r_prepeak_num'] = len(r_time_peak_check[r_time_peak_check < 0]) if r_time_peak_check is not None else 0

       
        # if there are more than min_detection detections before peak, set the phase to True, otherwise set to False
        lc_features['g_phase'] = lc_features['g_prepeak_num'] >= min_detection
        lc_features['r_phase'] = lc_features['r_prepeak_num'] >= min_detection

       

        return lc_features



    def fit_2d_gp(self):
        '''
        fit the Gaussian process to the light curve
        '''
        
        if not self.valid_lc:
            raise ValueError(f"No valid light curve for {self.ztf_object}")
            
        if not hasattr(self, 'lc_data') or self.lc_data is None:
            raise ValueError("Light curve data not available")
            

        self.astronomical_object = AstronomicalObject(self.lc_data, self.ztf_object)
        fitted_gp, _,_ = self.astronomical_object.fit_gaussian_process()

        min_time_obs = np.round(np.min(self.astronomical_object.observations["time"]))
        max_time_obs = np.round(np.max(self.astronomical_object.observations["time"]))
        min_time = min_time_obs
        max_time = max_time_obs
        self.pred_times = np.arange(min_time, max_time + 1)
        self.predictions, self.prediction_uncertainties = self.astronomical_object.predict_gaussian_process(self.astronomical_object.bands, 
                                                                            self.pred_times,
                                                                            uncertainties=True,
                                                                            fitted_gp=fitted_gp)

        self.mean_g, self.mean_r, self.std_g, self.std_r = self.astronomical_object.mean_g, self.astronomical_object.mean_r, self.astronomical_object.std_g, self.astronomical_object.std_r
      
      
        
    def uncertainty(self, mag_error):
        '''
        add some uncertainty to the magnitude error, to simlate the uncertain factors for error.
        '''
        np.random.seed(None)
        return mag_error * np.random.uniform(0.9, 1.1)
    
      
    def upsample_light_curve(self, seed = None):
        '''
        upsample the light curve to the same cadence as the original light curve
        '''
        if not self.valid_lc:
            print(f"No valid light curve for {self.ztf_object}")
            return None
            
        if not hasattr(self, 'predictions') or self.predictions is None or not hasattr(self, 'prediction_uncertainties') or self.prediction_uncertainties is None or not hasattr(self, 'pred_times') or self.pred_times is None:
            print("GP predictions not available. Run fit_2d_gp first.")
            return None
            
        if self.min_detection < 1:
            print("min_detection must be positive")
            return None
            
        if not hasattr(self, 'lc_features'):
            print("Light curve features not computed")
            return None
        
        if seed is not None:
            np.random.seed(seed)

        if self.lc_features is None:
            print("Light curve features not computed or None.")
            return None

        g_pred_num = np.random.randint(self.min_detection, self.lc_features['g_prepeak_num']+1) if self.lc_features['g_phase'] else 0
        r_pred_num = np.random.randint(self.min_detection, self.lc_features['r_prepeak_num']+1) if self.lc_features['r_phase'] else 0
        # print(f"DEBUG: Selected g_pred_num: {g_pred_num}, r_pred_num: {r_pred_num}")


        g_idx = np.random.choice(len(self.pred_times[self.pred_times <= self.lc_features['earliest_peak_time']]), g_pred_num, replace=False) 
        r_idx = np.random.choice(len(self.pred_times[self.pred_times <= self.lc_features['earliest_peak_time']]), r_pred_num, replace=False) 
        
        if len(g_idx) == 0:
            g_idx = None
        if len(r_idx) == 0:
            r_idx = None
        # print(f"DEBUG: g_idx shape: {g_idx.shape if g_idx is not None else None}, r_idx shape: {r_idx.shape if r_idx is not None else None}")

        g_pred_times = self.pred_times[g_idx] if g_idx is not None else np.array([])  
        r_pred_times = self.pred_times[r_idx] if r_idx is not None else np.array([])
        if g_idx is not None and r_idx is not None:
            pred_discovery_time = np.nanmin([np.nanmin(g_pred_times), np.nanmin(r_pred_times)])
        elif g_idx is not None:
            pred_discovery_time = np.nanmin(g_pred_times)
        elif r_idx is not None:
            pred_discovery_time = np.nanmin(r_pred_times)
        else:
            pred_discovery_time = None

        g_pred_times = g_pred_times - pred_discovery_time if pred_discovery_time is not None else np.array([])
        r_pred_times = r_pred_times - pred_discovery_time if pred_discovery_time is not None else np.array([])

        if g_idx is not None and r_idx is not None:
            g_pred_mag = self.predictions[0][g_idx]
            r_pred_mag = self.predictions[1][r_idx] 
        elif g_idx is not None:
            g_pred_mag = self.predictions[0][g_idx]
            r_pred_mag = []
        elif r_idx is not None:
            r_pred_mag = self.predictions[0][r_idx]
            g_pred_mag = []
        else:
            g_pred_mag = []
            r_pred_mag = []


        g_pred_mag_ = np.array(g_pred_mag) if g_idx is not None else np.array([])
        r_pred_mag_ = np.array(r_pred_mag) if r_idx is not None else np.array([])

        g_pred_num = len(g_pred_times) if g_idx is not None else 0
        r_pred_num = len(r_pred_times) if r_idx is not None else 0

        # np.random.seed(None)
        g_pred_mag_ = np.array([g_pred_mag_[i] for i in range(g_pred_num)]) if g_idx is not None else np.array([])
        r_pred_mag_ = np.array([r_pred_mag_[i] for i in range(r_pred_num)]) if r_idx is not None else np.array([])


        # convert to pandas dataframe
        upsampled_lc = pd.DataFrame({
            'time': np.concatenate([g_pred_times, r_pred_times]),
            'mag': np.concatenate([g_pred_mag_, r_pred_mag_]),
            'band': np.concatenate([np.repeat('ztfg', g_pred_num), np.repeat('ztfr', r_pred_num)])
        }) 
        
        converse_lc = self.apply_img_z_to_lc(upsampled_lc)


        if converse_lc is not None:
            converse_features = self.get_light_curve_statistics(converse_lc, min_detection = self.min_detection)
            
            if converse_features['g_phase'] is False and converse_features['r_phase'] is False:
                print(f"No more than min_detection detections (before peak) for z-shifted  {self.ztf_object} in g-band and r-band (g_num: {converse_features['g_prepeak_num']}, r_num: {converse_features['r_prepeak_num']})") 
                return None
            elif converse_features['g_peak'] is None and converse_features['r_peak'] is None:
                print(f'No peak for {self.ztf_object} in g-band and r-band')
                return None
            else:
                return converse_lc
        else:
            print(f"No upsampled light curve after z correctionfor {self.ztf_object}")
            return None

     
   
    def plot_light_curves(self, upsampled_lc):
        '''
        plot the original light curve, GP fitting, and the upsampled light curve, and the difference between them.
        '''

        def fitting_line(band, pred_times, predictions, prediction_uncertainties):
         
            # Plot GP fitting for g-band
            if band == 'g':
                c = 'green'
            else:
                c = 'red'
            plt.plot(
                pred_times, 
                predictions, 
                color=c, label=r'GP Fit ({band}-band)'
            )
            plt.fill_between(
                pred_times, 
                (predictions - prediction_uncertainties),
                (predictions + prediction_uncertainties),
                color=c, alpha=0.3, label=f'1σ Uncertainty ({band}-band)'
            )

        if not self.valid_lc:
            print(f"No valid light curve for {self.ztf_object}")
            return False
            
        if upsampled_lc is None:
            print("upsampled_lc cannot be None")
            return False
            
        if not isinstance(upsampled_lc, pd.DataFrame):
            print("upsampled_lc must be a pandas DataFrame")
            return False
            
        required_cols = ['time', 'mag', 'mag_err', 'band']
        if not all(col in upsampled_lc.columns for col in required_cols):
            print(f"upsampled_lc missing required columns: {required_cols}")
            return False
        
        if self.predictions is None or self.prediction_uncertainties is None:
            print("GP predictions not available. Run fit_2d_gp first.")
            return False
        if self.lc_features is None:
            print("Light curve features not computed. Run get_light_curve_statistics first.")
            return False

        if self.lc_data is not None:
            plt.figure()
            plt.errorbar(self.lc_data[self.lc_data['band'] == 'ztfg']['time'], self.lc_data[self.lc_data['band'] == 'ztfg']['mag'], self.lc_data[self.lc_data['band'] == 'ztfg']['mag_err'], fmt='o', color = 'green', label='original g-band')
            plt.errorbar(self.lc_data[self.lc_data['band'] == 'ztfr']['time'], self.lc_data[self.lc_data['band'] == 'ztfr']['mag'], self.lc_data[self.lc_data['band'] == 'ztfr']['mag_err'], fmt='o', color = 'red', label='original r-band')
            if self.predictions.shape[0] == 2:
                fitting_line('g', self.pred_times, self.predictions[0], self.prediction_uncertainties[0])
                fitting_line('r', self.pred_times, self.predictions[1], self.prediction_uncertainties[1])
            elif self.lc_features['g_num'] > 0:
                fitting_line('g', self.pred_times, self.predictions[0], self.prediction_uncertainties[0])
            elif self.lc_features['r_num'] > 0:
                fitting_line('r', self.pred_times, self.predictions[0], self.prediction_uncertainties[0])

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.gca().invert_yaxis() 
            plt.xlabel('Time (Days)')
            plt.ylabel('Magnitude')
            plt.title(f'{self.ztf_object} original light curve')
            plt.show()
            

        plt.figure()
        plt.errorbar(upsampled_lc[upsampled_lc['band'] == 'ztfg']['time'], upsampled_lc[upsampled_lc['band'] == 'ztfg']['mag'], upsampled_lc[upsampled_lc['band'] == 'ztfg']['mag_err'], fmt='o', color = 'blue',label='reconstructed g-band')
        plt.errorbar(upsampled_lc[upsampled_lc['band'] == 'ztfr']['time'], upsampled_lc[upsampled_lc['band'] == 'ztfr']['mag'], upsampled_lc[upsampled_lc['band'] == 'ztfr']['mag_err'], fmt= 'o', color = 'orange', label='reconstructed r-band')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gca().invert_yaxis() 
        plt.xlabel('Time (Days)')
        plt.ylabel('Magnitude')
        plt.title(f'{self.ztf_object} reconstructed light curve')
        plt.show()

        return True
    
    def get_needle_host_meta(self, only_complete = True):
        """
        Add host galaxy metadata from a CSV file.

        Parameters:
        - obj: str, object identifier
        - host_path: str, path to the host metadata files
        - only_complete: bool, whether to include only complete metadata

        Returns:
        - list, host metadata for the object or None if not found
        """
        def add_mag(host_dict, band):
            return host_dict.get(f'{band}Ap') or host_dict.get(f'{band}PSF') or None
        
        host_dict = self.matched_host_data
        if host_dict is not None:
            h_row = [add_mag(host_dict, band) for band in ['g', 'r', 'i', 'z', 'y', 'g-r_', 'r-i_']]
            
            return h_row if all(h_row) or not only_complete else None
        return None if only_complete else [None] * 7   


  
    def get_needle_meta(self, upsampled_lc):

        def find_peak_mag(upsampled_lc):
            if len(upsampled_lc) >= 1:

                # Extract mag, mag_err, time into a NumPy array
                mags = np.array([
                    [upsampled_lc.iloc[m]["mag"], upsampled_lc.iloc[m]["mag_err"], upsampled_lc.iloc[m]["time"]]
                    for m in range(len(upsampled_lc))
                ])
         
                # Find the index of the minimum magnitude (i.e., peak brightness)
                idx = np.argmin(mags[:, 0])
                peak_mag = mags[idx][0]
                peak_mjd = mags[idx][2]
                if peak_mag == 0 or peak_mag is None:
                    return None, None, None
                else:
                    return float(peak_mjd), float(peak_mag), int(idx)
            else:
                return None, None, None
          

        candids_g = upsampled_lc[upsampled_lc['band'] == 'ztfg']
        candids_g = candids_g.sort_values(by='time').reset_index(drop=True)
        candids_r = upsampled_lc[upsampled_lc['band'] == 'ztfr']
        candids_r = candids_r.sort_values(by='time').reset_index(drop=True)
        disdate = upsampled_lc['time'].min()
  
        peak_mjd_r, peak_mag_r, idx_r = find_peak_mag(candids_r)
        peak_mjd_g, peak_mag_g, idx_g = find_peak_mag(candids_g)
        meta_r = None
        meta_mixed = None
        flag_r = False
        flag_g = False
        host_g, host_r = None, None
        find_host = False
        host_meta = None
        sherlock_meta = None

        
        if self.matched_host_data is not None: # use NEEDLE-TH model
            host_g = self.matched_host_data['gAp']
            host_r = self.matched_host_data['rAp']
            sherlock_meta = self.matched_host_data['offset']
            find_host = True
            host_meta = self.get_needle_host_meta()
        else: # use NEEDLE-T model
            find_host = False
            host_g, host_r, sherlock_meta = None, None, None

        # get meta_mixed
        if peak_mag_r is not None and peak_mag_g is not None and peak_mjd_r is not None and peak_mjd_g is not None:
            peak_mag_g_minus_r = peak_mag_g - peak_mag_r
            peak_t_g_minus_r = peak_mjd_g - peak_mjd_r
            meta_data_g = self.get_obj_peak_meta(candids_g, idx_g, disdate, host_g, True)
            meta_data_r = self.get_obj_peak_meta(candids_r, idx_r, disdate, host_r, True)
            if meta_data_r is None:
                flag_r = False
            else:
                flag_r = True
            if meta_data_g is None:
                flag_g = False
            else:
                flag_g = True
      
            
        elif peak_mag_r is not None and peak_mag_g is None:
            peak_mag_g_minus_r, peak_t_g_minus_r = 0., 0.
            meta_data_r = self.get_obj_peak_meta(candids_r, idx_r, disdate, host_r, True)
            if meta_data_r is None:
                flag_r = False
            else:
                flag_r = True
            meta_data_g = [0.] * len(meta_data_r)
    

        elif peak_mag_g is not None and peak_mag_r is None:
            peak_mag_g_minus_r, peak_t_g_minus_r = 0., 0.
            meta_data_g = self.get_obj_peak_meta(candids_g, idx_g, disdate, host_g, True)
            meta_data_r = [0.] * len(meta_data_g)

        if find_host:
            # print('TEST meta_data_r: ', meta_data_r, np.nanmean(meta_data_r), np.nanstd(meta_data_r))
            # print('TEST meta_data_g: ', meta_data_g, np.nanmean(meta_data_g), np.nanstd(meta_data_g))
            # print('TEST host_meta: ', host_meta, np.nanmean(host_meta), np.nanstd(host_meta))
            # print('TEST peak_mag_g_minus_r: ', peak_mag_g_minus_r, np.nanmean(peak_mag_g_minus_r), np.nanstd(peak_mag_g_minus_r))
            # print('TEST peak_t_g_minus_r: ', peak_t_g_minus_r, np.nanmean(peak_t_g_minus_r), np.nanstd(peak_t_g_minus_r))

            meta_mixed = meta_data_r + meta_data_g + [peak_mag_g_minus_r, peak_t_g_minus_r] + host_meta + [sherlock_meta]
        else:
            meta_mixed = meta_data_r + meta_data_g + [peak_mag_g_minus_r, peak_t_g_minus_r]


        # get meta_r
        if flag_r:
            meta_data_r = self.get_obj_peak_meta(candids_r, idx_r, disdate, host_r, False)
            if meta_data_r is None:
                return None, None, None
            if find_host:
                meta_r = meta_data_r + host_meta + [sherlock_meta]
            else:
                meta_r = meta_data_r
        

        return  meta_r, meta_mixed, find_host

        
    def get_obj_peak_meta(self, candidates, candi_idx, disc_mjd, host_mag, for_mixed = False):
   
        def get_ratio(delta_mag_recent, delta_t_recent, delta_mag_disc, delta_t_disc):
            """
            Calculate the ratios of delta magnitude to delta time for recent and discovery values.

            Parameters:
            - delta_mag_recent: float, recent delta magnitude
            - delta_t_recent: float, recent delta time
            - delta_mag_disc: float, discovery delta magnitude
            - delta_t_disc: float, discovery delta time

            Returns:
            - tuple of floats, (ratio_recent, ratio_disc)
            """
            if isinstance(delta_mag_disc, np.ndarray):
                return (
                    np.divide(delta_mag_recent, delta_t_recent, out=np.zeros_like(delta_mag_recent), where=delta_t_recent != 0),
                    np.divide(delta_mag_disc, delta_t_disc, out=np.zeros_like(delta_mag_disc), where=delta_t_disc != 0)
                )
            else:
                if delta_t_disc == 0.0 or delta_t_recent == 0.0:
                    return 0.0, 0.0
                return delta_mag_recent / delta_t_recent, delta_mag_disc / delta_t_disc

        candi_mag = candidates.iloc[candi_idx]['mag']# peak mag 
        
        disc_idx = candidates['time'].argmin()
        disc_band_mag = candidates.iloc[disc_idx]['mag']

        
        delta_t_discovery_band = round(candidates.iloc[candi_idx]['time'] - candidates.iloc[disc_idx]['time'], 5)
        delta_t_discovery = round(candidates.iloc[candi_idx]['time'] - disc_mjd, 5)
        delta_mag_discovery = round(candi_mag - disc_band_mag, 5)

        if candi_idx < len(candidates) - 1: 
            delta_t_recent = round(candidates.iloc[candi_idx]['time'] - candidates.iloc[candi_idx + 1]['time'], 5)
            delta_mag_recent = round(candi_mag - candidates.iloc[candi_idx + 1]['mag'], 5)
        else:
            delta_t_recent = delta_t_discovery
            delta_mag_recent = delta_mag_discovery

        ratio_recent, ratio_disc = get_ratio(delta_mag_recent, delta_t_recent, delta_mag_discovery, delta_t_discovery_band)
        if for_mixed:
            row = [candi_mag, disc_band_mag, delta_mag_discovery, delta_t_discovery_band, delta_t_discovery, ratio_recent, ratio_disc]
        else: # for meta_r
            row = [candi_mag, disc_band_mag, delta_mag_discovery, delta_t_discovery, ratio_recent, ratio_disc]

        if host_mag is not None:
            delta_host_mag = round(candi_mag - host_mag, 5)
            row += [delta_host_mag]

        return row




    def display(self):
        '''
        display the original light curve, GP fitting, and the upsampled light curve, and the difference between them.
        '''
        upsampled_lc = self.upsample_light_curve(seed = 1)
        self.plot_light_curves(upsampled_lc)

    
    # def save_light_curve(self):
    #     '''
    #     save the light curve to a file
    #     '''
    #     if not os.path.exists(self.__output_dir):
    #         os.makedirs(self.__output_dir, exist_ok=True)
    #         if self.valid_lc:
    #             photo_dict = {}
    #             photo_dict['mean_g'] = self.mean_g
    #             photo_dict['mean_r'] = self.mean_r
    #             photo_dict['std_g'] = self.std_g
    #             photo_dict['std_r'] = self.std_r
    #             photo_dict['lc_features'] = self.lc_features
    #             np.save(self.__output_dir + '/photo_dict.npy', photo_dict)
    #             print(f'Light curve saved for {self.ztf_object} on {self.__output_dir}')
    #         else:
    #             print(f'No valid light curve for {self.ztf_object}')
    #     else:
    #         print(f'Light curve already exist for {self.ztf_object} on {self.__output_dir}')




    def save_gp_fitting(self): 
        '''
        save the GP fitting results to a file
        ''' 
        # print('DEBUG: SAVE: ',self.predictions)
        if not os.path.exists(self.__output_dir):
            os.makedirs(self.__output_dir, exist_ok=True)
            if self.predictions is not None and self.valid_lc:
                photo_dict = {}
                photo_dict['mean_g'] = self.mean_g
                photo_dict['mean_r'] = self.mean_r
                photo_dict['std_g'] = self.std_g
                photo_dict['std_r'] = self.std_r
                photo_dict['lc_features'] = self.lc_features
                photo_dict['predictions'] = self.predictions
                photo_dict['pred_times'] = self.pred_times
                photo_dict['pred_uncertainties'] = self.prediction_uncertainties
                np.save(self.__output_dir + '/photo_dict.npy', photo_dict)
                print(f'GP fitting results saved for {self.ztf_object} on {self.__output_dir}')
            else:
                print(f'No good GP fitting for {self.ztf_object}')
        else:
            print(f'GP fitting results already exist for {self.ztf_object} on {self.__output_dir}')
       
            
  

    def load_gp_fitting(self, input_path = PHOTO_OUTPUT_PATH): 
        '''
        load the GP fitting results from a file
        '''
        gp_file = os.path.join(input_path, self.ztf_object, 'photo_dict.npy')
        if os.path.exists(gp_file):
            photo_dict = np.load(gp_file, allow_pickle=True).item()
            self.predictions = photo_dict['predictions']
            self.pred_times = photo_dict['pred_times']
            self.prediction_uncertainties = photo_dict['pred_uncertainties']
            self.lc_features = photo_dict['lc_features']
            self.mean_g = photo_dict['mean_g']
            self.mean_r = photo_dict['mean_r']
            self.std_g = photo_dict['std_g']
            self.std_r = photo_dict['std_r']
            self.valid_lc = self.lc_features['g_phase'] or self.lc_features['r_phase']
            print(f'GP fitting results loaded for {self.ztf_object}')
            # print('DEBUG: LOAD: ',self.predictions)
            return True
        else:
            print(f'No GP fitting results found for {self.ztf_object}')
            return False




def process_objs(obj):
    if not os.path.exists(PHOTO_OUTPUT_PATH + '/' + obj):
        lc = LightCurveUpsamplingPipeline(ztf_object = obj, gp_fitting = True, min_detection = 2, load_gp = False)
        lc.save_gp_fitting()
    else:
        print(f'results already exist for {obj} on {PHOTO_OUTPUT_PATH}')
    


# def process_SN(obj):
#     if not os.path.exists(PHOTO_OUTPUT_PATH + '/' + obj):
#         lc = LightCurveUpsamplingPipeline(ztf_object = obj, gp_fitting = False, min_detection = 1)

#     else:
#         print(f' results already exist for {obj}')


if __name__ == '__main__':

    ztf_obj_info_path = '../info/ztf_train_valid_set.csv'
    info_df = pd.read_csv(ztf_obj_info_path)

    obj_list = info_df['ZTFID'].values.tolist()

    print(len(obj_list))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_objs, obj_list)
