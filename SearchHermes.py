from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


output_file = Path(__file__).parent / 'hermes_all.csv'

coords = np.loadtxt('coords.txt')


hermesobs = pd.read_csv('/STER/mercator/hermes/HermesFullDataOverview.tsv',sep='\t',skiprows=2)
hermesobs.columns = ['unseq', 'prog_id', 'obsmode', 'bvcor', 'observer', 'object', 'ra','dec',
                     'bjd', 'exptime', 'pmtotal', 'date-avg', 'airmass' , 'filename']

limit = 0.002777778 # this is 10 arcsec
conv = np.pi/180 # rad to deg


outfile = open(output_file, 'w')
outfile.write('unseq;object;ra;dec;exptime;airmass;night;ra_boss;dec_boss\n')
outfile.close()


last_obj = ''
last_obj_data = ''
last_was_found = True

for i in tqdm(range(len(coords))):
    ra = coords[i,0]
    dec = coords[i,1]
    distance=np.sqrt(((hermesobs['ra']-ra)*np.cos(ra*conv))**2 + (hermesobs['dec']-dec)**2)

    data_filtered = hermesobs.loc[(distance<limit) & (hermesobs['object']!='CALIBRATION')]

    # get a nights column so it's easier to save
    nights =[data_filtered['filename'].values[j].split('/')[4] for j in range(len(data_filtered))]
    data_filtered=data_filtered.assign(night=nights)

    last_was_found = len(data_filtered) >= 1
    if last_was_found:
        # We look for the best observation. This is a bit ambigous, but we want high exposure
        # time and low airmass. So I use a metric of exp_time / airmass^2 to decide.
        best_obs = 0
        best_obs_metric = 0
        for j in range(1, len(data_filtered)):
            try:
                metric = data_filtered['exptime'].values[j] / data_filtered['airmass'].values[j]**2
                if metric > best_obs_metric:
                    best_obs = j
                    best_obs_metric = metric
            except ValueError:
                pass

        last_obj_data = (str(data_filtered['unseq'].values[best_obs]) + ';' +
                         str(data_filtered['object'].values[best_obs]) + ';' +
                         str(data_filtered['ra'].values[best_obs]) + ';' +
                         str(data_filtered['dec'].values[best_obs]) + ';' +
                         str(data_filtered['exptime'].values[best_obs]) + ';' +
                         str(data_filtered['airmass'].values[best_obs]) + ';' +
                         str(data_filtered['night'].values[best_obs]) + ';')

        outfile = open(output_file, 'a')
        outfile.write(last_obj_data +
                      str(coords[i,0]) + ';' +
                      str(coords[i,1]) + '\n')
        outfile.close()

