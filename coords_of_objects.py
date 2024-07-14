from astropy.io import fits
from glob import glob
from tqdm import tqdm

files = glob('BOSS_spectra/*.fits')
outfile = open('coords.txt', 'w')

for file in tqdm(files):
    file = fits.open(file)
    outfile.write(str(file[0].header['PLUG_RA']) + ' ' + str(file[0].header['PLUG_DEC']) + '\n')

outfile.close()
