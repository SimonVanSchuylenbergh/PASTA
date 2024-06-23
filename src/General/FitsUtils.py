from astropy.io import fits # type: ignore


def print_header(file):
    image = fits.open(file)
    for key in image[0].header:
        print(key, "\t", image[0].header[key])


def write_header_to_file(file, out_file):
    image = fits.open(file)
    outfile = open(out_file, "w")
    for key in image[0].header:
        outfile.write(str(key) + "\t" + str(image[0].header[key]) + "\n")
    outfile.close()
