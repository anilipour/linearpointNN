from astropy.table import Table

def txt_to_table(sim_num, redshift): # converts text file of simulation to an astropy table
    simulation = f'latin_hypercube/{sim_num}/CF_m_1024_z={redshift}.txt' # input text file
    sim_list = []
    for line in open(simulation, 'r'): # get each line
        item = line.rstrip() # strip off newline and any other trailing whitespace
        sim_list.append(item)
        
    r_list, xi_list = [], []
    for item in sim_list: # get radius and correlation from each line
        r, xi = item.split() # each line has both radius and correlation, so split
        r_list.append(float(r))
        xi_list.append(float(xi))
        
    sim_table = Table([r_list, xi_list], names=('r', 'xi')) # astropy table
    return sim_table

def txt_to_table_fiducial(sim_num): # converts text file of simulation to an astropy table
    simulation = f'fiducialCF/{sim_num}/CF_m_z=127.txt' # input text file
    sim_list = []
    for line in open(simulation, 'r'): # get each line
        item = line.rstrip() # strip off newline and any other trailing whitespace
        sim_list.append(item)
        
    r_list, xi_list = [], []
    for item in sim_list: # get radius and correlation from each line
        r, xi = item.split() # each line has both radius and correlation, so split
        r_list.append(float(r))
        xi_list.append(float(xi))
        
    sim_table = Table([r_list, xi_list], names=('r', 'xi')) # astropy table
    return sim_table

def txt_to_table_fiducial_z0(sim_num): # converts text file of simulation to an astropy table
    simulation = f'fiducialCFz0/{sim_num}/CF_m_z=0.txt' # input text file
    sim_list = []
    for line in open(simulation, 'r'): # get each line
        item = line.rstrip() # strip off newline and any other trailing whitespace
        sim_list.append(item)
        
    r_list, xi_list = [], []
    for item in sim_list: # get radius and correlation from each line
        r, xi = item.split() # each line has both radius and correlation, so split
        r_list.append(float(r))
        xi_list.append(float(xi))
        
    sim_table = Table([r_list, xi_list], names=('r', 'xi')) # astropy table
    return sim_table