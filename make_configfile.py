import configparser

config = configparser.ConfigParser()

config.add_section('directories')
config.set('directories', 'local_top', '/Users/dml/Data/EIS')
config.set('directories', 'dens_dir', '/Users/dml/python_output/FIP/density')
config.set('directories', 'main_dir', '/Users/dml/python_output/FIP/')

with open(r"configfile.ini", 'w') as configfile:
    config.write(configfile)
