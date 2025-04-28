import pubchempy as pcp
import logging
from src import setting

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.run_specific_log, mode='a')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Processing chemicals")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

def smile2ichikey(smile):

    try:
        compounds = pcp.get_compounds(smile, namespace='smiles')
        if len(compounds) == 1:
            return compounds[0].inchikey

        else:
            logging.info("Found more than one inchikey")
            return [x.inchikey for x in compounds]

    except:
        return None


def smile2ichi(smile):

    try:
        compounds = pcp.get_compounds(smile, namespace='smiles')
        if len(compounds) == 1:
            return compounds[0].inchi

        else:
            logging.info("Found more than one inchikey")
            return [x.inchikey for x in compounds]

    except:
        return None
