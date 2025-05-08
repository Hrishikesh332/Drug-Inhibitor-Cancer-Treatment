import logging

import pubchempy as pcp

import trans_synergy.settings

setting = trans_synergy.settings.get()

# Setting up log file
logger = logging.getLogger("Processing chemicals")
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
