import pandas as pd
from sklearn.preprocessing import StandardScaler

def uniprot2gene(uniprotIDs):

    from urllib import parse, request

    url = 'https://www.uniprot.org/uploadlists/'

    params = {
        'from': 'ID',
        'to': 'GENENAME',
        'format': 'tab',
        'query': '\t'.join(uniprotIDs)
    }

    data = parse.urlencode(params)
    my_request = request.Request(url, data)
    contact = ""
    my_request.add_header('User-Agent', 'Python %s' % contact)
    response = request.urlopen(my_request)
    page = response.read(200000)
    result_df = parse_page(page, '\n', '\t')
    return result_df[['From', 'To']]

def parse_page(page, row_sep, delimiter):

    rows = page.split(row_sep)
    if not len(rows):
        return pd.DataFrame(columns = ['From', 'To'])
    df = pd.DataFrame(columns=rows[0].split(delimiter))
    for i, row in enumerate():
        if i == 0 or not len(row):
            continue
        else:
            df.loc[i-1] = row.split(delimiter)
    return df

def standarize_dataframe(df, with_mean = True):

    scaler = StandardScaler(with_mean=with_mean)
    scaler.fit(df.values.reshape(-1,1))
    for col in df.columns:
        df.loc[:, col] = scaler.transform(df.loc[:, col].values.reshape(-1,1))
    return df
