import pdb
import pandas as pd
import os


class HELPER:
    def __init__(self):
        DATA_DIR = './Data'
        if not os.path.isdir(DATA_DIR):
            DATA_DIR  = "../resource/asnlib/publicdata/data"

        self.DATA_DIR = DATA_DIR
        
        return

    def attrRename(self, df, ticker):
        """
        Rename attributes of DataFrame
        - prepend the string "T_" to original attribute name, where T is the string of ticker
        """
        rename_map = { orig:  ticker + "_" + orig.replace(" ", "_") for orig in df.columns.to_list() }
        
        return df.rename(columns=rename_map)

    def getData(self, tickers, indx, attrs):
        """
        Return DataFrame with data for a list of tickers plus and index
        
        Parameters
        ----------
        tickers: List
        - List of tickers
        
        indx: String
        - Ticker of index
        
        DATA_DIR: String
        - Directory of data
        
        attrs: List
        - List of data attributes to retain
        
        Returns
        -------
        DataFrame
        - attributes:
        -- each original attribute, prepended with "T_" where T is ticker or index string
        -- This is necessary to distinguish between attributes of different tickers
        """
        DATA_DIR = self.DATA_DIR
        
        dateAttr = "Dt"
        
        use_cols =  attrs.copy()
        use_cols.insert(0, dateAttr)
       
        # Read the CSV files
        dfs = []
        for ticker_num, ticker in enumerate(tickers):
            ticker_file = os.path.join(DATA_DIR, "{t}.csv".format(t=ticker) )
            ticker_df = pd.read_csv(ticker_file, index_col=dateAttr, usecols=use_cols)
            
            # Rename attributes with ticker name
            ticker_df = self.attrRename(ticker_df, ticker)
            
            dfs.append(ticker_df)
            
              
        index_file   = os.path.join(DATA_DIR, "{t}.csv".format(t=indx) )
        index_df   = pd.read_csv(index_file, index_col=dateAttr, usecols=use_cols)
        index_df = self.attrRename(index_df, indx)
        
        dfs.append(index_df)
        
        data_df = pd.concat( dfs, axis=1)
       
        return data_df

    def renamePriceToRet(self, df, priceAttr = "Adj Close"):
        rename_map = { }
        rename_map = { orig:  orig.replace( priceAttr.replace(" ", "_"), "Ret") for orig in df.columns.to_list() }
        
        return df.rename(columns = rename_map)
