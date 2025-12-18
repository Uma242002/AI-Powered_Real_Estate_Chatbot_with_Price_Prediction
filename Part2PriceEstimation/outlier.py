import plotly.express as px

def outlier_detect(df):
    cols = []
    for col in df.describe().columns:
        print("Column:",col)
        print("------------------------------------------------")
        print("Boxplot For Outlier Identification:")
        px.box(df[col], orientation='h', width=600, height=300, ).show()
        print()
        Q1 = df.describe().at['25%',col]
        Q3 = df.describe().at['75%',col]
        IQR = Q3 - Q1
        lowerbound = Q1 - 1.5 * IQR
        upperbound = Q3 + 1.5 * IQR
        
        print("********* Outlier Data Points *******")
        print()
        lowerout = []
        upperout = []

        for val in df[col]:
            if val<lowerbound:
                if val not in lowerout:
                    lowerout.append(val)
            elif val>upperbound:
                if val not in upperout:
                    upperout.append(val)

        lowerout.sort()
        upperout.sort()

        print("Lower Outliers:")
        print(lowerout)
        print()
        print()
        print("Upper Outliers:")
        print(upperout)
        print()
        print("===============================================")
        print()
        
        if lowerout!=[] or upperout!=[]:
            cols.append(col)
      
    return cols
        
def outlier_replacement(df, cols):
    for col in cols:
        print("Column:",col)
        print("------------------------------------------------")
        Q1 = df.describe().at['25%',col]
        Q3 = df.describe().at['75%',col]
        IQR = Q3 - Q1
        LTV = Q1 - 1.5 * IQR
        UTV = Q3 + 1.5 * IQR
        
        # replacement vals (any one of the below)
        
        # 1. Median
        median = df[col].median()
        
        # 2. Ltv, Utv
        low_bound = LTV
        high_bound = UTV
        
        # 3. 5th & 95th (Suggested)
        fifth = df[col].quantile(0.05)
        ninetyfifth = df[col].quantile(0.95)

        print("Replacing Outliers with 5th percentile for lower Outliers, 95th percentile for Upper Outliers....")
        print("Adjust the module code for any other replacements.........")
        print()
        
        # mask method is used to replace the values
        df[col] = df[col].mask(df[col]<LTV, round(fifth)) # replacing the lower outlier with 5th percentile value
        df[col] = df[col].mask(df[col]>UTV, round(ninetyfifth)) # replacing the outlier with 95th percentile value