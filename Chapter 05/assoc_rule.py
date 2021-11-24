import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
# Read the online retail transaction data of purchases
df = pd.read_excel('./OnlineRetail.xlsx')
print ("Completed reading the xlsx file")
print (df.head())
print ("Head of the xlsx file")

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

basket = (df[df['Country'] =="EIRE"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

print (basket.head())
print (basket)
basket.to_csv("country.csv",)
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
#basket_sets.drop('POSTAGE', inplace=True, axis=1)

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print ("The rules are given below: ")
print (rules.head())

rules.to_csv("rules.csv")

print (rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ])

print (basket['PINK REGENCY TEACUP AND SAUCER'].sum())
print (basket['GREEN REGENCY TEACUP AND SAUCER'].sum())
print (basket['ROSES REGENCY TEACUP AND SAUCER'].sum())

print (basket['ALARM CLOCK BAKELIKE GREEN'].sum())

print (basket['ALARM CLOCK BAKELIKE RED'].sum())

basket2 = (df[df['Country'] =="Australia"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets2 = basket2.applymap(encode_units)
#basket_sets2.drop('DOTCOM%20POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)

frequent_itemsets2.to_csv("Aus.csv")
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

rules2.to_csv("rules2.csv")
print (rules2[ (rules2['lift'] >= 8) &
        (rules2['confidence'] >= 0.9)])

