

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)

df_ = pd.read_excel("/kaggle/input/online-retail-ii-uci/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()

df.info()

df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = df.loc[(df["StockCode"] != "POST")]

df_germany = df[df['Country'] == "Germany"]

df_germany.head()

df_germany.shape

"""2) Germany Müşterileri Üzerinden Birliktelik Kuralları Üretiniz"""

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

ger_inv_pro_df = create_invoice_product_df(df_germany, id=True)

ger_inv_pro_df.head()

frequent_itemsets = apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

"""3) ID'leri Verilem Ürünlerin İsimleri Nelerdir?"""

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_germany, 21987)
# ['PACK OF 6 SKULL PAPER CUPS']

check_id(df_germany, 23235)
# ['STORAGE TIN VINTAGE LEAF']

check_id(df_germany, 22747)
# ["POPPY'S PLAYHOUSE BATHROOM"]

"""4) Sepetteki Kullanıcılar için Ürün Önerisi Yapınız

* Kullanıcı 1 ürün id'si: 21987
* Kullanıcı 2 ürün id'si: 23235
* Kullanıcı 3 ürün id'si: 22747

- antecedent support = ilk ürünün tek başına gözlenme olasılığı

- consequent support = son ürünün tek başına gözlenme olasılığı

- Support (X,Y) = Freq (X,Y)/N, range: [0,1]
- N : Bütün gözlem sayısı
- X ve Y'nin birlikte görülme olasılığı. Örnek: Yumurta ve çay, tüm alışverişlerin %40'ında birlikte gözlenmektedir.

- Confidence (X,Y) = Freq (X,Y) / Freq (X), range: [0,1]
- X ürünü satın alındığında, Y ürününün satılması olasılığı
- Confidence (Ekmek, Süt) = Freq(Ekmek, Süt) / Freq(Süt)      => Ekmek satın alındığında, süt satın alınması olasılığı.
- Örnek: Yumurta alan müşterilerin %67'si çay da almaktadır.

- Lift = Support (X,Y) / (Support (X) * Support (Y)), range: [0,∞]
- X ürünü satın alındığında, Y ürününün satın alınma olasılığı lift kat kadar artıyor.
- Eğer hesaplanan değer 1'den büyük çıkarsa, bunlar birbirlerine bağımlıdır, birbirlerini etkiliyorlardır yorumu yapılabilir.
- Örnek: Yumurta olan alışverişlerde çay ürününün satışı 1,11 kat artmaktadır.

- Leverage = levarage(X, Y)= support(X, Y) − (support(X) * support(Y)), range: [−1,1]
- Birlikte görülen X ve Y'nin gözlenme frekansı ile X ve Y bağımsız olsaydı beklenen frekans arasındaki farkı hesaplar.
- 0 leverage değeri, bağımsızlığı gösterir.

- Conviction = conviction(X,Y)= (1−support(Y)) / (1−confidence(X,Y)), range: [0,∞]
- Yüksek bir conviction değeri, sonucun büyük ölçüde antecedents'a bağlı olduğu anlamına gelir.
- Örneğin, mükemmel bir confidence puanı durumunda, payda 0 olur (1 - 1 nedeniyle) ve bunun için conviction puanı 'inf' olarak tanımlanır.
- Lift'e benzer şekilde, öğeler bağımsızsa, conviction 1'dir.
"""

frequent_itemsets.sort_values("support", ascending=False).head()

rules.sort_values("support", ascending=False).head()

rules.sort_values("lift", ascending=False).head()

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():

        for j in list(product):
            if j == product_id:
                for element in list(sorted_rules.loc[i]["consequents"]):
                    if element not in recommendation_list:
                        recommendation_list.append(element)
                        
                      
    return recommendation_list[:rec_count]

arl_recommender(rules, 21987, 3)

arl_recommender(rules, 23235, 2)

arl_recommender(rules, 22747, 2)

# Görev 5 Önerilen Ürünlerin İsimleri Nelerdir?
check_id(df, arl_recommender(rules, 21987, 1)[0])

check_id(df, arl_recommender(rules, 23235, 1)[0])

check_id(df, arl_recommender(rules, 22747, 1)[0])