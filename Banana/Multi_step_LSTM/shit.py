markets = {'Taipei1_price': ['109', 'models/price/Taipei1.h5'],
           'Taipei2_price': ['104', 'models/price/Taipei2.h5'],
           'Banqiao_price': ['220', 'models/price/Banqiao.h5'],
           'Sanchong_price': ['241', 'models/price/Sanchong.h5'],
           'Taichung_price': ['400', 'models/price/Taichung.h5'],
           'Fengyuan_price': ['420', 'models/price/Fengyuan.h5'],
           'Kaohsiung_price': ['800', 'models/price/Kaohsiung.h5'],
           'Fongshan_price': ['830', 'models/price/Fongshan.h5'],
           'Taipei1_amount': ['109', 'models/amount/Taipei1.h5'],
           'Taipei2_amount': ['104', 'models/amount/Taipei2.h5'],
           'Banqiao_amount': ['220', 'models/amount/Banqiao.h5'],
           'Sanchong_amount': ['241', 'models/amount/Sanchong.h5'],
           'Taichung_amount': ['400', 'models/amount/Taichung.h5'],
           'Fengyuan_amount': ['420', 'models/amount/Fengyuan.h5'],
           'Kaohsiung_amount': ['800', 'models/amount/Kaohsiung.h5'],
           'Fongshan_amount': ['830', 'models/amount/Fongshan.h5']}

for table_name in markets.keys():
    print(f'{table_name}, {markets[table_name][0]}, {markets[table_name][1]}')
