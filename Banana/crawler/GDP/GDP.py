from selenium import webdriver
import time
from selenium.webdriver.support.ui import Select
import pandas as pd
import MySQLdb
import datetime


# 連接資料庫
def connect_database():

    # 連接我的資料庫
    # db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='root', db='banana_database', port=3306, charset='utf8')
    # 連接Paul的資料庫
    db = MySQLdb.connect(host='127.0.0.1', user='dbuser', passwd='20200428', db='fruveg', port=3307, charset='utf8')

    cursor = db.cursor()
    db.autocommit(True)
    return cursor


# 得到網頁的函式
def get_the_page(url, start_date, end_date):

    driver = webdriver.Chrome('../chromedriver.exe')
    driver.get(url)
    time.sleep(2)

    # 進到國民所得重要指標
    driver.find_element_by_id('ContentPlaceHolder1_repKind_repMiddleKind_0_btnMiddleKind_0').click()

    # 週期選季
    select = Select(driver.find_element_by_id('ContentPlaceHolder1_ddlPeriod'))
    select.select_by_value('Q')
    time.sleep(1)

    # 起始日期選 101年 Q1
    select = Select(driver.find_element_by_id('ContentPlaceHolder1_ddlDateBeg'))
    select.select_by_value(start_date)
    time.sleep(1)

    # 結束日期選 108年 Q4
    select = Select(driver.find_element_by_id('ContentPlaceHolder1_ddlDateEnd'))
    select.select_by_value(end_date)

    driver.find_element_by_id('ContentPlaceHolder1_btnQuery').click()

    src = pd.read_html(driver.page_source)[3]
    driver.quit()

    return src


# 從網頁拿出資料
def get_data(src):

    df = src.iloc[4:36].drop(columns=[2, 11]).iloc[:, [0, 1, 2, 3]]
    df.columns = ['年', '季', 'GDP', '經濟成長率']

    year_list = convert_year(list(df['年']))
    season_list = list(df['季'])
    gdp_list = clean_gdp(list(df['GDP']))
    egr_list = list(df['經濟成長率'])
    start_date_list, end_date_list = new_date_column(year_list=year_list, season_list=season_list)

    return year_list, season_list, egr_list, gdp_list, start_date_list, end_date_list


# 民國轉西元
def convert_year(year_list):
    year = 2011
    for index in range(len(year_list)):
        if (index % 4) != 0:
            year_list[index] = year
        else:
            m_year = int(year_list[index][:-1]) + 1911
            year_list[index] = m_year
            year += 1
    return year_list


# 去除GDP裡的空格
def clean_gdp(gdp_list):

    g_list = []
    for gdp in gdp_list:
        g_list.append(int(gdp.replace(' ', '')))
    return g_list


# 新生成 start_date, end_date 的函式
def new_date_column(year_list, season_list):

    start_date_list = []
    end_date_list = []

    for index in range(len(season_list)):
        if season_list[index] == '1季':
            start_date_list.append(datetime.date(year_list[index], 1, 1))
            end_date_list.append(datetime.date(year_list[index], 3, 31))

        elif season_list[index] == '2季':
            start_date_list.append(datetime.date(year_list[index], 4, 1))
            end_date_list.append(datetime.date(year_list[index], 8, 30))

        elif season_list[index] == '3季':
            start_date_list.append(datetime.date(year_list[index], 7, 1))
            end_date_list.append(datetime.date(year_list[index], 9, 30))

        elif season_list[index] == '4季':
            start_date_list.append(datetime.date(year_list[index], 10, 1))
            end_date_list.append(datetime.date(year_list[index], 12, 31))

    return start_date_list, end_date_list


# 將數據寫入資料庫的函式
def insert_into_database(cursor, year_list, season_list, egr_list, gdp_list, start_date_list, end_date_list):

    for i in range(len(gdp_list)):
        try:
            sql_str = f"INSERT INTO Adler_gdp	(annual, season, start_date, end_date, gdp, egr) VALUES " \
                      f"({year_list[i]},  \'{'第' + season_list[i]}\', \'{start_date_list[i]}\', " \
                      f"\'{end_date_list[i]}\',  {gdp_list[i]}, {egr_list[i]} ); "
            cursor.execute(sql_str)
            print('Done')
        except Exception as err:
            print(err.args)


def main(url, start_date, end_date):

    cursor = connect_database()

    src = get_the_page(url=url, start_date=start_date, end_date=end_date)

    year_list, season_list, egr_list, gdp_list, start_date_list, end_date_list = get_data(src=src)

    insert_into_database(cursor, year_list, season_list, egr_list, gdp_list, start_date_list, end_date_list)


if __name__ == '__main__':
    '''
    start: 從哪一年哪一季開始抓
    end: 到哪一年哪一季結束
    '''

    start = '101Q1'
    end = '108Q4'
    gdp_url = 'https://dmz26.moea.gov.tw/GMWeb/common/CommonQuery.aspx'

    main(url=gdp_url, start_date=start, end_date=end)