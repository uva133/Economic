import os
import math
import numpy as np
from tkinter import *
from tkinter import ttk
import tkinter.filedialog
import matplotlib
import random as rnd
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from datetime import datetime, timedelta
from tkinter import messagebox
from numpy import array, arange, abs as np_abs
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.stats import norm
import openpyxl as ox 
import statistics
import warnings
import pylab

class Cls_task_:             # класс для решения тестовых заданий
    def __init__(self, b_mode):
        
        self.path_file   = ""  # создание глобальной переменной для хранения информации о файле
        self.file_name   = ""  # имя файла исходных данных
        self.b_read_file = False
        # переменные для решения задач по тестам
        self.mas_X = []
        self.mas_Y = []
        self.cIMFs  = []
        self.total_imf  = 0
        self.mas_regr_x = []
        self.mas_regr_y = []
        self.mas_noise = []
        self.mas_x_level = []
        self.mas_up = []
        self.mas_down = []
        self.fittedParameters = []
    def Read_task_1(self):
        try:
            wb = ox.Workbook()
            self.mas_X.clear()
            self.mas_Y.clear()
                        
            i_line = 0
            fn = tkinter.filedialog.Open(root, title = "Выберите исходный временной ряд ", filetypes = [('*.xlsx files', '*.xlsx')]).show()
            path_ref = fn
            if fn == '':
                return
            
            textbox.delete('1.0', 'end') 
                    
            wb = ox.load_workbook(path_ref)
            sheet_ranges = wb['Лист1']

            for i in range(1, 10):
                st = str(sheet_ranges['A'+ str(i)].value) + "; " + str(sheet_ranges['B'+ str(i)].value) + "\n"
                textbox.insert(str(i + 1.0), st)
                n = textbox.size
            
            textbox.insert(str(1.0), path_ref + '\n')
            textbox.insert(str(1.0), 'Прочитано 10 строк из файла ...\n')

            f_name = path_ref.rfind('/')

            for i in range(2, 716):
                rows = sheet_ranges['A3:'+'E3']
                R_value   = float(sheet_ranges['B'+ str(i)].value)
                R_time    = sheet_ranges['A'+ str(i)].value
                if (R_value == None):
                    root.config(cursor="")
                    break

                if (R_value != None):
                    self.mas_Y.append(R_value)
                    self.mas_X.append(R_time)
                   
            fig, ax = plt.subplots()
            self.b_read_file = True
            self.path_file = path_ref
            self.file_name = os.path.basename(path_ref)
            plt.subplot (1,1,1)
            plt.plot( self.mas_X , self.mas_Y, label = 'Котировки газа'  )  # Файл по времени нельзя 
            plt.title('Котировки газа')

            plt.grid()
            plt.show()
        except BaseException:
            tkinter.messagebox.showinfo("Title", "a Tk MessageBox")
    
    def Calc_Black_Shoulsh(self):
        # Решение задачи 1
        try:
            format = '%d.%m.%Y'
            if (self.b_read_file == FALSE):
                self.Read_task_1()

            value_start = datetime.strptime(v_start.get(), format)
            value_stop = datetime.strptime(v_stop.get(), format)                   
            value_period = (value_stop - value_start).days
            '''
            value_add = 2 # v_delta.get()
            value_min = 260 #float(v_min.get())
            '''
            # РАСЧЕТ ВОЛАТИЛЬНОСТИ
            std_signal = np.std(self.mas_Y)
            # среднегодовая волатильность
            min_date = min(self.mas_X)
            max_date = max(self.mas_X)
            delta_date = (max_date - min_date).days  # интервал в днях
            delta_year = delta_date/365
            std_year = std_signal/ (math.sqrt(delta_year))  # среднегодовая волатильность в абсолютных единицах
            mean = statistics.mean(self.mas_Y)    # среднее значение цены 
            
            std_year_proc = std_year / mean  # среднегодовая волатильность в процентах
            std_day = std_year/(math.sqrt(252))   # среднесуточная волатильность в абсолютных единицах
            
            price  = self.mas_Y[len(self.mas_Y)-1] # - текущая цена
            #strike = 260000 # - страйк опциона (цена к исполнению)
            Volume_buy = 1000 # обьем базовой закупки
            strike = Call_Strice(price, 2, 260)   # определение цены исполения 
            
            volatility = 0.3 #  0.3 # std_year_proc - волатильность  (ско)
            yearsToExpiry = 1/365 # - cрок опциона, лет
            riskFreeRate = 0.06 #- Безрисковая ставка
            N_iter = 1000  # количество итераций для поиска решения методом Монте-Карло
            N_day = value_period # 2      # количество покупки инструмента
            callp = CallPremiumBS(price,strike,volatility,yearsToExpiry,riskFreeRate) # Расчет премии методом Блэка - Шоулса + работает правильно

            # Метод Блэка - Шоулса
            mas_BS = [] # массив премии по методу Блэка - Шоулса
            mas_day = []
            #mas_BS.append(0)
            for  j in range(1, N_day):
                callp = Volume_buy * CallPremiumBS(price,strike,volatility,yearsToExpiry*j,riskFreeRate) # Расчет премии методом Блэка - Шоулса
                mas_BS.append(callp)
                mas_day.append(value_start + timedelta(days=j))
            
            i = 0
            fig, ax = plt.subplots()
                        
            plt.subplot (2,1,1)
            plt.plot( self.mas_X , self.mas_Y, label = 'Стоимость поставки', marker=".", markersize = 1  )  # Файл по времени нельзя 
            #for i in range(0, 50): #N_iter
            #    plt.plot( mas_day , mas_total_price[i], label = 'Стоимость поставки' )  # Файл по времени нельзя 
            plt.title('Спотовая цена актива ')
            
            plt.grid()

            price_total = np.sum(mas_BS)
            plt.subplot (2,1,2)
            plt.plot( mas_day, mas_BS, label = 'Модель Блэка - Шоуэлса', marker=".", markersize=1  )  # Файл по времени нельзя
            
            plt.title('Полная стоимость опционов =  ' + str (round (price_total,2)) + ' N_iter = ' + str (N_iter) )
            plt.ylabel("Цена")
            plt.xlabel("Дата")
            plt.grid()
            plt.show()
         
            
            #С = # цена опциона  
            # 252 торговых дня
        except BaseException:
            tkinter.messagebox.showinfo("Title", "a Tk MessageBox")
    
    
    def Calc_Monte_Carlo(self):
        # Решение задачи 1
        try:
            format = '%d.%m.%Y'
            if (self.b_read_file == FALSE):
                self.Read_task_1()

            value_start = datetime.strptime(v_start.get(), format)
            value_stop = datetime.strptime(v_stop.get(), format)                   
            value_period = (value_stop - value_start).days
            '''
            value_add = 2 # v_delta.get()
            value_min = 260 #float(v_min.get())
            '''
            # РАСЧЕТ ВОЛАТИЛЬНОСТИ
            std_signal = np.std(self.mas_Y)
            # среднегодовая волатильность
            min_date = min(self.mas_X)
            max_date = max(self.mas_X)
            delta_date = (max_date - min_date).days  # интервал в днях
            delta_year = delta_date/365
            std_year = std_signal/ (math.sqrt(delta_year))  # среднегодовая волатильность в абсолютных единицах
            mean = statistics.mean(self.mas_Y)    # среднее значение цены 
            
            std_year_proc = std_year / mean  # среднегодовая волатильность в процентах
            std_day = std_year/(math.sqrt(252))   # среднесуточная волатильность в абсолютных единицах
            
            price  = self.mas_Y[len(self.mas_Y)-1] # - текущая цена
            #strike = 260000 # - страйк опциона (цена к исполнению)
            Volume_buy = 1000 # обьем базовой закупки
            strike = Call_Strice(price, 2, 260)   # определение цены исполения 
            
            volatility = 0.3 # std_year_proc #  0.3 #  std_year_proc - волатильность  (ско)
            deltaToExpiry = 1/365 # - cрок опциона, лет
            riskFreeRate = 0.06 #- Безрисковая ставка
            N_iter = 1000  # количество итераций для поиска решения методом Монте-Карло
            N_day = value_period # 1      # количество дней для итераций
            
            #price = 260 # !!!! УБРАТЬ
            callp = CallPremiumBS(price,strike,volatility,deltaToExpiry*N_day,riskFreeRate) # Расчет премии методом Блэка - Шоулса
            
            # Метод Блэка - Шоулса
            mas_BS = [] # массив премии по методу Блэка - Шоулса
            #mas_BS.append(0)
            for  j in range(1, N_day+1):
                callp = Volume_buy * CallPremiumBS(price,strike,volatility,deltaToExpiry*j,riskFreeRate) # Расчет премии методом Блэка - Шоулса
                
                mas_BS.append(callp)

            # Метод Монте Карло
            # Очистка параметров
            mas_price_cur = [] # текущая цена
            mas_price_next = [] # следующая цена
            mas_premium = []
            mas_option = []
            mas_i = []
            mas_day = []
            mas_premium.clear()
            mas_price_next.clear() # цены на следующий день
            mas_price_cur.clear()  # цены на  текущий день
            mas_option.clear()
            mas_total_price = []
            mas_i.clear()
            avg_premium = 0
            price_option = 0
            v_temp = 0
           
            # заполнение стартового массива   
            
            for i in range(0, N_iter):    # итераций
                mas_price_cur.append(price)
                mas_price_next.append(0)
                mas_i.append(i)
                mas_total_price.append([])
                #mas_total_price[i].append(price)

            for  j in range(1, N_day+1):   # необходимо пройти по всем дням и для  каждого дня сгенерировать N вариантов
                
                mas_day.append(value_start + timedelta(days=j))
                mas_premium = []
                # генерируем массив из N вариантов цен на следующий день
                for i in range(0, N_iter):    # итераций
                    mas_price_next[i] = mas_price_cur[i] * math.exp(GenerateRandomReturn(riskFreeRate,volatility,deltaToExpiry))     
                    strike = 260
                    # расчет премии

                    if ((mas_price_next[i] - strike) >= 0):
                        premium = (mas_price_next[i] - strike) * Volume_buy
                    else:
                        premium = 0  
                    mas_premium.append(premium)
                '''
                for i in range(0, N_iter): 
                    strike = Call_Strice(mas_price_cur[i], 2, 260)  # определение цены исполнения
                    if ((mas_price_next[i] - strike) >= 0):
                        premium = (mas_price_next[i] - strike) * Volume_buy
                    else:
                        premium = 0  
                    mas_premium.append(premium)
                '''                
                avg_price = np.mean(mas_price_next)  # + среднее значение цены 
                avg_premium = np.mean(mas_premium)   # + среднее значение премии
                price_option = avg_premium*math.exp(-riskFreeRate*deltaToExpiry*j)   # дисконтированное среднее значение премии (это и есть теоретическая цена опциона)
                mas_option.append(price_option)
                mas_price_cur = mas_price_next

                for i in range(0, N_iter):  
                    mas_total_price[i].append(mas_price_cur[i])

            # вывод графиков
            i = 0
            fig, ax = plt.subplots()
                        
            plt.subplot (2,1,1)
            plt.plot( self.mas_X , self.mas_Y, label = 'Стоимость поставки', marker=".", markersize = 1  )  # Файл по времени нельзя 
            for i in range(0, 50): #N_iter
                plt.plot( mas_day , mas_total_price[i], label = 'Стоимость поставки' )  # Файл по времени нельзя 
            plt.title('Варианты развития СКО = ' + str (round (volatility,2)))
            
            plt.grid()

            price_total = np.sum(mas_option)
            textbox.insert(str(1.0), 'Полная стоимость опционов=  ' + str (round (price_total,2)) + '\n')
            plt.subplot (2,1,2)
            plt.plot( mas_day, mas_option, label = 'Стоимость поставки', marker=".", markersize=1  )  # Файл по времени нельзя 
            plt.plot( mas_day, mas_BS, label = 'Модель Блэка - Шоуэлса', marker=".", markersize=1  )  # Файл по времени нельзя
            
            plt.title('Полная стоимость опционов =  ' + str (round (price_total,2)) + ' N_iter = ' + str (N_iter) )
            plt.ylabel("Цена")
            plt.xlabel("Дата")
            plt.grid()
            plt.show()
         
            
            #С = # цена опциона  
            # 252 торговых дня
        except BaseException:
            tkinter.messagebox.showinfo("Title", "a Tk MessageBox")
    
    def Calc_task_2(self):
        try:
            value_miv_gkk = float(v_min_gkk.get())
            value_max_gkk = float(v_max_gkk.get())                    
            value_miv_skk = float(v_min_skk.get())
            value_max_skk = float(v_max_skk.get())
            x, y, z = makeData()
            fig = pylab.figure()
            axes = Axes3D(fig)
            
            a = len(x)
            b = len(y[0])
            mas = []
            for i in range(a):
                mas.append([])
                for j in range(b):
                    mas[i].append(0)

            z_n_min = value_miv_gkk
            z_n_max = value_max_gkk
            for i in range(len(x)-1,-1,-1):    # количество дней
                for j in range(len(y[0])-1,-1,-1): # количество вариантов забора в сутки
                    if (z[i][j] >= z_n_min) and (z[i][j] <= z_n_max) :
                        mas[i][j] = 5
                    else:
                        mas[i][j] = 10
                   
                z_n_min = z_n_min - value_max_skk
                z_n_max = z_n_max - value_miv_skk
            mas[0][0] = 0             

            sc = axes.scatter(x, y, z, c=mas, cmap= 'rainbow', s=5) # cmap='inferno'
            plt.xlabel("Время, дни")
            plt.ylabel("Суточный забор")
            #plt.legend(*sc.legend_elements())
            plt.colorbar(sc)
            plt.title("Возможные состояния забора газа")
            
            plt.grid()
            '''
            my_col = cm.jet(np.random.rand(z.shape[0],z.shape[1]))
            z_n_min = value_miv_gkk
            z_n_max = value_max_gkk
            for i in range(len(x)-1,-1,-1):    # количество дней
                for j in range(len(y[0])-1,-1,-1): # количество вариантов забора в сутки
                    #mas[i][j] = 0
                    if (z[i][j] >= z_n_min) and (z[i][j] <= z_n_max) :
                        my_col[i][j] = 2
                    else:
                        my_col[i][j] = 1
                   
                z_n_min = z_n_min - value_max_skk
                z_n_max = z_n_max - value_miv_skk
            my_col[0][0] = 0  
            surf = axes.plot_surface(x, y, z, rstride=1, cstride=1, facecolors = my_col, linewidth=0, antialiased=False)
            '''

            pylab.xlim(0, 365)  # сетка по оси X
            pylab.ylim(value_miv_skk, value_max_skk)  # сетка по оси Y
            pylab.show() 
            
                        
        except BaseException:
            tkinter.messagebox.showinfo("Title", "a Tk MessageBox")
    
    def Regressin_task3(self):
        try:
            self.mas_regr_x.clear()
            self.mas_regr_y.clear()
            self.mas_noise.clear()

            for i in range(0, len(self.mas_X)): 
                y= 3*math.log(self.mas_X[i]) + 3*math.sin(2*self.mas_X[i]) + 1
                noise = self.mas_Y[i] - y
                self.mas_regr_y.append(y)
                self.mas_regr_x.append(self.mas_X[i])
                self.mas_noise.append(noise)

            dx = self.mas_X[1] - self.mas_X[0]
            x =  self.mas_regr_x[len(self.mas_regr_x)-1]
            std_noise = np.std(self.mas_noise)
            while x < 15:
                x = x + dx
                y= 3*math.log(x) + 3*math.sin(2*x) + 1
                self.mas_regr_x.append(x) 
                self.mas_regr_y.append(y)
                self.mas_noise.append(None)
            
            self.mas_noise[len(self.mas_regr_x)-1]=0

            # Расчет нижней и верхней границы
            for i in range(len(self.mas_X), len(self.mas_regr_x)): 
                y_up = self.mas_regr_y[i] + 2 * std_noise
                y_down = self.mas_regr_y[i] - 2 * std_noise
                self.mas_x_level.append(self.mas_regr_x[i])
                self.mas_up.append(y_up)
                self.mas_down.append(y_down) 

           
        except BaseException:
            tkinter.messagebox.showinfo("Title", "a Tk MessageBox")    

    def Read_task3(self):
        try:
            
            self.mas_X.clear()
            self.mas_Y.clear()
            self.mas_regr_x.clear()
            self.mas_regr_y.clear()
            self.mas_down.clear()
            self.mas_up.clear()
            self.mas_x_level.clear()
            
            i_line = 0
            fn = tkinter.filedialog.Open(root, title = "Выберите исходный временной ряд ", filetypes = [('*.csv files', '*.csv')]).show()
            if fn == '':
                return
            
            textbox.delete('1.0', 'end') 
            self.path_file = fn
                    
            f = open(self.path_file, 'rt')  
            for i in range(0, 100): # записываем первые 100 строк из файла
                textbox.insert(str(i + 1.0), f.readline())
                n = textbox.size
            
            textbox.insert(str(1.0), self.path_file + '\n')
            textbox.insert(str(1.0), 'На форму выведены строки из файла ...\n')
            f.close()
            f_name = self.path_file.rfind('/')
            f = open(self.path_file, 'rt')
            for line in f:
                i_line = i_line + 1
                if (line != '\n'):
                    separat = define_split(line)  # сохраняем информацию о разделителе 0 - между значениями, 1 - между целой и десятичной частью 
                    str_spl = line.split(separat[0])   #  .split(';') ,
                    v_x = float(str_spl[0])
                    v_y = float(str_spl[1])
                    
                    self.mas_X.append(v_x)
                    self.mas_Y.append(v_y)
            
            self.Regressin_task3()

            fig, ax = plt.subplots()
            
            plt.subplot (2,1,1)
            plt.plot( self.mas_X , self.mas_Y, label = 'Регрессия y= 3*ln(x) + 3*sin(2*x) + 1'  )  # Файл по времени нельзя 
            plt.plot( self.mas_regr_x , self.mas_regr_y, c='r' , label = 'Регрессионная модель'  )  # Файл по времени нельзя 
            plt.plot( self.mas_x_level , self.mas_up, c = 'g' , linestyle='--', label = 'Верхняя граница'  )  # Файл по времени нельзя 
            plt.plot( self.mas_x_level , self.mas_down, c = 'g', linestyle='--' , label = 'Нижняя граница'  )  # Файл по времени нельзя 
            plt.title('Аналитическая регрессия y= 3*ln(x) + 3*sin(2*x) + 1')
            plt.grid()

            plt.subplot (2,1,2)
            plt.plot( self.mas_regr_x , self.mas_noise, label = 'Оставшийся шум'  )  # Файл по времени нельзя 
            plt.title('Оставшийся шум')
            plt.xlabel('X')
            plt.grid()
            plt.show()
            
        except BaseException:
            tkinter.messagebox.showinfo("Title", "a Tk MessageBox")

    def Task3_Regression(self):
        try:
            i_line = 0
            fn = tkinter.filedialog.Open(root, title = "Выберите исходный временной ряд ", filetypes = [('*.csv files', '*.csv')]).show()
            if fn == '':
                return
            textbox.delete('1.0', 'end') 
            self.path_file = fn
            f = open(self.path_file, 'rt')  
            for i in range(0, 1000): # записываем первые 1000 строк из файла
                textbox.insert(str(i + 1.0), f.readline())
                n = textbox.size
            
            textbox.insert(str(1.0), self.path_file + '\n')
            textbox.insert(str(1.0), 'На форму выведено строки из файла ...\n')
            f.close()
            f_name = self.path_file.rfind('/')
            f = open(self.path_file, 'rt')
            self.mas_X.clear()
            self.mas_Y.clear()
            for line in f:
                i_line = i_line + 1
                if (line != '\n'):
                    separat = define_split(line)  # сохраняем информацию о разделителе 0 - между значениями, 1 - между целой и десятичной частью 
                    str_spl = line.split(separat[0])   #  .split(';') ,
                    v_x = float(str_spl[0])
                    v_y = float(str_spl[1])
                    self.mas_X.append(v_x)
                    self.mas_Y.append(v_y)
            geneticParameters = generate_Initial_Parameters()
            self.fittedParameters, pcov = curve_fit(func_regr, self.mas_X, self.mas_Y, geneticParameters)
            print('Parameters', self.fittedParameters)
            print()
            
            graphWidth = 800
            graphHeight = 600
            ModelAndScatterPlot(graphWidth, graphHeight)

        except BaseException:
            tkinter.messagebox.showinfo("Title", "a Tk MessageBox")

def CallPremiumBS(price, strike, volatility, yearsToExpiry, riskFreeRate):
    #расчет внутренней стоимости опциона по методу Блэка-Шоулза
    # price - текущая цена
    # strike - страйк опциона (цена к исполнению)
    # volatility - волатильность  (ско)
    # yearsToExpiry - cрок опциона, лет
    # riskFreeRate - Безрисковая ставка
    if (yearsToExpiry == 0):
        value = np.max(0, price - strike)

    d1 = (math.log(price / strike) + (riskFreeRate + 0.5 * volatility * volatility) * yearsToExpiry) / (volatility * math.sqrt(yearsToExpiry))
    d2 = d1 - volatility * math.sqrt(yearsToExpiry)
    nd1 = norm.cdf(d1) # np.random.normal(d1)
    nd2 = norm.cdf(d2) # np.random.normal(d2)

    value = price * nd1 - strike * math.exp(-riskFreeRate * yearsToExpiry) * nd2
    return value

def Call_Strice(Price_current, add, price_max):
    # Определение цены исполнения
    
    # Вариант 1: максимальная цена

    return price_max

    # Вариант 2: минимум из {текущая цена + 2, max цена}
    '''price_v1 = Price_current + add

    if (price_v1 > price_max):
        return price_max
    else:
        return price_v1'''
    # определение цены страйка


def GenerateRandomReturn(expectedReturn , volatility, years): 
    
    # генерирование случайного значения цены в соответствии с ее моделью
    # expectedReturn - без рисковая ставка
    # volatility - волатильность  (ско)
    # волатильность  (ско)
    r = rnd.normalvariate(0.5,0.5) # rnd.random()
    normalRandomValue = rnd.normalvariate(0,1) # norm.ppf(r) # + random.normalvariate()

    value = (expectedReturn - 0.5 * volatility * volatility) * years + volatility * math.sqrt(years) * normalRandomValue
  
    return value



Task = Cls_task_(False)



def makeData():
    
    value_miv_skk = float(v_min_skk.get())
    value_max_skk = float(v_max_skk.get())
    
    x = np.arange(1, 366, 1)  # время  

    y = np.arange(value_miv_skk, value_max_skk, 0.000025)  # суточный забор
    ygrid, xgrid = np.meshgrid(y, x)  # создание сетки для по времени и суточному забору
    zgrid = xgrid * ygrid # сумарный забор за время
    for i in range(0, len(x)):    # количество дней
        for j in range(0,len(y)): # количество вариантов забора в сутки
            t = xgrid[i][j]
            v = ygrid[i][j]
            if t == 1:
                zgrid[i][j] = v
            else:
                zgrid[i][j] = zgrid[i-1][j] + v
    return xgrid, ygrid, zgrid
 
def func_regr(x, a, b, Offset, c, d ): 
    return  a * np.log(x*b) + Offset + c*np.sin(d*x)


def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") 
    
    val = func_regr(np.array(Task.mas_X), *parameterTuple)
    return np.sum((np.array(Task.mas_Y) - val) ** 2.0)

def generate_Initial_Parameters():
    
    maxX = max(Task.mas_X)
    minX = min(Task.mas_X)
    maxY = max(Task.mas_Y)
    minY = min(Task.mas_Y)

    parameterBounds = []
    parameterBounds.append([minX, maxX]) # параметры для a
    parameterBounds.append([minX, maxX]) # параметры для b
    parameterBounds.append([0.0, maxY])  # параметры для Offset
    parameterBounds.append([minX, maxX]) # параметры для c
    parameterBounds.append([minX, maxX]) # параметры для d
    
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=5)
    return result.x

def ModelAndScatterPlot(graphWidth, graphHeight):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    
    axes = f.add_subplot(2,1,1)
    axes.plot(Task.mas_X, Task.mas_Y, 'D', marker=".", markersize=1)

    xModel = np.array(Task.mas_X) 
    yModel = func_regr(xModel, *Task.fittedParameters)
    mas_noise = yModel.tolist() - np.array(Task.mas_Y)

    mas_regr_x = Task.mas_X
    dx = Task.mas_X[1] - Task.mas_X[0]
    x =  xModel[len(xModel)-1]
    std_noise = np.std(mas_noise) 
    sp =  mas_noise.tolist()
    
    while x < 15:
       x = x + dx
       mas_regr_x.append(x)
       sp.append(None)
    n = len(mas_regr_x)
    sp[n - 1]=0

    l = (15 - max(Task.mas_X)) / dx
    n = len(Task.mas_X) + int(l)

    xModel = np.linspace(min(Task.mas_X), 15, num = n)
    yModel = func_regr(xModel, *Task.fittedParameters)
    mas_up  = yModel + 2 * std_noise
    mas_down =  yModel - 2 * std_noise
    axes.plot(xModel, yModel, c = 'r')
    axes.plot(xModel, mas_up, c = 'g', linestyle='--', linewidth = 1)
    axes.plot(xModel, mas_down, c = 'g', linestyle='--', linewidth = 1)
    axes.set_xlabel('X') # ось Х
    axes.set_ylabel('Y') # ось Y
    strn = 'Регрессионное уравнение y= ' + str(round(Task.fittedParameters[0],2)) + '*ln(' + str(round(Task.fittedParameters[1],2)) + '*x) + ' +  str(round(Task.fittedParameters[3],2)) + "*sin("+str(round(Task.fittedParameters[4],2)) + '*x) + ' + str(round(Task.fittedParameters[2],2))
    plt.title(strn) 
    plt.grid()

    axes = f.add_subplot(2,1,2)
    Save_file(xModel, yModel, mas_up,mas_down,'yModel.csv','x;y;up;down') # сохраняем файл
    axes.plot(mas_regr_x, sp)
    plt.grid()
    plt.show()
    plt.close('all') 

def define_split(line):  # определение разделителя в строке
    try:
        str_spl = line.split(';')
        n_spl = len(str_spl)
        if n_spl > 1:
            return ";", ","
        else:
            return ",", "."
    except BaseException:
        tkinter.messagebox.showinfo("Title", "a Tk MessageBox")
    return ";", ","

def Save_file(mas_t, mas_v, mas_up, mas_down, file_name, header): # процедура для сохранения результатов
    try:

        f = open(file_name, 'w')  
        f.write(header+"\n")
        n = len(mas_t)
        for i in range(0, n):
            str_wr = str(mas_t[i]) + ";" + str(mas_v[i]) + ";" + str(mas_up[i]) + ";" + str(mas_down[i]) + '\n' 
            f.write(str_wr)
        f.close()
    except BaseException:
        tkinter.messagebox.showinfo("Title", "a Tk MessageBox")

def add_list(a, b):
   return list(map(lambda x, y: (x or 0) + (y or 0), a, b))

def Save_big_file(path_file, rec_write):
    try:
        f = open(path_file, 'w')  
        n = len(rec_write)
        i = 0 
        rec = 0
        str_wr = ""
        for j in range(0, n):
            str_wr = str_wr + str(j) + ";" 
        str_wr =str_wr +  '\n'
        f.write(str_wr) 
        
        while True:
            rec = 0
            str_wr = ""
            for j in range(0, n):
                if (len(rec_write[j])> i):
                    str_wr =str_wr + str(rec_write[j][i]) 
                    rec = rec + 1
                str_wr =str_wr + ";" 
            
            if (rec == 0):
                break

            str_wr =str_wr +  '\n'
            f.write(str_wr)
            i = i + 1
        f.close()       
    except BaseException:
        tkinter.messagebox.showinfo("Title", "a Tk MessageBox")

def form_Black_Shouls():
    try:
        children = Toplevel(root)
        children.title("Модель Блэка-Шоулса")
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        w = w//2 # середина экрана
        h = h//2 
        w = w - 200 # смещение от середины
        h = h - 200
        frm_BS_panel = Frame(children, height = 330, width = 350)       #    создание панели
        frm_BS_panel.pack(side = 'top',    fill = 'x')                
        lf_task1 = LabelFrame  (frm_BS_panel, text='Параметры модели')
        lf_task1.place(x = 5,   y = 5, width = 280,  height = 80, anchor="nw")
        
        lbl_file  = Label  (frm_BS_panel,text = 'Исходные данные:')
        lbl_file.place    (x = 10,    y = 30)    

        if (Task.b_read_file == TRUE):
            txt_input = Task.file_name
        else:    
            txt_input = '       --- '

        lbl_file  = Label  (frm_BS_panel,text = txt_input)
        lbl_file.place     (x = 120,    y = 30)    

        children.geometry('300x100+{}+{}'.format(w, h))
        children.minsize(width=200,height=100)
        Btn_BS  = Button (children,  text =  'Построить модель', anchor='center', command = Task.Calc_Black_Shoulsh)  
        Btn_BS.place   (x = 10, y = 60, width = 120, height = 20)

    except BaseException:
        tkinter.messagebox.showinfo("Title", "a Tk MessageBox")

def form_Monte_Carlo():
    try:
        children = Toplevel(root)
        children.title("Метод Монте-Карло")
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        w = w//2 # середина экрана
        h = h//2 
        w = w - 200 # смещение от середины
        h = h - 200
        
        frm_BS_panel = Frame(children, height = 330, width = 350)       #    создание панели
        frm_BS_panel.pack(side = 'top',    fill = 'x')                
        lf_task1 = LabelFrame  (frm_BS_panel, text='Параметры модели')
        lf_task1.place(x = 5,   y = 5, width = 280,  height = 80, anchor="nw")
        
        lbl_file  = Label  (frm_BS_panel,text = 'Исходные данные:')
        lbl_file.place    (x = 10,    y = 30)    

        if (Task.b_read_file == TRUE):
            txt_input = Task.file_name
        else:    
            txt_input = '       --- '

        lbl_file  = Label  (frm_BS_panel,text = txt_input)
        lbl_file.place     (x = 120,    y = 30)    

        children.geometry('300x100+{}+{}'.format(w, h))
        children.minsize(width=200,height=100)
        Btn_BS  = Button (children,  text =  'Построить модель', anchor='center', command = Task.Calc_Monte_Carlo)  
        Btn_BS.place   (x = 10, y = 60, width = 120, height = 20)

    except BaseException:
        tkinter.messagebox.showinfo("Title", "a Tk MessageBox")

def Create_menu(root_use):
    # Создание меню 
    mainmenu = Menu(root_use) 
    root_use.config(menu=mainmenu)

    calcmenu = Menu(mainmenu, tearoff=0)
    calcmenu.add_command(label="Волатильность")
    calcmenu.add_command(label="Блэк-Шоуэлс",
                        command=form_Black_Shouls)
    calcmenu.add_command(label="Монте-Карло", 
                        command=form_Monte_Carlo)

    helpmenu = Menu(mainmenu, tearoff=0)
    helpmenu.add_command(label="Помощь")
    helpmenu.add_command(label="О программе")

    mainmenu.add_command(label='Файл')
    mainmenu.add_cascade(label='Расчет',
                        menu=calcmenu)
    mainmenu.add_cascade(label="Справка",
                        menu=helpmenu)




root = Tk()


panel_model = Frame(root, height = 330, width = 350)       #    создание панели
textFrame   = Frame(root, height = 140, width = 350)       #    

panel_model.pack(side = 'top',    fill = 'x')                
textFrame.pack  (side = 'bottom', fill = 'both', expand = 1)

textbox   = Text(textFrame, font='Arial 11', wrap='word',height = 10, width = 35)   # создание текстбокса для чтения файла
scrollbar = Scrollbar(textFrame)

scrollbar['command'] = textbox.yview
textbox['yscrollcommand'] = scrollbar.set

textbox.pack(side = 'left', fill = 'both', expand = 1)
#textbox.pack(side = 'left')
scrollbar.pack(side = 'right', fill = 'y')

lf_task1 = LabelFrame  (panel_model, text='1. задача - стоимость финансового инструмента')
lf_task1.place(x = 5,   y = 10, width = 330,  height = 100, anchor="nw")

lf_filter = LabelFrame  (panel_model, text='2. задача - траектории отбора газ')
lf_filter.place(x = 5,   y = 120, width = 330,  height = 100, anchor="nw")

lf = LabelFrame(panel_model, text='3. задача - модель зависимости')
lf.place       (x = 5, y = 230, width = 330, height = 80, anchor="nw")

p2x = 5
p2y = 130
# 1. компненты формы для решения задача 1
Btn_read_task1  = Button (panel_model,  text =  'Чтение временного ряда'      , anchor='center', command = Task.Read_task_1)  
Btn_read_task1.place   (x = 10, y = 80, width = 150, height = 20)

Btn_calc_task1  = Button (panel_model,  text =  'Метод Монте-Карло' , anchor='center', command = form_Monte_Carlo)  
Btn_calc_task1.place   (x = 180, y = 80, width = 150, height = 20)

#Btn_calc_task1  = Button (panel_model,  text =  'Метод Монте-Карло' , anchor='center', command = Task.Calc_task_1)  
#Btn_calc_task1.place   (x = 180, y = 110, width = 150, height = 20)

label_start   = Label  (panel_model,  text = 'Старт:', font = "Courier 10")
label_start.place    (x = 10,    y = 30)

label_stop   = Label  (panel_model,  text = 'Стоп:', font = "Courier 10")
label_stop.place    (x = 10,    y = 50)

label_start   = Label  (panel_model,  text = 'Дельта:', font = "Courier 10")
label_start.place    (x = 180,    y = 30)

label_stop   = Label  (panel_model,  text = 'Минимум:', font = "Courier 10")
label_stop.place    (x = 180,    y = 50)

v_start = StringVar()
TB_start = Entry(panel_model, textvariable = v_start)
TB_start.place(x = 90,  y = 30, width = 70, height = 20,  anchor="nw")   
TB_start.insert(0,"01.10.2018")

v_stop = StringVar()
TB_stop = Entry(panel_model, textvariable = v_stop)
TB_stop.place(x = 90,  y = 50, width = 70, height = 20,  anchor="nw")   
TB_stop.insert(0,"31.03.2019")

v_delta = StringVar()
TB_delta = Entry(panel_model, textvariable = v_delta)
TB_delta.place(x = 260,  y = 30, width = 70, height = 20,  anchor="nw")   
TB_delta.insert(0,str(2))

v_min = StringVar()
TB_min = Entry(panel_model, textvariable = v_min)
TB_min.place(x = 260,  y = 50, width = 70, height = 20,  anchor="nw")   
TB_min.insert(0,str(260))

# 2. Компоненты формы для задачи 2

label_min_gkk   = Label  (panel_model,  text = 'Мin ГКК:', font = "Courier 10")
label_min_gkk.place    (x = 10,    y = 140)

label_max_gkk   = Label  (panel_model,  text = 'Мах ГКК:', font = "Courier 10")
label_max_gkk.place    (x = 10,    y = 160)

label_min_skk   = Label  (panel_model,  text = 'Мin СКК:', font = "Courier 10")
label_min_skk.place    (x = 180,    y = 140)

label_max_skk   = Label  (panel_model,  text = 'Мах СКК:', font = "Courier 10")
label_max_skk.place    (x = 180,    y = 160)

v_min_gkk = StringVar()
TB_min_gkk = Entry(panel_model, textvariable = v_min_gkk)
TB_min_gkk.place(x = 90,  y = 140, width = 70, height = 20,  anchor="nw")   
TB_min_gkk.insert(0,str(0.765))

v_max_gkk = StringVar()
TB_max_gkk = Entry(panel_model, textvariable = v_max_gkk)
TB_max_gkk.place(x = 90,  y = 160, width = 70, height = 20,  anchor="nw")   
TB_max_gkk.insert(0,str(1))

v_min_skk = StringVar()
TB_min_skk = Entry(panel_model, textvariable = v_min_skk)
TB_min_skk.place(x = 260,  y = 140, width = 70, height = 20,  anchor="nw")  
TB_min_skk.insert(0,str(0.0009589))

v_max_skk = StringVar()
TB_max_skk = Entry(panel_model, textvariable = v_max_skk)
TB_max_skk.place(x = 260,  y = 160, width = 70, height = 20,  anchor="nw")  
TB_max_skk.insert(0,str(0.002884))


Btn_read_2task  = Button (panel_model,  text =  'Трактории'      , anchor='center', command = Task.Calc_task_2)
Btn_read_2task.place   (x = 260, y = 190, width = 70, height = 20)

# Задача 3

Btn_read_task3 = Button (panel_model,  text = 'Вариант 1. Модель аналитическая', anchor='w', command = Task.Read_task3)
Btn_read_task3.place      (x = 10, y = 250, width = 250, height = 20) 

Btn_read_task3 = Button (panel_model,  text = 'Вариант 2. Модель регрессионная', anchor='w', command = Task.Task3_Regression) 
Btn_read_task3.place      (x = 10, y = 280, width = 250, height = 20) 

root.title(string='Решение тестовых заданий')


Create_menu(root)

root.mainloop ()
