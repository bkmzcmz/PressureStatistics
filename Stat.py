import time
from math import log
import matplotlib.pyplot as plt

N = 158
f = open('davl1.txt')
s = f.readline()
sys = []    #массив систолического давления
dias = []   #массив диастолического давления
for i in range(N):
    x, y = [int(m) for m in f.readline().split(',')]
    sys.append(x)
    dias.append(y)
    
max_sys, min_sys = max(sys), min(sys)
print('Максимальное значение систолического давления:', max_sys)
print('Минимальное значение систолического давления:', min_sys)
print('Размах выборки систолического давления:', max_sys - min_sys)
print('Среднее число интервалов:', 1 + log(N, 2))
print('Берем число интервалов: 7')
nx = 7
# Длина интервалов с округлением до 1 знака
k_x = round((max_sys - min_sys)/nx, 1)   
print('Получаем длину интервалов, равную:', k_x)
count_int_X = [0]*nx   #число вхождений в каждый интервал
middle_int_X = [0]*nx    #середина каждого интервала
print('Сами интервалы вы можете наблюдать ниже:')
for i in range(nx):
    if i==0:
        print('[',round(min_sys+i*k_x,1) , '; ',round(min_sys+(i+1)*k_x,1)  , ']', sep='', end='  ')
    elif i==nx-1:
        print('(',round(min_sys+i*k_x,1) , '; ',max_sys, ']', sep='', end='  ')
    else:
        print('(',round(min_sys+i*k_x,1) , '; ',round(min_sys+(i+1)*k_x,1)  , ']', sep='', end='  ')
print()
print('А значения середин каждого из интервалов равны:')
for i in range(nx):
    middle_int_X[i] = round(((min_sys+i*k_x) + (min_sys+(i+1)*k_x))/2, 1)
print(middle_int_X)
  
# Подсчет частоты вхождений СВ X в каждый интервал
for i in range(N):
    for j in range(nx):
        if (j==0) and (sys[i] <= min_sys + k_x):
            count_int_X[j] += 1
            break
        if (j==(nx-1)) and (min_sys + (nx-1)*k_x <= sys[i]):
            count_int_X[j] += 1
            break
        if ((min_sys+j*k_x < sys[i]) and (sys[i] <= min_sys+(j+1)*k_x)):
            count_int_X[j] += 1
            break
        
print('Частота вхождений в каждый из интервалов равна:')
for i in range(nx):
    if i==0:
        print('[',round(min_sys+i*k_x,1), '; ',round(min_sys+(i+1)*k_x,1), '] -- ', count_int_X[i], sep='')
    elif i==nx-1:
        print('(',round(min_sys+i*k_x,1), '; ',max_sys, '] -- ', count_int_X[i], sep='')
    else:
        print('(',round(min_sys+i*k_x,1), '; ',round(min_sys+(i+1)*k_x,1)  , '] -- ', count_int_X[i], sep='')

# Объединение последних двух интервалов 
if nx==7:
    print('Объединим два последних интервала, согласно необходимому условию применения критерия Пирсона:')
    count_int_X[nx-2] += count_int_X[nx-1]
    count_int_X[nx-1] = 0
    count_int_X.pop(nx-1)
    middle_int_X[nx-1] = 0
    middle_int_X.pop(nx-1)
    nx -= 1
    for i in range(nx):
        if i==0:
            print('[',round(min_sys+i*k_x,1), '; ',round(min_sys+(i+1)*k_x,1), '] -- ', count_int_X[i], sep='')
        elif i==nx-1:
            print('(',round(min_sys+i*k_x,1), '; ',max_sys, '] -- ', count_int_X[i], sep='')
        else:
            print('(',round(min_sys+i*k_x,1), '; ',round(min_sys+(i+1)*k_x,1)  , '] -- ', count_int_X[i], sep='')

# Расчет числовых характеристик СВ Х - систолического давления 
print('Несмещенная состоятельная оценка генерального среднего: Mean = ', end='')
s = 0
for i in range(nx):
    s += middle_int_X[i]*count_int_X[i]
mean_X = s/N
print(mean_X)

print('Cмещенная оценка генеральной дисперсии: D = ', end='')
s = 0
for i in range(nx):
    s += (middle_int_X[i]**2)*count_int_X[i]
s /= N
D_X = s - mean_X**2
print(D_X)

print('Исправленная оценка генеральной дисперсии: S2 = ', end='')
S2_X = (D_X*N)/(N-1)
print(S2_X)

print('Исправленное среднее квадратическое отклонение: S = ', end='')
msd_X = S2_X**0.5
print(msd_X)

print('Ассиметрия: A = ', end='')
s=0
for i in range(nx):
    s += (middle_int_X[i] - mean_X)**3
s /= N
A_X = s/(msd_X**3)
print(A_X)

print('Эксцесс: E = ', end='')
s=0
for i in range(nx):
    s += (middle_int_X[i] - mean_X)**4
s /= N
E_X = s/(msd_X**4) - 3
print(E_X)

# Задаю размеры полей графиков и работаю с их выводом
plt.figure(figsize = (13, 5))

#Полигон частот
plt.subplot(121)
plt.plot(middle_int_X, count_int_X)
plt.title('Полигон частот систолического давления')
plt.xlabel('Значение систолического давления')
plt.ylabel('Частота встречаемости')

#Гистограмма частот
plt.subplot(122)
plt.bar(middle_int_X, count_int_X)
plt.title('Гистограмма частот систолического давления')
plt.xlabel('Значение систолического давления')
plt.ylabel('Частота встречаемости')
plt.show()

print()

print('Подключим в наши рассуждения СВ Y:')

max_dias, min_dias = max(dias), min(dias)
print('Максимальное значение диастолического давления:', max_dias)
print('Минимальное значение диастолического давления:', min_dias)
print('Размах выборки диастолического давления:', max_dias - min_dias)
print('Среднее число интервалов:', 1 + log(N, 2))
print('Берем число интервалов: 8')
ny = 8
#длина интервалов с округлением до 1 знака
k_y = round((max_dias - min_dias)/ny, 1)   
print('Получаем длину интервалов, равную:', k_y)
print('Сами интервалы вы можете наблюдать ниже:')
count_int_Y = [0]*ny   #число вхождений в каждый интервал
middle_int_Y = [0]*ny    #середина каждого интервала
for i in range(ny):
    if i==0:
        print('[',round(min_dias+i*k_y,1) , '; ',round(min_dias+(i+1)*k_y,1)  , ']', sep='', end='  ')
    elif i==ny-1:
        print('(',round(min_dias+i*k_y,1) , '; ',max_dias, ']', sep='', end='  ')
    else:
        print('(',round(min_dias+i*k_y,1) , '; ',round(min_dias+(i+1)*k_y,1)  , ']', sep='', end='  ')
print()
print('А значения середин каждого из интервалов равны:')
for i in range(ny):
    middle_int_Y[i] = round(((min_dias+i*k_y) + (min_dias+(i+1)*k_y))/2, 1)
print(middle_int_Y)

# Подсчет частоты вхождений СВ Y в каждый интервал
for i in range(N):
    for j in range(ny):
        if (j==0) and (dias[i] <= min_dias + k_y):
            count_int_Y[j] += 1
            break
        if (j==(ny-1)) and (min_dias + (ny-1)*k_y <= dias[i]):
            count_int_Y[j] += 1
            break
        if ((min_dias+j*k_y < dias[i]) and (dias[i] <= min_dias+(j+1)*k_y)):
            count_int_Y[j] += 1
            break
        
print('Частота вхождений в каждый из интервалов равна:')
for i in range(ny):
    if i==0:
        print('[',round(min_dias+i*k_y,1), '; ',round(min_dias+(i+1)*k_y,1), '] -- ', count_int_Y[i], sep='')
    elif i==ny-1:
        print('(',round(min_dias+i*k_y,1), '; ',max_dias, '] -- ', count_int_Y[i], sep='')
    else:
        print('(',round(min_dias+i*k_y,1), '; ',round(min_dias+(i+1)*k_y,1)  , '] -- ', count_int_Y[i], sep='')

#Числовые характеристики СВ Y - диастолического давления
print('Несмещенная состоятельная оценка генерального среднего: Mean_Y = ', end='')
s = 0
for i in range(ny):
    s += middle_int_Y[i]*count_int_Y[i]
mean_Y = s/N
print(mean_Y)

print('Cмещенная оценка генеральной дисперсии: D_Y = ', end='')
s = 0
for i in range(ny):
    s += (middle_int_Y[i]**2)*count_int_Y[i]
s /= N
D_Y = s - mean_Y**2
print(D_Y)

print('Исправленная оценка генеральной дисперсии: S2_Y = ', end='')
S2_Y = (D_Y*N)/(N-1)
print(S2_Y)

print('Исправленное среднее квадратическое отклонение: S_Y = ', end='')
msd_Y = S2_Y**0.5
print(msd_Y)

# Начинаем работу с двумерной выборкой
print('Составим корелляционную таблицу:')

correl_tabel = [0]*nx
for i in range(nx):
    correl_tabel[i] = [0]*ny

#Распределяем пары двумерной выборки по корреляционной таблице
for i in range(N):
    for j in range(nx):
        flag = False
        if (j==0) and (sys[i] <= min_sys + k_x):
            for k in range(ny):
                if (k==0) and (dias[i] <= min_dias + k_y):
                    correl_tabel[j][k] += 1
                    flag = True
                    break
                elif (k==(ny-1)) and (min_dias + (ny-1)*k_y <= dias[i]):
                    correl_tabel[j][k] += 1
                    flag = True
                    break
                elif ((min_dias+k*k_y < dias[i]) and (dias[i] <= min_dias+(k+1)*k_y)):
                    correl_tabel[j][k] += 1
                    flag = True
                    break
        elif (j==(nx-1)) and (min_sys + (nx-1)*k_x <= sys[i]):
            for k in range(ny):
                if (k==0) and (dias[i] <= min_dias + k_y):
                    correl_tabel[j][k] += 1
                    flag = True
                    break
                elif (k==(ny-1)) and (min_dias + (ny-1)*k_y <= dias[i]):
                    correl_tabel[j][k] += 1
                    flag = True
                    break
                elif ((min_dias+k*k_y < dias[i]) and (dias[i] <= min_dias+(k+1)*k_y)):
                    correl_tabel[j][k] += 1
                    flag = True
                    break
        elif ((min_sys+j*k_x < sys[i]) and (sys[i] <= min_sys+(j+1)*k_x)):
            for k in range(ny):
                if (k==0) and (dias[i] <= min_dias + k_y):
                    correl_tabel[j][k] += 1
                    flag = True
                    break
                elif (k==(ny-1)) and (min_dias + (ny-1)*k_y <= dias[i]):
                    correl_tabel[j][k] += 1
                    flag = True
                    break
                elif ((min_dias+k*k_y < dias[i]) and (dias[i] <= min_dias+(k+1)*k_y)):
                    correl_tabel[j][k] += 1
                    flag = True
                    break
        if flag:
            break

# Вывод корреляционной таблицы (Х - строки, Y - столбцы)     
for i in range(nx):
    print(correl_tabel[i])

# Через центр распределения обязательно проходит теоретическая линия регрессии
print('Центр распределения: (',mean_X, '; ', mean_Y, ')', sep='')

# Поиск условных средних Y при фиксированном X
print('Найдем условные средние Y при фиксированном X=xi:')
condition_mean_Y = []
for i in range(nx):
    s, m = 0, 0
    for j in range(ny):
        s += correl_tabel[i][j]*middle_int_Y[j]
        m += correl_tabel[i][j]
    condition_mean_Y.append(s/m)    
    print('Условное среднее Y при X=', middle_int_X[i], ' равно: ', condition_mean_Y[i], sep='')

# Вывод графика эмпирической линии регрессии
plt.plot(middle_int_X, condition_mean_Y)
plt.title('Опытная линия регрессии')
plt.xlabel('Значение систолического давления')
plt.ylabel('Условное среднее диастолического давления')
plt.show()

# Найдем параметры теоретической линии регрении alpha и beta:
s=0
for i in range(nx):
    for j in range(ny):
        s += middle_int_X[i]*middle_int_Y[j]*correl_tabel[i][j]
mean_XY = s/N #выборочное среднее произведения СВ
print('Выборочное среднее произведения mean_XY =', mean_XY)
cov_XY = mean_XY - mean_X*mean_Y #коэффициент ковариации
print('Коэффициент ковариации cov_XY =', cov_XY)
beta = cov_XY/D_X
alpha = mean_Y - beta*mean_X
print('Бета и альфа соответственно равны:', beta, alpha)

# Задаем функцию теоретической линии регрессии y = b*x + a
def regrY(X, alpha, beta):
    return beta*X+alpha

# Задал множнество иксов и множество значений функции от этих иксов
iksy = [100, 200]
igreky = [regrY(100, alpha, beta), regrY(200, alpha, beta)]

# По выше заданным множествам (по сути две точки на плоскости)
# строю теоретическую линию регрессии поверх старой эмпирической
plt.plot(middle_int_X, condition_mean_Y)
plt.plot(iksy, igreky)
plt.title('Опытная и теоретическая линии регрессии')
plt.xlabel('Значение систолического давления')
plt.ylabel('Условное среднее диастолического давления')
plt.show()

# Находим коэффициент корреляции и оценивам тесноту СВ
print('Найдем коэффициет корреляции: r_XY=', end='')
r_XY = cov_XY/(msd_X*msd_Y)
print('Коэффициент корреляции r_XY =', r_XY)
if abs(r_XY)<=0.1:
    print('Следовательно, можем сделат вывод, что скорее всего зависимости нет')
elif 0.1<abs(r_XY)<=0.3:
    print('Следовательно, можем сделат вывод, что у нас слабый характер тесноты Y и X')
elif 0.3<abs(r_XY)<=0.5:
    print('Следовательно, можем сделат вывод, что у нас умеренный характер тесноты Y и X')
elif 0.5<abs(r_XY)<=0.7:
    print('Следовательно, можем сделат вывод, что у нас заметный характер тесноты Y и X')
elif 0.7<abs(r_XY)<=0.9:
    print('Следовательно, можем сделат вывод, что у нас высокий характер тесноты Y и X')
elif 0.9<abs(r_XY)<=0.999:
    print('Следовательно, можем сделат вывод, что у нас линейный характер тесноты Y и X')

# Завершение работы программы
print('Программа заверишит свою работу через 60 секунд')
time.sleep(60)
