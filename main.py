import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import openpyxl
import seaborn as see
#
#
# # x = [21,27,33,15,35]
# # ticki = ['A','B','C','D','E']
# # nazwy = ['21','27','33','15','35']
# # kolory =['red','cyan','#FA9DA8','lightblue','#A51F6D']
# # plt.pie(x, colors=kolory,labels=nazwy,startangle=90)
# # plt.title('Tytuł')
# # plt.legend(loc=2, labels=ticki)
# # plt.savefig('wykres.pdf')
# # plt.show()
#
# exel = pd.ExcelFile('ceny.xlsx')
# data = pd.read_excel(exel, header=0)
# df = pd.DataFrame(data)
# df = df[df['Rok']==2017]
# df = (df.groupby(['Rodzaje towarów i usług']).agg({'Wartosc': ['mean']}))
# print(df)
# # print(df)
# # exel = np.linspace(0,2,100)
# plt.plot(df, label="liniowa")
# plt.show()


#zadanie1

x = np.arange(-10, 10, 0.1)
fig, ax1 = plt.subplots()

y1 = x ** 2 + 2 * x - 4
y2 = -x ** 3 + x - 2

ax1.plot(x, y1, label="x^2+2x-4", linestyle=":")
ax1.plot(x, y2, label="-x^3+x-2", linestyle="-.")
ax1.set_title("Parę wykresów")
ax1.set_ylim(-500, 500)
ax1.set_xlim(-15, 15)
ax1.set_ylabel("oś pionowa po lewej stronie")
ax1.set_xlabel("oś pozioma")
ax1.grid()
ax1.legend()
ax2 = ax1.twinx()
y3 = 2 * np.cos(x + 5)
ax2.plot(x, y3, color="green", linestyle="--", label="2*cos(x+5)")
ax2.legend(loc=4)
ax2.set_ylim(-3, 3)
ax2.set_ylabel("oś pionowa po prawej stronie")
plt.tight_layout()
plt.savefig("zad1.jpg", format="jpg")
plt.show()


#zadanie2
dane = pd.read_excel("ceny.xlsx")  ## wymaga openpyxl
unique = np.unique(dane["Rodzaje towarów i usług"])
prod1 = dane[dane["Rodzaje towarów i usług"] == unique[0]]
prod2 = dane[dane["Rodzaje towarów i usług"] == unique[1]]
#wyświetl  na  konsoli  średnią  cenę  poszczególnych  produktów  z  poszczególnych  półroczy
prod1_1 = prod1.iloc[:6, :]
prod1_2 = prod1.iloc[6:, :]
print(np.mean(prod1_1["Wartosc"]))
print(np.mean(prod1_2["Wartosc"]))

prod2_1 = prod2.iloc[:6, :]
prod2_2 = prod2.iloc[6:, :]
print(np.mean(prod2_1["Wartosc"]))
print(np.mean(prod2_2["Wartosc"]))
#stwórz  wykres  liniowy  prezentujący  dane  zawarte  w  ramce  danych
x = np.arange(12)
y1 = prod1["Wartosc"]
y2 = prod2["Wartosc"]

plt.plot(x, y1, label=unique[0])
plt.plot(x, y2, label=unique[1])
lab = prod1["Miesiące"]
plt.xticks(x, labels=lab, rotation=45)
plt.legend()
plt.grid()
plt.xlabel("Miesiące")
plt.ylabel("Ceny z zł")
plt.title("Wykres cen w 2016 roku")
plt.tight_layout()
plt.savefig("zad2.png", format="png")
plt.show()

#zadanie3
dane = pd.read_csv("wynagrodzenia.csv", sep="#", header=None, index_col=0)
dane2 = dane.T

r16 = dane2[dane2["Rok"] == "2016"]
r17 = dane2[dane2["Rok"] == "2017"]
r18 = dane2[dane2["Rok"] == "2018"]

lab = r16["Nazwa"]
x = np.arange(len(lab))
w16 = r16["Wartosc"].str.replace(',', '.').astype(float)
w17 = r17["Wartosc"].str.replace(',', '.').astype(float)
w18 = r18["Wartosc"].str.replace(',', '.').astype(float)
plt.bar(x - .25, w16, width=0.25)
plt.bar(x, w17, width=0.25)
plt.bar(x + 0.25, w16, width=0.25)
plt.xticks(x, lab, rotation=90)
plt.tight_layout()
plt.savefig("zad3.png", format="png")
plt.show()


#wykres pioziomy słupkowy
x = np.arange(5)
plt.subplot(2, 1, 1)
w1 = [-30, -10, -20, -50, -70]
c1 = ["purple", "darkgoldenrod", "indigo", "lime", "fuchsia"]
plt.barh(x, w1, color=c1)
plt.xlim(-70, 0)
e1 = [6, 10, 8, -2, 2]
plt.yticks(x, e1)
plt.title("Tytuł1")
#wykres pionowy slupkowy z mieszanymi kolorami
plt.subplot(2, 1, 2)
w2 = [22, 12, 28, 11, 50]
c2 = ["orange", "deepskyblue", "plum", "darkslateblue", "cyan"]
plt.bar(x, w2, color=c2)
w3 = [26, 30, 14, 27, 25]
c3 = ["hotpink", "aquamarine", "slategrey", "blueviolet", "brown"]
plt.bar(x, w3, bottom=w2, color=c3)
plt.title("Tytuł")
plt.ylim(0, 150)
e2 = ["A", "B", "C", "D", "E"]
plt.xticks(x, e2)
plt.tight_layout()
plt.show()

#wykres liniow
x = np.arange(0.01, 15, 0.01)  # nie od zera bo logarytm w zerze nie ejst okreslony
y1 = np.log(x)
fig, ax1 = plt.subplots()
ax1.plot(x, y1, color="g")
ax1.set_title("Dwa wykresy")
ax1.set_ylabel('ln(x)', color='g')
ax1.set_xlabel('x')
ax1.tick_params('y', colors='g')
ax2 = ax1.twinx()
y2 = x ** 2 + x
ax2.plot(x, y2, "--", color="r")
ax2.tick_params('y', colors='r')
ax2.set_ylabel('x^2+x', color='r')
fig.tight_layout()
plt.show()
#slupokowy z odbiciem
s1=[51, -78, 50,-40,58]
s2=[175, 52, -12, -30, 111]
s3= [62, -70, 30, 182, 0]
x=np.arange(5)

plt.bar(x,s2, width=0.25, color="green", label="B")
plt.bar(x+0.25,s3, width=0.25, color="pink", label="C")
plt.bar(x-0.25,s1, width=0.25, color="olive", label="A")

plt.legend(loc=1)
plt.title("Wykres")
plt.ylabel("Podpis osi y")
plt.xlabel("Podpis osi x")
plt.grid()
plt.savefig("wykres.jpg",format="jpg")
plt.show()


#2 wykresy slupokowe poziome
x = np.arange(5)
plt.subplot(3, 2, 1)
w1 = np.array([-30, -11, -20, -50, -70])
k1 = ['violet', 'darkmagenta', 'yellowgreen', 'lawngreen', 'beige']
plt.barh(x, w1, color=k1)
plt.title("Tytuł1")
plt.yticks(x, [6, 10, 8, -2, 2])
plt.tight_layout()
plt.subplot(3, 2, 6)
w2 = w1 * (-1)
k1 = ['indigo', 'hotpink', 'crimson', 'darkred', 'mediumvioletred']
plt.barh(x, w2, color=k1)
plt.title("Tytuł2")
plt.yticks(x, [6, 10, 8, -2, 2])
plt.tight_layout()
plt.show()

