import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

serial_name = "./serial_cat"
parallel_name = "./cuda_cat"

def complexPlot(dataF, name):
    fig, ax = plt.subplots()

    xf = df.x
    yf = df.y

    ax.plot(xf,yf)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(name)
    plt.savefig("{name}_plot.png")

# def magPlot(dataF):
#     fig, ax = plt.subplots()

#     xf = np.linspace(0,500,32)
#     yf = np.sqrt(df.x**2 + df.y**2)

#     ax.plot(xf,yf)
#     plt.grid()
#     plt.xlabel("Freq")
#     plt.ylabel("Magnitude")
#     plt.savefig("magPlot.png")
#     plt.show()


df = pd.read_csv(f"{serial_name}.csv")
complexPlot(df, serial_name)
df = pd.read_csv(f"{parallel_name}.csv")
complexPlot(df, parallel_name)
plt.show()
# magPlot(df)


