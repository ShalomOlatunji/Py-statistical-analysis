from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
data = iris.data.copy()

print(data.head())

print("\nCOLUMN INFORMATION")
data.info()
print("\nSUMMARY STATISTICS")
print(data.describe().round(3).to_string())

print("\nMEAN — average")
print(data.mean().round(3).to_string())

print("\nMEDIAN — middle value")
print(data.median().round(3).to_string())

print("\nMODE - most common value")
for column in data.columns:
    print(column, ":", data[column].mode().tolist())

print("\nSTANDARD DEVIATION")
print(data.std().round(3).to_string())

print("\nCORRELATION BETWEEN MEASUREMENTS")
print(data.corr().round(3).to_string())