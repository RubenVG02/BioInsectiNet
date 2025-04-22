import pandas as pd

# DataFrame de ejemplo
data = {
    'producto': ['A', 'B', 'C', 'D'],
    'precio': [120, 80, 150, 60],
    'descuento': [0.1, 0.2, 0.15, 0.1]
}
df = pd.DataFrame(data)

def aplicar_descuento(fila):
    precio_final = fila["precio"] * (1 - fila["descuento"])

    if precio_final >=100:
        precio_final *= 0.9

    return precio_final

# Aplicar descuento

df["precio_final"] = df.apply(aplicar_descuento, axis=1)

print(df)