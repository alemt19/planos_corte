import numpy as np
from scipy.optimize import linprog

class PlanoDeCorte:
    """
    Resuelve un problema de programación lineal entera utilizando el método de planos de corte de Gomory.
    """

    def __init__(self, c, A, b):
        """
        Inicializa el problema de programación lineal.

        Args:
            c (list): Coeficientes de la función objetivo (para maximizar, usar signos negativos).
            A (list of lists): Matriz de coeficientes de las restricciones.
            b (list): Vector de términos independientes de las restricciones.
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.iteracion = 0

    def es_entero(self, solucion):
        """Verifica si la solución es entera."""
        return np.all(np.abs(solucion - np.round(solucion)) < 1e-5)

    def resolver(self):
        """
        Aplica el método de planos de corte para encontrar la solución entera óptima.
        """
        while True:
            self.iteracion += 1
            print(f"\n--- Iteración {self.iteracion} ---")

            # Resuelve la relajación del problema de PL
            resultado = linprog(c=-self.c, A_ub=self.A, b_ub=self.b, method='highs')

            if not resultado.success:
                print("El problema de PL no pudo ser resuelto en esta iteración.")
                return None

            solucion = resultado.x
            print(f"Solución actual: {solucion}")
            print(f"Valor objetivo: {-resultado.fun}")

            # Verifica si la solución es entera
            if self.es_entero(solucion):
                print("\n¡Se ha encontrado una solución entera!")
                return solucion, -resultado.fun

            # Genera el corte de Gomory
            fila_fraccionaria_idx = -1
            max_frac = -1
            for i in range(len(solucion)):
                fraccion = solucion[i] - np.floor(solucion[i])
                if fraccion > 1e-5:
                    if fraccion > max_frac:
                        max_frac = fraccion
                        fila_fraccionaria_idx = i

            if fila_fraccionaria_idx == -1:
                print("\nNo se encontraron más variables fraccionarias. La solución es óptima.")
                return solucion, -resultado.fun


            # Extrae la fila del tableau simplex (en este caso, de las restricciones)
            # Para simplificar, generamos el corte a partir de una de las restricciones originales
            # que esté activa y tenga una variable básica fraccionaria.
            # Una implementación más robusta requeriría el tableau simplex completo.

            # Identificamos la restricción que genera la variable fraccionaria
            # (Enfoque simplificado para este ejemplo)
            try:
                # Invertimos la matriz de restricciones activas para obtener el tableau
                A_inv = np.linalg.inv(self.A)
                tableau_fila = A_inv[fila_fraccionaria_idx]
            except np.linalg.LinAlgError:
                print("No se pudo generar el corte de Gomory por un problema con la matriz.")
                return None


            parte_fraccionaria_coef = tableau_fila - np.floor(tableau_fila)
            parte_fraccionaria_b = solucion[fila_fraccionaria_idx] - np.floor(solucion[fila_fraccionaria_idx])

            # Crea la nueva restricción de corte
            corte = -parte_fraccionaria_coef
            nuevo_b = -parte_fraccionaria_b

            # Añade la nueva restricción al problema
            self.A = np.vstack([self.A, corte])
            self.b = np.append(self.b, nuevo_b)

            print(f"Añadiendo corte de Gomory para la variable x{fila_fraccionaria_idx+1}")
            print(f"Nueva restricción: {corte} <= {nuevo_b}")

if __name__ == '__main__':
    # Solicita los datos al usuario
    print("Resolución de Problemas de Programación Entera con Planos de Corte")
    print("------------------------------------------------------------------")

    try:
        num_vars = int(input("Introduce el número de variables de decisión: "))
        num_restricciones = int(input("Introduce el número de restricciones: "))

        c = []
        print("\nIntroduce los coeficientes de la función objetivo (para maximizar):")
        for i in range(num_vars):
            c.append(float(input(f"Coeficiente para x{i+1}: ")))

        A = []
        print("\nIntroduce los coeficientes de las restricciones (matriz A):")
        for i in range(num_restricciones):
            fila = []
            for j in range(num_vars):
                fila.append(float(input(f"Coeficiente de x{j+1} en la restricción {i+1}: ")))
            A.append(fila)

        b = []
        print("\nIntroduce los términos independientes de las restricciones (vector b):")
        for i in range(num_restricciones):
            b.append(float(input(f"Término para la restricción {i+1}: ")))

        # Crea y resuelve el problema
        problema = PlanoDeCorte(c, A, b)
        solucion_optima, valor_optimo = problema.resolver()

        if solucion_optima is not None:
            print("\n--- Resultados Finales ---")
            print(f"Solución Óptima Entera: {np.round(solucion_optima).astype(int)}")
            print(f"Valor Óptimo de la Función Objetivo: {valor_optimo}")

    except ValueError:
        print("\nError: Por favor, introduce solo valores numéricos.")
    except Exception as e:
        print(f"\nHa ocurrido un error inesperado: {e}")